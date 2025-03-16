# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
import cv2
import numpy as np
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanV2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
    ):
        r"""
        Initializes the video-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`, *optional*, defaults to 0):
                GPU device ID to use
            rank (`int`, *optional*, defaults to 0):
                Process rank in distributed environment
            t5_fsdp (`bool`, *optional*, defaults to False):
                Whether to use FSDP for text encoder
            dit_fsdp (`bool`, *optional*, defaults to False):
                Whether to use FSDP for diffusion transformer
            use_usp (`bool`, *optional*, defaults to False):
                Whether to use usp for context parallel
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to keep text encoder on CPU
            init_on_cpu (`bool`, *optional*, defaults to True):
                Whether to initialize model weights on CPU
        """
        self.device = f"cuda:{device_id}"
        self.param_dtype = torch.float16
        self.t5_fsdp = t5_fsdp
        self.dit_fsdp = dit_fsdp
        self.use_usp = use_usp
        self.sp_size = config.sp_size if hasattr(config, 'sp_size') else 1
        self.num_train_timesteps = 1000
        self.sample_neg_prompt = config.sample_neg_prompt if hasattr(
            config, 'sample_neg_prompt') else ""
        self.vae_checkpoint = config.vae_checkpoint
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.t5_cpu = t5_cpu

        # 设置shard函数
        shard_fn = partial(shard_model, device_id=device_id)
        
        # t5 encoder
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu') if t5_cpu else self.device,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer)
            if hasattr(config, 't5_tokenizer') else config.t5_model,
            shard_fn=shard_fn if t5_fsdp else None,
        )
        
        # vae
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)
            
        # Visual Model CLIP (如果配置中有)
        if hasattr(config, 'clip_model'):
            self.use_clip = True
            self.clip = CLIPModel(
                dtype=config.clip_dtype,
                device=self.device,
                checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
                tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer) 
                if hasattr(config, 'clip_tokenizer') else config.clip_model
            )
        else:
            self.use_clip = False
            
        # DIT
        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)
        
        # 如果使用USP
        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)

    def get_video_frames(self, input_video_path, video_length, h, w, fps=None):
        """
        读取视频文件并提取帧，并根据VAE的时间压缩比调整帧数
        
        Args:
            input_video_path (str): 输入视频的路径
            video_length (int): 需要的视频帧数
            h (int): 目标高度
            w (int): 目标宽度
            fps (int, optional): 目标帧率，如果为None则使用原视频帧率
            
        Returns:
            torch.Tensor: 视频帧张量，形状为[3, video_length, h, w]
            torch.Tensor: 视频掩码张量，如果不需要遮罩则全为1
        """
        # 根据VAE的时间压缩比调整视频长度
        video_length = int((video_length - 1) // self.vae_stride[0] * self.vae_stride[0]) + 1 if video_length != 1 else 1
        
        if isinstance(input_video_path, str):
            cap = cv2.VideoCapture(input_video_path)
            input_video = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = 1 if fps is None else int(original_fps // fps)

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame = cv2.resize(frame, (w, h))
                    input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                frame_count += 1

            cap.release()
        else:
            input_video = input_video_path

        # 确保没有超出所需的帧数
        input_video = np.array(input_video)[:video_length]
        actual_frames = input_video.shape[0]
        
        # 如果实际帧数小于所需帧数，则复制最后一帧
        if actual_frames < video_length:
            last_frame = input_video[-1:]
            padding = np.tile(last_frame, (video_length - actual_frames, 1, 1, 1))
            input_video = np.concatenate([input_video, padding], axis=0)
            
        # 转换为torch张量，并规范化到[-1, 1]
        input_video = torch.from_numpy(input_video)
        # 调整维度顺序：[frames, height, width, channels] -> [channels, frames, height, width]
        input_video = input_video.permute(3, 0, 1, 2) / 127.5 - 1
        
        # 创建视频掩码（所有像素都有效）
        input_video_mask = torch.ones_like(input_video[0:1])
        
        return input_video, input_video_mask

    def generate(self,
                 input_prompt,
                 video_path,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 denoise_strength=0.7,
                 seed=-1,
                 fps=None,
                 offload_model=True):
        r"""
        从输入视频和文本提示中生成新的视频，通过扩散过程重新绘制视频内容。

        Args:
            input_prompt (`str`):
                用于内容生成的文本提示。
            video_path (`str`):
                输入视频的路径。
            max_area (`int`, *optional*, defaults to 720*1280):
                潜在空间计算的最大像素面积。控制视频分辨率缩放。
            frame_num (`int`, *optional*, defaults to 81):
                从视频中采样的帧数。数字应为4n+1。
            shift (`float`, *optional*, defaults to 5.0):
                噪声调度移位参数。影响时间动态。
                [注意]: 如果你想生成480p视频，建议将shift值设为3.0。
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                用于采样视频的求解器。
            sampling_steps (`int`, *optional*, defaults to 40):
                扩散采样步骤的数量。较高的值提高质量但会减慢生成速度。
            guide_scale (`float`, *optional*, defaults 5.0):
                无分类器引导比例。控制提示遵从性与创造性。
            n_prompt (`str`, *optional*, defaults to ""):
                用于内容排除的负面提示。如果未给出，则使用`config.sample_neg_prompt`。
            denoise_strength (`float`, *optional*, defaults to 0.7):
                降噪强度，控制对原始视频的修改程度。值越大，生成的视频与原始视频差异越大。
            seed (`int`, *optional*, defaults to -1):
                噪声生成的随机种子。如果为-1，则使用随机种子。
            fps (`int`, *optional*, defaults to None):
                目标视频的帧率。如果为None，则使用原始视频的帧率。
            offload_model (`bool`, *optional*, defaults to True):
                如果为True，则在生成过程中将模型卸载到CPU以节省VRAM。

        Returns:
            torch.Tensor:
                生成的视频帧张量。维度: (C, N, H, W)，其中:
                - C: 颜色通道(3 for RGB)
                - N: 帧数
                - H: 帧高度(来自max_area)
                - W: 帧宽度(来自max_area)
        """
        F = frame_num
        
        # 计算适当的视频尺寸
        if isinstance(video_path, str):
            cap = cv2.VideoCapture(video_path)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            aspect_ratio = h / w
            cap.release()
        else:
            aspect_ratio = 9 / 16  # 默认16:9比例
            
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        # 确保帧数符合VAE的时间压缩要求
        F = int((F - 1) // self.vae_stride[0] * self.vae_stride[0]) + 1 if F != 1 else 1
        
        # 计算潜在帧数
        latent_frames = (F - 1) // self.vae_stride[0] + 1
        
        # 确保潜在帧数能被patch_size整除
        if self.patch_size[0] > 1 and latent_frames % self.patch_size[0] != 0:
            additional_frames = self.patch_size[0] - latent_frames % self.patch_size[0]
            F += additional_frames * self.vae_stride[0]
            latent_frames += additional_frames
        
        max_seq_len = latent_frames * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        # 获取视频帧
        input_video, input_video_mask = self.get_video_frames(
            video_path, F, h, w, fps=fps)
        input_video = input_video.to(self.device)
        input_video_mask = input_video_mask.to(self.device)

        # 设置种子
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # 预处理文本
        # 当使用FSDP时，必须在GPU上处理文本
        if self.t5_fsdp:
            # 如果使用FSDP，必须在GPU上处理
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            # 注意：使用FSDP时不能将模型移回CPU
        elif self.t5_cpu:
            # 如果启用t5_cpu但没有启用FSDP
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
        else:
            # 正常情况，可以在GPU上处理后再移回CPU
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model and not self.t5_fsdp:
                self.text_encoder.model.cpu()
                
        # 修复：确保context和context_null是兼容的格式
        # T5EncoderModel.__call__方法只返回一个元素，我们需要转换成WanModel期望的格式
        # 检查context是否为列表且只包含一个元素
        if isinstance(context, list) and len(context) == 1 and isinstance(context[0], torch.Tensor):
            # 创建适用于WanModel的格式
            context_tensor = context[0]
            context_mask = torch.ones((1, context_tensor.size(0)), dtype=torch.bool, device=self.device)
            # 转换为[tensor, tensor, mask]格式
            context = [context_tensor, context_tensor, context_mask]
            
        if isinstance(context_null, list) and len(context_null) == 1 and isinstance(context_null[0], torch.Tensor):
            # 对context_null做同样的处理
            context_null_tensor = context_null[0]
            context_null_mask = torch.ones((1, context_null_tensor.size(0)), dtype=torch.bool, device=self.device)
            # 转换为[tensor, tensor, mask]格式
            context_null = [context_null_tensor, context_null_tensor, context_null_mask]
        
        # 编码视频 - 分批处理以减少显存使用
        # 检查WanVAE是否有model属性，如果有，移动该model
        if hasattr(self.vae, 'model'):
            self.vae.model.to(self.device)
        # 也可能有encoder和decoder属性
        if hasattr(self.vae, 'encoder'):
            self.vae.encoder.to(self.device)
        if hasattr(self.vae, 'decoder'):
            self.vae.decoder.to(self.device)
        # 或者，如果WanVAE具有自定义的device属性
        if hasattr(self.vae, 'device'):
            self.vae.device = self.device
        
        # 分批编码视频以减少显存使用
        batch_size = 4  # 可以根据显存大小调整
        latents_list = []
        
        for i in range(0, input_video.shape[1], batch_size):
            end_idx = min(i + batch_size, input_video.shape[1])
            batch_frames = input_video[:, i:end_idx]
            with torch.no_grad():
                batch_latents = self.vae.encode([batch_frames])[0]
            latents_list.append(batch_latents)
            
            # 清理显存
            del batch_frames
            torch.cuda.empty_cache()
            
        latents = torch.cat(latents_list, dim=1)
        
        # 确保潜在变量维度符合模型要求
        if self.patch_size[0] > 1 and latents.shape[1] % self.patch_size[0] != 0:
            pad_size = self.patch_size[0] - (latents.shape[1] % self.patch_size[0])
            # 复制最后几帧进行填充
            padding = latents[:, -pad_size:]
            latents = torch.cat([latents, padding], dim=1)
        
        # 如果启用了offload_model，将VAE移回CPU
        if offload_model:
            if hasattr(self.vae, 'model'):
                self.vae.model.cpu()
            if hasattr(self.vae, 'encoder'):
                self.vae.encoder.cpu()
            if hasattr(self.vae, 'decoder'):
                self.vae.decoder.cpu()
            torch.cuda.empty_cache()
        
        # 基于降噪强度添加噪声
        noise = torch.randn(
            latents.shape,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)
        
        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # 评估模式
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps = retrieve_timesteps(sample_scheduler,
                                              sigmas=sampling_sigmas,
                                              device=self.device)

            # 根据降噪强度计算起始步骤 - 修正映射逻辑
            start_step = int(denoise_strength * self.num_train_timesteps)
            if denoise_strength <= 0.01:  # 如果降噪强度几乎为0，直接返回原始视频
                return input_video
                
            # 创建新的schedular用于添加噪声
            noise_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            
            # 添加噪声到latents
            start_timestep = torch.tensor([start_step], device=self.device)
            latents_noisy = noise_scheduler.add_noise(latents, noise, start_timestep)
            
            # 计算对应的时间步索引
            timestep_indices = []
            for t in timesteps:
                diff = torch.abs(t - start_step)
                if diff <= self.num_train_timesteps / sampling_steps:
                    timestep_indices.append(True)
                else:
                    timestep_indices.append(False)
            
            # 取需要处理的时间步
            working_timesteps = timesteps[timestep_indices]
            
            # 如果没有时间步需要处理，取最接近的几个
            if len(working_timesteps) == 0:
                t_start = max(1, int(denoise_strength * len(timesteps)))
                working_timesteps = timesteps[-t_start:]
            
            # 将模型移到GPU
            self.model.to(self.device)
            
            # 降噪循环
            latents_sample = latents_noisy
            for i, t in tqdm(
                enumerate(working_timesteps), 
                desc="V2V Denoising",
                total=len(working_timesteps)):
                # 在每个去噪步骤中，获取调节
                latent_model_input = torch.cat([latents_sample] * 2)
                
                # 预测噪声残差
                # 由于掩码尺寸不匹配问题，我们完全不使用掩码，改为全部使用context_mask=None的方式处理
                if len(context) >= 1 and len(context_null) >= 1:
                    # 确保至少有一个元素可以连接
                    noise_pred = self.model(
                        input_latents=latent_model_input,
                        timestep=t,
                        context=[
                            torch.cat([context_null[0], context[0]])
                        ],
                        context_mask=None).sample
                    
                    # 记录使用了无掩码方式
                    if i == 0:  # 只在第一次迭代记录
                        logging.info("使用无掩码方式进行去噪生成")
                else:
                    # 如果连第一个元素都没有，这是非常异常的情况
                    logging.error("严重错误：context或context_null为空，无法进行生成")
                    raise ValueError("无效的context数据")
                
                # 执行调节
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_text - noise_pred_uncond)
                
                if sample_solver == 'unipc':
                    latents_sample = sample_scheduler.step(
                        noise_pred, t, latents_sample).prev_sample
                elif sample_solver == 'dpm++':
                    step_result = sample_scheduler.step(
                        noise_pred, t, latents_sample, step_index=i)
                    latents_sample = step_result.prev_sample
                
                # 清理
                if offload_model:
                    del noise_pred, latent_model_input
                    if i < len(working_timesteps) - 1:
                        torch.cuda.empty_cache()
            
            # 将模型移回CPU以节省显存 - 只有当不使用FSDP时才移回CPU
            if offload_model and not self.dit_fsdp:
                self.model.cpu()
                torch.cuda.empty_cache()
            
            # 将VAE移到GPU进行解码
            if hasattr(self.vae, 'model'):
                self.vae.model.to(self.device)
            if hasattr(self.vae, 'encoder'):
                self.vae.encoder.to(self.device)
            if hasattr(self.vae, 'decoder'):
                self.vae.decoder.to(self.device)
            
            # 分批解码以减少显存使用
            video_chunks = []
            chunk_size = 4  # 可以根据显存大小调整
            
            for i in range(0, latents_sample.shape[1], chunk_size):
                end_idx = min(i + chunk_size, latents_sample.shape[1])
                chunk_latents = latents_sample[:, i:end_idx]
                with torch.no_grad():
                    chunk_video = self.vae.decode([chunk_latents])[0]
                video_chunks.append(chunk_video)
                
                # 清理显存
                del chunk_latents, chunk_video
                torch.cuda.empty_cache()
            
            # 合并视频块
            video = torch.cat(video_chunks, dim=1)
            
            # 将所有模型卸载到CPU
            if offload_model:
                if hasattr(self.vae, 'model'):
                    self.vae.model.cpu()
                if hasattr(self.vae, 'encoder'):
                    self.vae.encoder.cpu()
                if hasattr(self.vae, 'decoder'):
                    self.vae.decoder.cpu()
                if not self.dit_fsdp:
                    self.model.cpu()
                if not self.t5_fsdp:
                    self.text_encoder.model.cpu()
                if hasattr(self, 'clip') and self.use_clip:
                    self.clip.model.cpu()
                torch.cuda.empty_cache()
        
        return video 