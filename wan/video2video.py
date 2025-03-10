# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
import cv2
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import get_video_frames


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
        初始化 Wan 视频到视频转换模型组件。

        Args:
            config (EasyDict):
                包含从 config.py 初始化的模型参数的对象
            checkpoint_dir (`str`):
                包含模型检查点的目录路径
            device_id (`int`, *optional*, defaults to 0):
                目标 GPU 设备 ID
            rank (`int`, *optional*, defaults to 0):
                分布式训练的进程排名
            t5_fsdp (`bool`, *optional*, defaults to False):
                为 T5 模型启用 FSDP 分片
            dit_fsdp (`bool`, *optional*, defaults to False):
                为 DiT 模型启用 FSDP 分片
            use_usp (`bool`, *optional*, defaults to False):
                启用 USP 分布式策略
            t5_cpu (`bool`, *optional*, defaults to False):
                是否将 T5 模型放在 CPU 上，仅在不使用 t5_fsdp 时有效
            init_on_cpu (`bool`, *optional*, defaults to True):
                在 CPU 上初始化 Transformer 模型，仅在不使用 FSDP 或 USP 时有效
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dit_fsdp:
            self.model = shard_fn(self.model)
        elif not init_on_cpu:
            self.model = self.model.to(self.device)

    def generate(self,
                 input_prompt,
                 video_path,
                 mask_path=None,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 denoise_strength=0.7,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        根据输入提示和源视频生成新的视频。

        Args:
            input_prompt (`str`):
                生成视频的文本提示
            video_path (`str`):
                源视频的路径
            mask_path (`str`, *optional*, defaults to None):
                视频蒙版的路径，用于局部重绘。如果为 None，将重绘整个视频
            max_area (`int`, *optional*, defaults to 720*1280):
                生成视频的最大面积
            frame_num (`int`, *optional*, defaults to 81):
                生成视频的帧数，应为 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                用于生成的 flow shift 参数
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                用于采样的求解器
            sampling_steps (`int`, *optional*, defaults to 40):
                采样步骤数
            guide_scale (`float`, *optional*, defaults to 5.0):
                分类器引导比例
            denoise_strength (`float`, *optional*, defaults to 0.7):
                视频重绘的去噪强度，范围为 [0, 1]
            n_prompt (`str`, *optional*, defaults to ""):
                负面提示，用于指导生成过程中应避免的内容
            seed (`int`, *optional*, defaults to -1):
                随机种子。如果为 -1，则随机选择种子
            offload_model (`bool`, *optional*, defaults to True):
                是否在每次迭代后将模型卸载到 CPU

        Returns:
            视频帧列表
        """
        if self.rank == 0:
            logging.info(f"开始视频到视频转换...")
            logging.info(f"提示: {input_prompt}")
            logging.info(f"源视频: {video_path}")
            logging.info(f"蒙版: {mask_path}")
            logging.info(f"去噪强度: {denoise_strength}")

        if seed < 0:
            seed = random.randint(0, sys.maxsize)
        torch.manual_seed(seed)
        
        # 处理输入视频和掩码
        video_frames, width, height, fps = get_video_frames(video_path, frame_count=frame_num)
        
        # 确保宽度和高度满足要求
        aspect_ratio = width / height
        if width * height > max_area:
            # 缩放视频以符合最大面积限制
            height = int(math.sqrt(max_area / aspect_ratio))
            width = int(height * aspect_ratio)
            
            # 确保宽度和高度是 8 的倍数（VAE 要求）
            height = height - height % 8
            width = width - width % 8
        
        # 缩放视频帧
        scaled_frames = []
        for frame in video_frames:
            scaled_frame = cv2.resize(frame, (width, height))
            scaled_frames.append(scaled_frame)
        
        # 处理视频掩码（如果有）
        mask_frames = None
        if mask_path is not None:
            mask_frames, _, _, _ = get_video_frames(mask_path, frame_count=frame_num)
            
            # 缩放掩码
            scaled_mask_frames = []
            for mask in mask_frames:
                # 确保掩码是二值图像
                if len(mask.shape) == 3 and mask.shape[2] == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                scaled_mask = cv2.resize(binary_mask, (width, height))
                scaled_mask_frames.append(scaled_mask)
            
            mask_frames = scaled_mask_frames
        
        # 将视频帧转换为PyTorch张量
        video_tensor = torch.from_numpy(np.array(scaled_frames)).float() / 255.0
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
        
        if mask_frames is not None:
            mask_tensor = torch.from_numpy(np.array(mask_frames)).float() / 255.0
            mask_tensor = mask_tensor.unsqueeze(1) if len(mask_tensor.shape) == 3 else mask_tensor  # 确保掩码是 [T, 1, H, W]
        else:
            # 如果没有掩码，创建全1掩码（重绘整个视频）
            mask_tensor = torch.ones((video_tensor.shape[0], 1, height, width))
        
        # 添加批次维度
        video_tensor = video_tensor.unsqueeze(0).to(self.device)  # [1, T, C, H, W]
        mask_tensor = mask_tensor.unsqueeze(0).to(self.device)    # [1, T, 1, H, W]
        
        # 使用VAE编码视频帧
        with torch.no_grad():
            # VAE.encode 方法期望输入是一个视频列表，每个视频形状为 [C, T, H, W]
            # 因此我们需要将视频张量转换为列表
            video_list = [video_tensor.squeeze(0)]  # 移除批次维度
            video_latents_list = self.vae.encode(video_list)  # 这会返回一个列表
            # 获取列表中的第一个（也是唯一的）潜在表示
            video_latents = video_latents_list[0]
            # 添加批次维度
            video_latents = video_latents.unsqueeze(0)  # [1, C, T, H, W]

        # 处理文本提示
        device = torch.device("cpu") if self.t5_cpu else self.device
        text_embeds = self.text_encoder(
            [input_prompt], device=device)
        
        if self.t5_cpu:
            text_embeds = [x.to(self.device) for x in text_embeds]
        
        # T5EncoderModel.__call__ 返回一个列表，我们取第一个元素（只有一个提示）
        text_embeds = text_embeds[0]

        # 处理负面提示
        neg_text_embeds = None
        if guide_scale > 1.0 and n_prompt:
            neg_text_embeds = self.text_encoder(
                [n_prompt], device=device)
            if self.t5_cpu:
                neg_text_embeds = [x.to(self.device) for x in neg_text_embeds]
            neg_text_embeds = neg_text_embeds[0]

        # 确保模型在正确的设备上
        if not isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model = self.model.to(self.device)
        
        # 尝试修改模型类型以解决断言错误（此代码可能在某些情况下不起作用）
        try:
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                if hasattr(self.model.module, 'model_type') and self.model.module.model_type == 'i2v':
                    logging.info("尝试修改 I2V 模型（DDP）的 model_type 为 't2v'")
                    self.model.module.model_type = 't2v'
            else:
                if hasattr(self.model, 'model_type') and self.model.model_type == 'i2v':
                    logging.info("尝试修改 I2V 模型的 model_type 为 't2v'")
                    self.model.model_type = 't2v'
        except Exception as e:
            logging.warning(f"修改模型类型失败: {e}，将提供必要的参数以解决断言错误")
        
        # 设置采样器
        if sample_solver == 'unipc':
            # 使用UniPC采样器
            sampler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                prediction_type='flow_prediction',
                shift=shift,
            )
        else:
            # 使用DPM采样器
            sampler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps,
                prediction_type='flow_prediction',
                shift=shift,
                solver_order=2,
                algorithm_type="dpmsolver++",
                solver_type="midpoint",
            )

        # 计算采样时间步长
        timesteps, _ = retrieve_timesteps(
            sampler, sampling_steps, self.device, 
            timesteps=None)
        latent_timestep = timesteps[:1]
        
        # 根据去噪强度计算起始时间步
        start_timestep_idx = int(denoise_strength * len(timesteps))
        timesteps = timesteps[start_timestep_idx:]
        
        # 初始化latent
        noise = torch.randn_like(video_latents)
        init_latents = video_latents
        latents = sampler.add_noise(init_latents, noise, timesteps[:1])

        # 采样循环
        with tqdm(total=len(timesteps), disable=self.rank != 0) as progress_bar:
            model = self.model
            for t in timesteps:
                model_t = torch.tensor([t]).to(self.device)
                
                # 准备分类器引导
                latent_model_input = torch.cat([latents] * 2) if neg_text_embeds is not None else latents
                
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    # 从视频的第一帧提取 CLIP 特征
                    first_frame = video_tensor[:, 0]  # [B, C, H, W]
                    clip_features = self.clip.visual([first_frame])
                    
                    model_output = torch.utils.checkpoint.checkpoint(
                        model.module.forward,
                        latent_model_input,  # x
                        model_t,  # t
                        text_embeds.repeat(2, 1, 1) if neg_text_embeds is not None else text_embeds,  # context
                        self.config.text_len,  # seq_len
                        clip_features,  # clip_fea
                        init_latents,  # y (使用初始潜在表示作为条件输入)
                    )
                else:
                    # 从视频的第一帧提取 CLIP 特征
                    first_frame = video_tensor[:, 0]  # [B, C, H, W]
                    clip_features = self.clip.visual([first_frame])
                    
                    model_output = torch.utils.checkpoint.checkpoint(
                        model.forward,
                        latent_model_input,  # x
                        model_t,  # t
                        text_embeds.repeat(2, 1, 1) if neg_text_embeds is not None else text_embeds,  # context
                        self.config.text_len,  # seq_len
                        clip_features,  # clip_fea
                        init_latents,  # y (使用初始潜在表示作为条件输入)
                    )
                
                # 分类器引导处理
                if neg_text_embeds is not None:
                    noise_pred_uncond, noise_pred_text = model_output.chunk(2)
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = model_output
                
                # 更新latents
                latents = sampler.step(noise_pred, t, latents).prev_sample
                
                # 应用掩码
                mask_weight = mask_tensor.to(latents.dtype)
                init_latents_proper = sampler.add_noise(init_latents, noise, torch.tensor([t]))
                latents = (1 - mask_weight) * init_latents_proper + mask_weight * latents
                
                progress_bar.update(1)

        # 使用VAE解码生成的latents
        video_output = self.vae.decode(latents)
        
        # 转换为numpy数组
        video_output = (video_output.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)[0]
        
        # 清理GPU内存
        if offload_model and not isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()
            
        return video_output, fps

    @contextmanager
    def noop_no_sync(self):
        yield 