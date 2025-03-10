# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan V2V 14B ------------------------#

v2v_14B = EasyDict(__name__='Config: Wan V2V 14B')
v2v_14B.update(wan_shared_cfg)

v2v_14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
v2v_14B.t5_tokenizer = 'google/umt5-xxl'

# clip
v2v_14B.clip_model = 'clip_xlm_roberta_vit_h_14'
v2v_14B.clip_dtype = torch.float16
v2v_14B.clip_checkpoint = 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
v2v_14B.clip_tokenizer = 'xlm-roberta-large'

# vae
v2v_14B.vae_checkpoint = 'Wan2.1_VAE.pth'
v2v_14B.vae_stride = (4, 8, 8)

# transformer
v2v_14B.patch_size = (1, 2, 2)
v2v_14B.dim = 5120
v2v_14B.ffn_dim = 13824
v2v_14B.freq_dim = 256
v2v_14B.num_heads = 40
v2v_14B.num_layers = 40
v2v_14B.window_size = (-1, -1)
v2v_14B.qk_norm = True
v2v_14B.cross_attn_norm = True
v2v_14B.eps = 1e-6 