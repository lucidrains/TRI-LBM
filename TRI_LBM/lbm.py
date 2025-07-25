from __future__ import annotations

import torch
from torch import nn, Tensor, tensor, is_tensor
from torch.nn import Module, ModuleList

# ein

from einops import rearrange

# dogfooding

from x_transformers import (
    Encoder,
    TransformerWrapper
)

from denoising_diffusion_pytorch import (
    GaussianDiffusion1D
)

# functions

def exists(v):
    return v is not None

# classes

class LBM(Module):
    def __init__(
        self,
        action_dim,
        dim = 768,
        depth = 8, # Table 2. - not very deep at all
        dim_head = 64,
        heads = 12,
        action_chunk_length = 16,
        action_mean_std_for_norm: Tensor | None = None, # Float['d 2'] - last dimension must be shift and inv scale
        diffusion_timesteps = 1000,
        transformer_kwargs: dict = dict(),
        diffusion_kwargs: dict = dict(),
        clip_language_model = 'ViT-B-32',
        language_pretrained_name = 'laion2b_s34b_b79k',
        clip_image_model = 'ViT-B-16',
        image_pretrained_name = 'openai',
    ):
        super().__init__()
        # Clip, they use

        # ViT-B-16 for images
        # ViT-B-32 for language

        # reading in between the lines, they struggled with language steering
        # we will try to improve on that with the finding from Bytedance's GR-3 with the prediction of positive / negative task status (contrastive learning between command / action)

        language_model, _, preprocess = open_clip.create_model_and_transforms(clip_language_model, pretrained = language_pretrained_name)
        language_model.eval()
        tokenizer = open_clip.get_tokenizer(clip_language_model)

        image_model, _, image_preprocess = open_clip.create_model_and_transforms(clip_image_model, pretrained = image_pretrained_name)

        self.language_model = language_model
        self.image_model = image_model

        self.diffusion_transformer = Encoder(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            cross_attend = True,
            use_adaptive_layernorm = True,
            use_adaptive_layerscale = True
        )

        self.gaussian_diffusion_1d = GaussianDiffusion1D(
            self.diffusion_transformer,
            seq_length = action_chunk_length,
            timesteps = diffusion_timesteps
        )

        # one contribution of the paper is that Russ claims huge improvements (40x) by simply normalizing actions correctly

        if exists(action_mean_std_for_norm):
            assert action_mean_std_for_norm.shape == (action_dim, 2)
            self.register_buffer('action_mean_std_for_norm', action_mean_std_for_norm)

    def sample(
        self,
        text: list[str] | Tensor,
        images: Tensor,
        sample_timesteps = 16 # ddim
    ):
        raise NotImplementedError

    def forward(
        self,
        text: list[str] | Tensor,
        images: Tensor,
        actions: Tensor | None = None,
    ):
        if not exists(actions):
            return self.sample(text = text, images = images)

        # take care of normalizing actions, if statistics were set on init

        if exists(self.action_mean_std_for_norm):
            mean, std = self.action_mean_std_for_norm.unbind(dim = -1)
            actions = (actions - mean) / std

        raise NotImplementedError
