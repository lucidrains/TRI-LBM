from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor, is_tensor, cat, stack
from torch.nn import Module, ModuleList

# ein notation
# b - batch
# t - time
# c - channels
# h - height
# w - width

from einops import rearrange
from einops.layers.torch import Rearrange

# dogfooding

from x_transformers import (
    Encoder,
    TransformerWrapper
)

from denoising_diffusion_pytorch import (
    GaussianDiffusion1D
)

# open clip

import open_clip

from vit_pytorch.accept_video_wrapper import AcceptVideoWrapper

from bidirectional_cross_attention import BidirectionalCrossAttentionTransformer as BiCrossAttnTransformer

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

def divisible_by(num, den):
    return (num % den) == 0

# random sinusoidal for times - used by deepmind a lot

class RandomSinusoidalPosEmb(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = False)

    def forward(self, x):
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * torch.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        return fouriered

# DiT wrapper

class DiffusionTransformerWrapper(Module):
    def __init__(
        self,
        dim_input,
        dim_time,
        transformer: Encoder
    ):
        super().__init__()

        self.transformer = transformer

        dim = transformer.dim

        self.proj_in = nn.Linear(dim_input, dim)

        self.to_time_cond = nn.Sequential(
            RandomSinusoidalPosEmb(dim),
            nn.Linear(dim, dim_time),
            nn.SiLU(),
        )

        self.proj_out = nn.Linear(dim, dim_input)

    def forward(
        self,
        actions,
        times,
        text,
        images,
        pose
    ):
        batch_size = actions.shape[0]

        time_cond = self.to_time_cond(times)

        tokens = self.proj_in(actions)

        images = rearrange(images, 'b t d -> b (t d)')
        condition = cat((time_cond, text, images, pose), dim = -1)

        attended = self.transformer(tokens, condition = condition)

        pred = self.proj_out(attended)
        return pred

# classes

class LBM(Module):
    def __init__(
        self,
        action_dim,
        dim_pose,
        dim = 768,
        depth = 8, # Table 2. - not very deep at all
        dim_head = 64,
        heads = 12,
        action_chunk_length = 16,
        action_mean_std_for_norm: Tensor | None = None, # Float['d 2'] - last dimension must be shift and inv scale
        diffusion_timesteps = 1000,
        diffusion_sampling_timesteps = 16,
        transformer_kwargs: dict = dict(),
        diffusion_kwargs: dict = dict(),
        clip_language_model = 'ViT-B-32',
        language_pretrained_name = 'laion2b_s34b_b79k',
        clip_image_model = 'ViT-B-16',
        image_pretrained_name = 'openai',
        norm_clip_embeds = True,
        num_image_frames = 3,
        dim_tactile_input = None,
        tactile_image_fusion_depth = 2
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
        self.language_tokenizer = tokenizer

        self.image_preprocess = preprocess
        self.image_model = image_model
        self.accept_video_wrapper = AcceptVideoWrapper(image_model, forward_function = 'encode_image')

        self.norm_clip_embeds = norm_clip_embeds

        # cheap way to get feat dimensions
        # assume one image for starters

        dim_text_feats = language_model.encode_text(tokenizer(['test'])).shape[-1]
        dim_image_feats = image_model.encode_image(torch.randn(1, 3, 224, 224)).shape[-1]
        dim_time = dim * 2

        dim_observation = (
            dim_time +
            dim_text_feats +
            dim_image_feats * num_image_frames +
            dim_pose
        )

        self.images_shape = (3, num_image_frames, 224, 224) # just enforce this shape to begin with

        self.diffusion_transformer = DiffusionTransformerWrapper(
            dim_input = action_dim,
            dim_time = dim_time,
            transformer = Encoder(
                dim = dim,
                depth = depth,
                heads = heads,
                attn_dim_head = dim_head,
                dim_condition = dim_observation,
                use_adaptive_layernorm = True,
                use_adaptive_layerscale = True
            )
        )

        self.gaussian_diffusion_1d = GaussianDiffusion1D(
            self.diffusion_transformer,
            seq_length = action_chunk_length,
            timesteps = diffusion_timesteps,
            sampling_timesteps = diffusion_sampling_timesteps,
            channels = action_dim,
            self_condition = False,
            channel_first = False
        )

        # tactile

        if exists(dim_tactile_input):
            self.to_tactile_tokens = nn.Linear(dim_tactile_input, dim)

            self.tactile_fusion = BiCrossAttnTransformer(
                dim = dim_image_feats,
                context_dim = dim,
                heads = heads,
                dim_head = dim_head,
                depth = tactile_image_fusion_depth
            )

        # one contribution of the paper is that Russ claims huge improvements (40x) by simply normalizing actions correctly

        self.normalize_actions = exists(action_mean_std_for_norm)

        if self.normalize_actions:
            assert action_mean_std_for_norm.shape == (action_dim, 2)
            self.register_buffer('action_mean_std_for_norm', action_mean_std_for_norm)

    def get_clip_text_image_feats(
        self,
        text: list[str] | Tensor,
        images: Tensor,               # (b c t h w)
        touch: Tensor | None = None,  # (b nt, dt)
    ):
        if not is_tensor(text):
            text = self.language_tokenizer(text)

        with torch.no_grad():
            self.language_model.eval()
            text = self.language_model.encode_text(text)

        images = self.accept_video_wrapper(images, eval_with_no_grad = True)

        if exists(touch):
            assert exists(self.to_tactile_tokens), f'`dim_tactile_input` must be set if tactile data is passed in'

            tactile_tokens = self.to_tactile_tokens(touch)
            images, tactile_tokens = self.tactile_fusion(images, tactile_tokens)

        if self.norm_clip_embeds:
            text, images = map(l2norm, (text, images))

        return text, images

    def sample(
        self,
        text: list[str] | Tensor,
        images: Tensor,
        pose: Tensor,
        touch: Tensor | None = None,
        return_noise = False
    ):
        batch_size = images.shape[0]

        text, images = self.get_clip_text_image_feats(text, images, touch = touch)
        
        sampled_actions, noise =  self.gaussian_diffusion_1d.sample(batch_size = batch_size, return_noise = True, model_forward_kwargs = dict(text = text, images = images, pose = pose))

        if self.normalize_actions:
            mean, std = self.action_mean_std_for_norm.unbind(dim = -1)
            sampled = sampled * std + mean

        if not return_noise:
            return sampled_actions

        return sampled_actions, noise

    def forward(
        self,
        text: list[str] | Tensor,
        images: Tensor,
        pose: Tensor,
        touch: Tensor | None = None,
        actions: Tensor | None = None,
    ):
        assert images.shape[1:] == self.images_shape

        if not exists(actions):
            return self.sample(text = text, images = images)

        # take care of normalizing actions, if statistics were set on init

        if self.normalize_actions:
            mean, std = self.action_mean_std_for_norm.unbind(dim = -1)
            actions = (actions - mean) / std

        text, images = self.get_clip_text_image_feats(text, images, touch = touch)

        loss = self.gaussian_diffusion_1d(actions, model_forward_kwargs = dict(text = text, images = images, pose = pose))
        return loss
