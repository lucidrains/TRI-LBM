from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor, tensor, is_tensor, cat, stack
from torch.utils._pytree import tree_map
from torch.nn import Module, ModuleList

# ein notation
# b - batch
# t - time
# c - channels
# h - height
# w - width
# d - dimension
# na - num actions

import einx
from einops import rearrange, repeat, pack, unpack, reduce
from einops.layers.torch import Rearrange, Reduce

# dogfooding

from x_transformers import (
    Encoder,
    TransformerWrapper,
    ContinuousTransformerWrapper
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

def identity(t):
    return t

def compact(arr):
    return [*filter(exists, arr)]

def maybe_cat(arr, *, dim):
    if len(arr) == 0:
        return None

    return cat(arr, dim = dim)

def xnor(x, y):
    return not (x ^ y)

def l2norm(t):
    return F.normalize(t, dim = -1)

def detach_all(obj):
    return tree_map(lambda t: t.detach() if is_tensor(t) else t, obj)

def divisible_by(num, den):
    return (num % den) == 0

# dataset normalization related
# Russ Tedrake's proposal

class ActionClassifier(Module):
    def __init__(
        self,
        *,
        dim_action,
        num_action_types,
        dim = 256,
        depth = 3,
        attn_kwargs: dict = dict(
            attn_dim_head = 64,
            heads = 8
        ),
        action_counts: Tensor | None = None,
        action_mean: Tensor | None = None,
        action_variance: Tensor | None = None,
        action_sum_diff_squared: Tensor | None = None,
        eps = 1e-5
    ):
        super().__init__()

        self.num_action_types = num_action_types

        self.to_tokens = nn.Sequential(
            nn.Linear(dim_action, dim)
        )

        self.transformer = Encoder(
            dim = dim,
            depth = depth,
            **attn_kwargs
        )

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, num_action_types)
        )

        # store norm related

        assert not (exists(action_variance) and exists(action_sum_diff_squared))

        if exists(action_variance):
            assert exists(action_counts) and (action_counts > 1).all()
            action_sum_diff_squared = einx.multiply('b d, b', action_variance, action_counts - 1.)

        self.register_buffer('action_counts', default(action_counts, torch.zeros(num_action_types)))
        self.register_buffer('action_mean', default(action_mean, torch.zeros(num_action_types, dim_action)))
        self.register_buffer('action_sum_diff_squared', default(action_sum_diff_squared, torch.zeros(num_action_types, dim_action)))

    @property
    def device(self):
        return self.action_counts.device

    @property
    def action_variance(self):
        return einx.divide('b d, b', self.action_sum_diff_squared, self.action_counts - 1)

    def update_action_statistics_with_welford_(
        self,
        actions,      # (b d) | (b t d)
        action_types  # (b) | (b t)
    ):
        if actions.ndim == 3:
            times = actions.shape[1]
            actions = rearrange(actions, 'b t d -> (b t) d')
            action_types = repeat(action_types, 'b -> (b t)', t = times)

        if action_types.ndim == 2:
            action_types = rearrange(action_types, 'b t -> (b t)')

        for one_action, action_type in zip(actions, action_types):

            count = self.action_counts[action_type]
            mean = self.action_mean[action_type]
            sum_diff_squared = self.action_sum_diff_squared[action_type]

            count += 1
            delta = one_action - mean
            new_mean = mean + delta / count
            sum_diff_squared += delta * (one_action - new_mean)

            self.action_counts[action_type] = count
            self.action_mean[action_type] = new_mean
            self.action_sum_diff_squared[action_type] = sum_diff_squared

    def update_action_statistics_with_parallel_welford_(
        self,
        actions,      # (b d) | (b t d)
        action_types  # (b) | (b t)
    ):
        if actions.ndim == 3:
            times = actions.shape[1]
            actions = rearrange(actions, 'b t d -> (b t) d')

            # the entire time chunk is of one action type

            if action_types.ndim == 1:
                action_types = repeat(action_types, 'b -> (b t)', t = times)

        if action_types.ndim == 2:
            action_types = rearrange(action_types, 'b t -> (b t)')

        assert actions.shape[0] == action_types.shape[0]

        batch, device = actions.shape[0], actions.device

        num_actions_seq = torch.arange(self.num_action_types, device = device)

        mask = einx.equal('na, b -> na b', num_actions_seq, action_types)

        old_sum_diff_squared, old_mean, old_count = self.action_sum_diff_squared, self.action_mean, self.action_counts

        count = reduce(mask, 'na b -> na', 'sum')

        next_count = old_count + count

        repeated_actions = repeat(actions, '... -> na ...', na = self.num_action_types)

        masked_repeated_actions = einx.where('na b, na b d,', mask, repeated_actions, 0.)

        # calculate new mean

        numerator = reduce(masked_repeated_actions, 'na b d -> na d', 'sum')

        new_mean = einx.divide('na d, na -> na d', numerator, count.clamp(min = 1e-5))

        # delta

        delta = new_mean - old_mean

        # next mean

        ratio = count / next_count.clamp(min = 1e-5)

        next_mean = old_mean + einx.multiply('b ..., b', delta, ratio)

        # new sum square

        diff_squared = einx.subtract('na b d, na d', masked_repeated_actions,  new_mean).pow(2)

        masked_diff_squared = einx.where('na b, na b d,', mask, diff_squared, 0.)

        new_sum_diff_squared = reduce(masked_diff_squared, 'na b d -> na d', 'sum')

        ratio = (count * old_count) / (next_count)

        next_sum_diff_squared = old_sum_diff_squared + new_sum_diff_squared + einx.multiply('b, b ...', ratio, delta.pow(2))

        # update

        update_mask = count > 0

        self.action_counts.copy_(next_count)
        self.action_sum_diff_squared.copy_(einx.where('b, b ...,', update_mask, next_sum_diff_squared, 0.))
        self.action_mean.copy_(einx.where('b, b ...,', update_mask, next_mean, 0.))

    def normalize(
        self,
        actions,
        action_types
    ):
        counts = self.action_counts[action_types]
        means = self.action_mean[action_types]
        variances = self.action_variance[action_types]

        inv_std = variances.clamp(min = 1e-5).rsqrt()

        inv_std = einx.where('b,, b d', counts == 0, 1., inv_std)

        mean_centered = einx.subtract('b t d, b d', actions, means)
        normed = einx.multiply('b t d, b d', mean_centered, inv_std)
        return normed

    def inverse_normalize(
        self,
        normed_actions,
        action_types
    ):
        counts = self.action_counts[action_types]
        means = self.action_mean[action_types]
        variances = self.action_variance[action_types]

        std = variances.clamp(min = 1e-5).sqrt()
        std = einx.where('b,, b d', counts == 0, 1., std)

        normed_actions = einx.multiply('b t d, b d', normed_actions, std)
        actions = einx.add('b t d, b d', normed_actions, means)
        return actions

    def get_action_statistic(
        self,
        actions_dataset: Dataset,
        batch_size = 16
    ):
        dl = DataLoader(actions_dataset, batch_size = batch_size)

        for actions, action_types in dl:

            actions = actions.to(self.device)
            action_types = action_types.to(self.device)

            self.update_action_statistics_with_parallel_welford_(actions, action_types)

    def forward(
        self,
        actions,                # (b t d)
        action_types = None,    # (b)
        actions_are_normalized = False,
        return_denormalized_actions = True
    ):

        assert (self.action_counts > 0).any(), 'you need to have run through the entire dataset for the action statistics before being able to train a classifier'

        is_training = exists(action_types)
        is_inferencing = not is_training

        # when training, normalize the actions

        if is_training and not actions_are_normalized:
            actions = self.normalize(actions, action_types)

        # to tokens, attention, then to logits

        tokens = self.to_tokens(actions)

        attended = self.transformer(tokens)

        logits = self.to_logits(attended)

        # if not training, return predicted action class

        if not is_inferencing:
            return F.cross_entropy(logits, action_types)

        pred_action_types = logits.argmax(dim = -1)

        unnormed_actions = self.inverse_normalize(actions, pred_action_types)

        if not return_denormalized_actions:
            return pred_action_types

        return pred_action_types, unnormed_actions

# random sinusoidal for times - used by deepmind a lot

class RandomSinusoidalPosEmb(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = False)

    def forward(self, x):
        freqs = einx.multiply('b, d -> b d', x, self.weights) * 2 * torch.pi
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
        pose,
        *,
        prepend_embeds = None,
        context = None,
        context_mask = None,
        vlm_key_values = None,
        vlm_seq_mask = None
    ):
        batch_size = actions.shape[0]

        time_cond = self.to_time_cond(times)

        tokens = self.proj_in(actions)

        images = rearrange(images, 'b t d -> b (t d)')
        condition = cat((time_cond, text, images, pose), dim = -1)

        if exists(prepend_embeds):
            tokens, prepend_packed_shape = pack((prepend_embeds, tokens), 'b * d')

        attended = self.transformer(
            tokens,
            condition = condition,
            context = context,
            context_mask = context_mask,
            detach_additional_kv = True,
            self_attn_additional_kv = vlm_key_values,
            additional_kv_mask = vlm_seq_mask
        )

        if exists(prepend_embeds):
            _, attended = unpack(attended, prepend_packed_shape, 'b * d')

        pred = self.proj_out(attended)
        return pred

# classes

class LBM(Module):
    def __init__(
        self,
        action_dim,
        dim_pose,
        action_chunk_size = None,
        action_chunk_normalizer: ActionClassifier | None = None,
        dim = 768,
        depth = 8, # Table 2. - not very deep at all
        dim_head = 64,
        heads = 12,
        max_time_seq_len = 16,
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
        tactile_image_fusion_depth = 2,
        dim_depth_embed = None,
        add_task_status_prediction = True,  # Bytedance reports doing a crude contrastive learning on action / language pairs during training significantly improves instruction following - https://arxiv.org/abs/2507.15493
        accept_additional_context = False,  # cross attend to additional context, will be used on CLIP text encoding to improve on language following
        additional_context_dim = None,
        cross_attend_text_encodings = False,
        dropout_text_encodings_prob = 0.5
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

        # cheap way to get feat dimensions
        # assume one image for starters

        dim_text_feats = language_model.encode_text(tokenizer(['test'])).shape[-1]
        dim_image_feats = image_model.encode_image(torch.randn(1, 3, 224, 224)).shape[-1]

        # store language and image model as video frame processor

        self.language_model = language_model
        self.language_tokenizer = tokenizer

        self.image_preprocess = preprocess.transforms[-1]

        self.image_model = image_model

        self.accept_video_wrapper = AcceptVideoWrapper(
            image_model,
            forward_function = 'encode_image',
            add_time_pos_emb = True,
            time_seq_len = max_time_seq_len,
            dim_emb = dim_image_feats
        )

        self.norm_clip_embeds = norm_clip_embeds

        # determine cross attention dim

        additional_context_dim = default(additional_context_dim, dim)

        # whether to have the diffusion transformer cross attend to the fine text tokens from clip, for better language following

        self.cross_attend_text_encodings = cross_attend_text_encodings

        self.dropout_text_encodings_prob = dropout_text_encodings_prob

        self.text_encodings_to_cross_attn_embed = nn.Linear(dim_text_feats, additional_context_dim) if cross_attend_text_encodings else None

        # whether to do task status prediction

        self.add_task_status_prediction = add_task_status_prediction
        maybe_task_status_dim = bool(self.add_task_status_prediction)

        dim_time = dim * 2

        dim_observation = (
            dim_time +
            dim_text_feats +
            dim_image_feats * num_image_frames +
            dim_pose
        )

        self.images_shape = (3, num_image_frames, 224, 224) # just enforce this shape to begin with

        self.diffusion_transformer = DiffusionTransformerWrapper(
            dim_input = action_dim + maybe_task_status_dim,
            dim_time = dim_time,
            transformer = Encoder(
                dim = dim,
                depth = depth,
                heads = heads,
                cross_attend = accept_additional_context or cross_attend_text_encodings,
                cross_attn_dim_context = additional_context_dim,
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
            channels = action_dim + maybe_task_status_dim,
            self_condition = False,
            channel_first = False
        )

        # optional action normalizer - needs to be pretrained if passed in

        assert not (exists(action_chunk_normalizer) and exists(action_mean_std_for_norm))

        self.action_chunk_size = action_chunk_size
        self.action_chunk_normalizer = action_chunk_normalizer
        self.normalize_with_action_classifier = exists(action_chunk_normalizer)

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

        # depth embeds (Adapt3R paper)

        self.accept_depth_embed = exists(dim_depth_embed)

        self.to_depth_tokens = None

        if self.accept_depth_embed:
            self.to_depth_tokens = nn.Linear(dim_depth_embed, dim)

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
        # whether to extract text encodings from clip to forward to DiT

        text_encodings = text_mask = None

        if self.cross_attend_text_encodings:

            def hook(_, __, final_outputs):
                nonlocal text_encodings
                text_encodings = final_outputs

            text_forward_hook = self.language_model.ln_final.register_forward_hook(hook)

        # process text

        if not is_tensor(text):
            text = self.language_tokenizer(text)

        # forward through clip vit for text encoding

        with torch.no_grad():
            self.language_model.eval()
            text_embeds = self.language_model.encode_text(text)

        # image preprocess

        images = self.image_preprocess(images)

        image_embeds = self.accept_video_wrapper(images, eval_with_no_grad = True)

        if exists(touch):
            assert exists(self.to_tactile_tokens), f'`dim_tactile_input` must be set if tactile data is passed in'

            tactile_tokens = self.to_tactile_tokens(touch)
            image_embeds, tactile_tokens = self.tactile_fusion(image_embeds, tactile_tokens)

        if self.norm_clip_embeds:
            text_embeds, image_embeds = map(l2norm, (text_embeds, image_embeds))

        if self.cross_attend_text_encodings:
            text_forward_hook.remove()
            assert exists(text_encodings)

            text_encodings = self.text_encodings_to_cross_attn_embed(text_encodings)

            # get text lens, remove padding, and generate text mask for encoding

            text_mask = text != 0
            text_lens = text_mask.sum(dim = -1)
            max_text_len = text_lens.amax().item()

            text_encodings = text_encodings[:, :max_text_len]
            text_mask = text_mask[:, :max_text_len]

        return text_embeds, image_embeds, text_encodings, text_mask

    def sample(
        self,
        text: list[str] | Tensor,
        images: Tensor,
        pose: Tensor,
        touch: Tensor | None = None,
        depth_embed: Tensor | None = None,
        context: Tensor | None = None,      # Float[b n d]
        context_mask: Tensor | None = None, # Bool[b n]
        vlm_key_values: list[tuple[Tensor, Tensor]] | None = None,
        return_noise = False,
        remove_task_status = True
    ):
        batch_size = images.shape[0]

        text, images, maybe_text_encodings, maybe_text_mask = self.get_clip_text_image_feats(text, images, touch = touch)

        context = maybe_cat(compact([maybe_text_encodings, context]), dim = 1)

        context_mask = maybe_cat(compact([maybe_text_mask, context_mask]), dim = 1)

        model_forward_kwargs = dict(
            text = text,
            images = images,
            pose = pose,
            context = context,
            context_mask = context_mask,
            vlm_key_values = vlm_key_values
        )

        # maybe add depth tokens

        assert xnor(exists(depth_embed), self.accept_depth_embed)

        if exists(depth_embed):
            depth_tokens = self.to_depth_tokens(depth_embed)

            model_forward_kwargs.update(prepend_embeds = depth_tokens)

        # sample actions

        sampled_actions, noise =  self.gaussian_diffusion_1d.sample(batch_size = batch_size, return_noise = True, model_forward_kwargs = model_forward_kwargs)

        if self.normalize_with_action_classifier:
            action_len = sampled_actions.shape[1]
            needs_chunking = exists(self.action_chunk_size) and action_len > self.action_chunk_size

            if needs_chunking:
                assert divisible_by(action_len, self.action_chunk_size)

                sampled_actions = rearrange(sampled_actions, 'b (c t) ... -> b c t ...')
                sampled_actions, packed_shape = pack([sampled_actions], '* t d')

            pred_action_types, sampled_actions = self.action_chunk_normalizer(sampled_actions)

            if needs_chunking:
                sampled_actions, = unpack([sampled_actions], packed_shape, '* t d')
                sampled_actions = rearrange(sampled_actions, 'b c t ... -> b (c t) ...')

        if self.add_task_status_prediction and remove_task_status:
            # remove task status during inference
            # todo - should consider also fixing it at 0 and infill

            sampled_actions = sampled_actions[..., :-1]
            noise = noise[..., :-1]

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
        depth_embed: Tensor | None = None,
        actions: Tensor | None = None,
        action_types: Tensor | None = None, # Int[b]
        context: Tensor | None = None,      # Float[b n d]
        context_mask: Tensor | None = None, # Bool[b n]
        task_status: Tensor | None = None,  # must be Int['b'] of {-1, 0, 1} - `-1` for invalid action / language pair
        vlm_key_values: list[tuple[Tensor, Tensor]] | None = None
    ):
        batch, device = images.shape[0], images.device
        assert images.shape[1:] == self.images_shape

        if not exists(actions):
            return self.sample(text = text, images = images)

        # take care of normalizing actions, if statistics were set on init

        if self.normalize_actions:
            mean, std = self.action_mean_std_for_norm.unbind(dim = -1)
            actions = (actions - mean) / std

        text, images, maybe_text_encodings, maybe_text_mask = self.get_clip_text_image_feats(text, images, touch = touch)

        # take care of dropping out text encoding if enabled

        if self.training and exists(maybe_text_mask):
            dropout_text_encoding = torch.rand(batch, device = device) < self.dropout_text_encodings_prob

            maybe_text_mask = einx.where('b, , b n', dropout_text_encoding, False, maybe_text_mask)

        context = maybe_cat(compact([maybe_text_encodings, context]), dim = 1)

        context_mask = maybe_cat(compact([maybe_text_mask, context_mask]), dim = 1)

        # maybe add task status

        if self.add_task_status_prediction:
            is_invalid_task = task_status == -1

            if not exists(task_status):
                task_status = torch.zeros((batch,), device = device)

            task_status = repeat(task_status.float(), 'b -> b n 1', n = actions.shape[1])

            actions = cat((actions, task_status), dim = -1)

        # gaussian diffusion 1d loss

        model_forward_kwargs = dict(
            text = text,
            images = images,
            pose = pose,
            context = context,
            context_mask = context_mask,
            vlm_key_values = vlm_key_values
        )

        # maybe add depth tokens

        assert xnor(exists(depth_embed), self.accept_depth_embed), f'`dim_depth_embed` must be set if `depth_embed` were passed in (batch, seq, <dim_depth_embed>)'

        if exists(depth_embed):
            depth_tokens = self.to_depth_tokens(depth_embed)

            model_forward_kwargs.update(prepend_embeds = depth_tokens)

        # normalize action chunks if needed

        assert xnor(self.normalize_with_action_classifier, exists(action_types))

        if self.normalize_with_action_classifier:

            action_len = actions.shape[1]
            needs_chunking = exists(self.action_chunk_size) and action_len > self.action_chunk_size

            # if action being trained on is multiple chunks

            if needs_chunking:
                assert divisible_by(action_len, self.action_chunk_size)

                if action_types.ndim == 1:
                    action_types = rearrange(action_types, 'b -> b 1')

                actions = rearrange(actions, 'b (c t) d -> b c t d', t = self.action_chunk_size)
                action_types = repeat(action_types, 'b c -> b (c r)', r = action_len // self.action_chunk_size)

                actions, action_types = tuple(rearrange(t, 'b c ... -> (b c) ...') for t in (actions, action_types))

            # normalize actions

            actions = self.action_chunk_normalizer.normalize(actions, action_types)

            if needs_chunking:
                actions, = unpack(actions, '* t d')
                actions = rearrange(actions, 'b ... d -> b (...) d')

        # diffusion loss

        loss = self.gaussian_diffusion_1d(
            actions,
            model_forward_kwargs = model_forward_kwargs,
            return_reduced_loss = False
        )

        # for any invalid status, they omit the diffusion loss for those action, please open an issue if this is a misunderstanding

        if self.add_task_status_prediction:
            loss, task_status_loss = loss[..., :-1], loss[..., -1:]

            loss = loss[~is_invalid_task]

            all_losses, _ = pack((loss, task_status_loss), '*')

            # reduce

            loss = all_losses.mean()

        return loss
