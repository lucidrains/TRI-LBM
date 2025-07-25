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

from denoising_diffusion_pytorch import GaussianDiffusion



# functions

def exists(v):
    return v is not None

# classes

class LBM(Module):
    def __init__(
        self,
        clip_language_model = 'ViT-B-32',
        language_pretrained_name = 'laion2b_s34b_b79k',
        clip_image_model = 'ViT-B-16',
        image_pretrained_name = 'openai',
    ):
        super().__init__()
        # Clip, they use

        # ViT-B-16 for images
        # ViT-B-32 for langauge

        # reading in between the lines, they struggled with language steering
        # we will try to improve on that with the finding from Bytedance's GR-3 with the prediction of positive / negative task status (contrastive learning between command / action)

        language_model, _, preprocess = open_clip.create_model_and_transforms(clip_language_model, pretrained = language_pretrained_name)
        language_model.eval()
        tokenizer = open_clip.get_tokenizer(clip_language_model)

        image_model, _, image_preprocess = open_clip.create_model_and_transforms(clip_image_model, pretrained = image_pretrained_name)

        self.language_model = language_model
        self.image_model = image_model

    def forward(
        self,
        text: list[str],
        images: Tensor
    ):
        raise NotImplementedError
