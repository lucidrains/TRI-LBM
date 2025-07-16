import torch
from torch import nn
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
    def __init__(self):
        super().__init__()
        raise NotImplementedError
