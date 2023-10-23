from torch import Tensor
import torch

from hw_asr.augmentations.base import AugmentationBase


class AddGaussianNoise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.mean = kwargs.get('mean', 0)
        self.std_min  = kwargs['std_min']
        self.std_max  = kwargs['std_max']
        self.p = kwargs.get('p', 1.0)
    
    def __call__(self, x: Tensor):
        t = torch.zeros(size=(1,)).uniform_()
        if (t <= self.p):
            std = torch.zeros(size=(1,)).uniform_(self.std_min, self.std_max)
            return x + ((torch.randn(size=x.shape) * std) + self.mean)
        return x
