import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBatchBased


class PitchShift(AugmentationBatchBased):
    def __init__(self, *args, **kwargs):
        super().__init__(torch_audiomentations.PitchShift(*args, **kwargs))
