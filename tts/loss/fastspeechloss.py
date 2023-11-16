import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn

class FastSpeechLoss(nn.Module):
    def __init__(self, w_pitch=1, w_align=1, w_energy=1) -> None:
        super().__init__()
        self.w_pitch = w_pitch
        self.w_align = w_align
        self.w_energy = w_energy

    def forward(self, spectrogram_hat, spectrogram, pitch_hat, pitch, alignment_hat, 
                alignment, energy_hat, energy, **batch) -> Tensor:
        f = F.mse_loss
        loss_spectr = f(spectrogram_hat, spectrogram)
        loss_pitch  = f(pitch_hat, pitch)
        loss_align  = f(alignment_hat, alignment)
        loss_energy = f(energy_hat, energy)
        out = {
            "loss": loss_spectr + self.w_pitch * loss_pitch + self.w_align * loss_align +\
                    self.w_energy * loss_energy,
            "loss_spectrogram": loss_spectr,
            "loss_pitch": loss_pitch,
            "loss_align": loss_align,
            "loss_energy": loss_energy
        }
        return out