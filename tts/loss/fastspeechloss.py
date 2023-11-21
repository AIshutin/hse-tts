import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn

class FastSpeechLoss(nn.Module):
    def __init__(self, w_align=1) -> None:
        super().__init__()
        self.w_align = w_align
        self.l1_loss = nn.L1Loss()

    def forward(self, spectrogram_hat, spectrogram, alignment_hat, 
                alignment, **batch) -> Tensor:
        f = F.mse_loss
        loss_spectr = f(spectrogram_hat, spectrogram)
        loss_align  = self.l1_loss(alignment_hat.float(), alignment.float())
        out = {
            "loss": loss_spectr  + self.w_align * loss_align,
            "loss_spectrogram": loss_spectr,
            "loss_pitch": torch.tensor(0),
            "loss_align": loss_align,
            "loss_energy": torch.tensor(0)
        }
        return out
    

class FastSpeech2Loss(nn.Module):
    def __init__(self, w_pitch=1, w_align=1, w_energy=1) -> None:
        super().__init__()
        self.w_pitch = w_pitch
        self.w_align = w_align
        self.w_energy = w_energy

    def forward(self, spectrogram_hat, spectrogram, pitch_hat, pitch, alignment_hat, 
                alignment, energy_hat, energy, **batch) -> Tensor:
        f = F.mse_loss
        loss_spectr = f(spectrogram_hat, spectrogram)
        if pitch_hat.shape != pitch.shape:
            pitch = torch.cat((pitch.real, pitch.imag), dim=-1)
        loss_pitch  = f(pitch_hat, pitch)

        if batch.get('pitch_params_hat') is not None:
            loss_pitch += f(batch['pitch_params_hat'], batch['pitch_params'])
        loss_align  = f(alignment_hat, alignment.log1p())
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