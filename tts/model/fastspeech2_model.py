import torch
from torch import nn
from torch.nn import Sequential
from torch.nn import functional as F
from tts.base import BaseModel
import numpy as np
from .fastspeech_model import FFTBlock, Transpose, create_alignment, ScaledDotProductAttention, \
                              MultiHeadAttention, PositionwiseFeedForward, get_non_pad_mask, \
                              get_attn_key_pad_mask, Encoder, Decoder, get_mask_from_lengths, \
                              AttributeDict


class XPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config):
        super().__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.x_predictor_filter_size
        self.kernel = model_config.x_predictor_kernel_size
        self.conv_output_size = model_config.x_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
        out = self.linear_layer(encoder_output)
        out = out.squeeze()
        return out


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = XPredictor(model_config)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0].item()
        expand_max_len = int(expand_max_len)
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            t = (duration_predictor_output.expm1() * alpha).round().to(torch.long)
            if len(t.shape) == 1:
                t = t.unsqueeze(0)
            output = self.LR(x, t)

            mel_pos = torch.stack([
                torch.Tensor([i + 1 for i in range(output.size(1))])
            ]).long().to(x.device)
            return output, mel_pos



class EnergyRegulator(nn.Module):
    """ Energy Regulator """

    def __init__(self, model_config):
        super().__init__()
        self.energy_predictor = XPredictor(model_config)
        self.min_energy = model_config.min_energy
        self.max_energy = model_config.max_energy
        self.n_quant = model_config.n_quant_energy
        self.energy_embs = nn.Embedding(self.n_quant + 1, model_config.energy_emb_dim) 
        self.div_quant = (self.max_energy - self.min_energy) / self.n_quant


    def forward(self, x, alpha=1.0, target=None, mel_lengths=None):
        energy_predictor_output = self.energy_predictor(x)

        with torch.no_grad():
            t = energy_predictor_output if target is None else target
            quant_id = torch.clamp(t * alpha[:, None], min=self.min_energy, max=self.max_energy)
            quant_id -= self.min_energy
            quant_id = (quant_id / self.div_quant).round().int() + 1
        
        if mel_lengths is not None:
            mask = get_mask_from_lengths(mel_lengths).to(quant_id.device)
            quant_id = quant_id * mask
            energy_predictor_output = energy_predictor_output * mask

        energy_embed = self.energy_embs(quant_id)
        return energy_embed, energy_predictor_output


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, model_config):
        super().__init__()

        self.encoder = Encoder(model_config)
        self.length_regulator = LengthRegulator(model_config)
        self.energy_regulator = EnergyRegulator(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, model_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, \
                energy_target=None, mel_length=None, alpha_duration=1.0, alpha_energy=1.0):
        x, non_pad_mask = self.encoder(src_seq, src_pos)

        if length_target is not None:
            output, duration_predictor_output = self.length_regulator(x, alpha_duration, 
                                                        length_target, mel_max_length)
            output_energy, energy_predictor_output = self.energy_regulator(output, alpha_energy, 
                                                        energy_target, mel_length)
            output = output + output_energy
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            output = self.mel_linear(output)
            return output, duration_predictor_output, energy_predictor_output
        else:
            output, mel_pos = self.length_regulator(x, alpha_duration)
            output_energy, energy_predictor_output = self.energy_regulator(output, alpha_energy)
            output = output + output_energy
            output = self.decoder(output, mel_pos)
            output = self.mel_linear(output)
            return output, mel_pos, energy_predictor_output


class FastSpeech2Model(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()

        self.fs = FastSpeech2(AttributeDict(**kwargs))

    def forward(self, alpha_duration, alpha_energy, **batch):
        spectrogram_hat, durations_hat, energy_hat = self.fs(
                      alpha_duration=alpha_duration,
                      alpha_energy=alpha_energy,
                      src_seq=batch['text_encoded'].to(torch.long),
                      src_pos=batch.get('src_pos'),
                      mel_pos=batch.get('mel_pos'),
                      mel_max_length=batch.get('max_mel_len'),
                      length_target=batch.get('alignment'),
                      mel_length=batch.get('spectrogram_length'),
                      energy_target=batch.get('energy'))
        out = {
            "spectrogram_hat": spectrogram_hat,
            "alignment_hat": durations_hat,
            "energy_hat": energy_hat
        }
        if "pitch" in batch:
            out['pitch_hat'] = batch['pitch']

        return out