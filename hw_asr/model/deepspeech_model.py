from torch import nn
from torch.nn import Sequential
import torch

from hw_asr.base import BaseModel

class ClampModule(nn.Module):
    def __init__(self, max=None, min=None) -> None:
        super().__init__()
        self.max = max
        self.min = min
    
    def forward(self, X):
        return torch.clamp(X, max=self.max, min=self.min)


class DeepSpeechModel2(BaseModel):
    def __init__(self, n_channels, n_class, conv_channel=32, rnn_layers=3, fc_hidden=512, rnn_hidden=512, clip_max=20, dropout_p=0.05, **batch):
        super().__init__(n_channels, n_class, **batch)

        self.cnn = Sequential(
            nn.Conv2d(in_channels=1, out_channels=conv_channel, kernel_size=(11, 42), stride=(2, 2)),
            nn.BatchNorm2d(conv_channel),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_p)
        )

        self.rnn = nn.GRU(input_size=1408, num_layers=rnn_layers, hidden_size=rnn_hidden, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(in_features=2 * rnn_hidden, out_features=fc_hidden), # again not quite the same as in deepspeech
            ClampModule(max=clip_max),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(in_features=fc_hidden, out_features=n_class),
        )

    def forward(self, spectrogram, **batch):
        spectrogram = spectrogram.transpose(1, 2)
        assert(len(spectrogram.shape) == 3) # B, T, F
        B, T, F = spectrogram.shape
        before_shape = spectrogram.shape[1:]
        # print(spectrogram.shape, '!!')
        X = self.cnn(spectrogram.unsqueeze(1))
        after_shape = X.shape[1:]
        X = X.transpose(1, 2) # B, T, C, F
        # print(X.shape, '$$')
        B, T, C, F = X.shape
        X = X.reshape((B, T, C * F))
        # print(batch['spectrogram_length'].min().item(), batch['spectrogram_length'].max().item())
        # print('length', T, batch['text_encoded_length'].min(), self.transform_input_lengths(batch['spectrogram_length']).min())
        # print(before_shape, '->', after_shape, '->', X.shape[1:])
        out, hidden = self.rnn(X)
        out = self.head(out)
        return {"logits": out}

    def transform_input_lengths(self, input_lengths):
        return torch.round((input_lengths - 1 * (11 - 1)) / 2).long()


class DeepSpeechModel(BaseModel):
    def __init__(self, n_channels, C, n_class, fc_hidden=512, rnn_hidden=512, clip_max=20, dropout_p=0.05, **batch):
        super().__init__(n_channels, n_class, **batch)
        self.C = C
        self.lower_net = Sequential(
            nn.Linear(in_features=n_channels * (2 * C + 1), out_features=fc_hidden),
            ClampModule(max=clip_max),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            ClampModule(max=clip_max),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            ClampModule(max=clip_max),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        
        # it's not quite the same as in deepspeech, but hopefully it doesn't matter
        self.rnn = nn.RNN(input_size=fc_hidden, hidden_size=rnn_hidden, bidirectional=True) 

        self.head = nn.Sequential(
            nn.Linear(in_features=2 * rnn_hidden, out_features=fc_hidden), # again not quite the same as in deepspeech
            ClampModule(max=clip_max),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Linear(in_features=fc_hidden, out_features=n_class),
        )

    def forward(self, spectrogram, **batch):
        spectrogram = spectrogram.transpose(1, 2)
        assert(len(spectrogram.shape) == 3) # B, T, W
        B, T, W = spectrogram.shape
        assert(spectrogram.shape[-1] >= 1 + 2 * self.C)
        X = torch.zeros((B, T, W, (1 + 2 * self.C)), device=spectrogram.device)
        for i, j in enumerate(range(-self.C, self.C + 1)):
            tmp = spectrogram.roll(j, dims=-1)
            X[:, :, :, i] = tmp

        X = X.reshape((B, T, -1))
        X = X[:, self.C:-self.C:2, :]
        X = self.lower_net(X)
        out, hidden = self.rnn(X)
        out = self.head(out)
        return {"logits": out}

    def transform_input_lengths(self, input_lengths):
        return (input_lengths - 2 * self.C + 1) // 2