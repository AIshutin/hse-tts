import logging
from pathlib import Path

import torchaudio

from tts.base.base_dataset import BaseDataset
import torch
from torch.utils.data import Dataset
from copy import deepcopy

logger = logging.getLogger(__name__)


class CustomTextDataset(Dataset):
    def __init__(self, texts, configurations, text_encoder,
                config_parser, **kwargs):
        super().__init__()
        self.samples = []
        
        for c in configurations:
            for text in texts:
                c['text'] = text
                self.samples.append(deepcopy(c))
        
        self.text_encoder = text_encoder
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        text = self.samples[i]['text']
        encoded_text = self.text_encoder.encode(text)
        name = f"{text.split()[0]}-dur={self.samples[i]['duration']}-pitch={self.samples[i]['pitch']}-energy={self.samples[i]['energy']}"
        out_dict = {
            "audio": torch.zeros(1, 13),
            "spectrogram": torch.zeros(1, 13, 80),
            "duration": 1,
            "text": text,
            "text_encoded": encoded_text,
            "audio_path": name,
            "alignment": torch.zeros(13),
            "idx": i,
            "energy": torch.zeros(1, 13),
            "pitch": torch.zeros(1, 13),
            "alpha_duration": self.samples[i]['duration'],
            "alpha_pitch": self.samples[i]['pitch'],
            "alpha_energy": self.samples[i]['energy']
        }
        return out_dict