import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union

import numpy as np
from torch import Tensor

from hw_asr.base.base_text_encoder import BaseTextEncoder
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer


class BPETextEncoder(BaseTextEncoder):
    def __init__(self, file: str, pad_token=None):
        self.tokenizer = Tokenizer.from_file(file)
        if pad_token is not None:
            self.tokenizer.pad_token = pad_token

    def __len__(self):
        return len(self.tokenizer.get_vocab())

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.tokenizer.decode([item])

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        out = Tensor([self.tokenizer.encode(text).ids])
        return out

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        return self.tokenizer.decode(vector)

    def dump(self, file):
        self.tokenizer.save_pretrained(file)

    @classmethod
    def from_file(cls, file):
        return cls(file)
