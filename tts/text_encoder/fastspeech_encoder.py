import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union

import numpy as np
from torch import Tensor

from ..base.base_text_encoder import BaseTextEncoder
from ..text import text_to_sequence, _id_to_symbol


class FastSpeechEncoder(BaseTextEncoder):
    PAD_ID = 0

    def __init__(self, cleaners=["english_cleaners"], pad_token="<pad>"):
        self.PAD_TOKEN = pad_token
        self.cleaners = cleaners

    def __len__(self):
        return len(_id_to_symbol) + 1

    def encode(self, text) -> Tensor:
        seq = text_to_sequence(text, self.cleaners)
        out = Tensor([[el + 1 for el in seq]])
        return out