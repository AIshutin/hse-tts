import re
from typing import List, Union

import numpy as np
from torch import Tensor


class BaseTextEncoder:
    def encode(self, text) -> Tensor:
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    @staticmethod
    def normalize_text(text: str):
        return text
