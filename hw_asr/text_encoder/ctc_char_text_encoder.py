from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder
from pyctcdecode import build_ctcdecoder
import numpy as np


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.decoder = build_ctcdecoder([''] + vocab[1:])

    def ctc_decode(self, inds: List[int]) -> str:
        s = []
        last_char = ""
        for c in self.decode(inds):
            if c == last_char:
                continue
            if c != self.EMPTY_TOK:
                s.append(c)
            last_char = c
        return ''.join(s).replace('  ', ' ')

    def ctc_beam_search(self, probs: torch.tensor,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self)
        tmp = self.decoder.decode_beams(logits=probs.detach().cpu().numpy(), beam_width=beam_size)
        hypos = [Hypothesis(el[0], el[-1]) for el in tmp]

        return hypos
