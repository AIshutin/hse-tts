from typing import List, NamedTuple

import torch

from .bpe_text_encoder import BPETextEncoder
from pyctcdecode import build_ctcdecoder
import numpy as np


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCBPETextEncoder(BPETextEncoder):
    EMPTY_TOK = "<pad>"

    def __init__(self, file, **kwargs):
        super().__init__(file=file, pad_token=self.EMPTY_TOK)
        alphabet_dict = self.tokenizer.get_vocab()
        alphabet_list = []
        for el, i in sorted(alphabet_dict.items(), key=lambda x: x[1]):
            assert(len(alphabet_list) == i)
            alphabet_list.append(el)
        assert(alphabet_list[0] == self.EMPTY_TOK)

        self.decoder = build_ctcdecoder(alphabet_list)

    def ctc_decode(self, inds: List[int]) -> str:
        ids = []
        last_id = ""
        for c in inds:
            if c == last_id:
                continue
            if c != 0:
                ids.append(c)
            last_id = c
        return self.decode(ids)

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
