from typing import List, NamedTuple

import torch

from .bpe_text_encoder import BPETextEncoder
from pyctcdecode import build_ctcdecoder
import numpy as np
import math
from collections import defaultdict


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCBPETextEncoder(BPETextEncoder):
    EMPTY_TOK = "<pad>"

    def __init__(self, file, **kwargs):
        super().__init__(file=file, pad_token=self.EMPTY_TOK, alpha=0.5,
                         beta=1.0, lm_path="3-gram.arpa", )
        alphabet_dict = self.tokenizer.get_vocab()
        alphabet_list = []
        for el, i in sorted(alphabet_dict.items(), key=lambda x: x[1]):
            assert(len(alphabet_list) == i)
            alphabet_list.append(el)
        assert(alphabet_list[0] == self.EMPTY_TOK)

        self.decoder = build_ctcdecoder(alphabet_list,
                                        kenlm_model_path=lm_path,
                                        alpha=alpha,
                                        beta=beta)

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
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self)
        states = dict()
        states[((), 0)] = 1.0
        for prob_list in probs:
            new_states = defaultdict(float)
            for i, p in enumerate(prob_list):
                for state, state_p in states.items():
                    prefix = state[0]
                    if state[1] != i and i != 0:
                        prefix = tuple(list(prefix) + [i])
                    new_states[(prefix, i)] += state_p * p.item()
            states = dict()
            for el in sorted(new_states.items(), key=lambda x: -x[1])[:beam_size]:
                states[el[0]] = el[1]
        hypos = [Hypothesis(self.decode(el[0][0]), el[-1]) for el in states.items()]
        hypos.sort(key=lambda x: -x.prob)

        return hypos

    def ctc_beam_search_fast(self, probs: torch.tensor,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self)
        tmp = self.decoder.decode_beams(logits=probs.log().detach().cpu().numpy(), beam_width=beam_size)
        hypos = [Hypothesis(el[0], math.exp(el[-1])) for el in tmp]

        return hypos
