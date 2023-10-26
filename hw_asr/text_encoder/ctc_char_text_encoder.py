from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder
from pyctcdecode import build_ctcdecoder
import numpy as np
from collections import defaultdict
import math


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, alpha=0.5, 
                 beta=1.0, lm_path="3-gram.arpa", **kwargs):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        print('ALPHA', alpha, 'BETA', beta)
        self.decoder = build_ctcdecoder([''] + [el.upper() for el in vocab[1:]],
                                        kenlm_model_path=lm_path,
                                        alpha=alpha,
                                        beta=beta)

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


    def ctc_beam_search_lm(self, probs: torch.tensor,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self)
        logits = probs.log().detach().cpu().numpy()
        tmp = self.decoder.decode_beams(logits=logits, beam_width=beam_size)
        hypos = [Hypothesis(el[0].lower(), math.exp(el[-1])) for el in tmp]

        return hypos