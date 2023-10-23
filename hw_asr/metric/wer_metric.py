from typing import List
from functools import lru_cache

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_wer
import logging


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = kwargs.get('beam_size', 5)

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        with torch.no_grad():
            predictions = log_probs.detach()
            lengths = log_probs_length.detach()
            for log_prob_vec, length, target_text in zip(predictions, lengths, text):
                target_text = BaseTextEncoder.normalize_text(target_text)
                if hasattr(self.text_encoder, "ctc_beam_search"):
                    pred_text = self.text_encoder.ctc_beam_search(log_prob_vec[:length], self.beam_size)[0].text
                else:
                    logging.warning('CTC Beam Search is not implemented, but required in BeamSearchWERMetric')
                    pred_text = self.text_encoder.decode(log_prob_vec[:length])
                wers.append(calc_wer(target_text, pred_text))
            return sum(wers) / len(wers)