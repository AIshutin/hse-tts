import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from tts.base.base_text_encoder import BaseTextEncoder
from scipy import signal
import pyworld as pw


logger = logging.getLogger(__name__)


def extract_pitch(wave, fs):
    _f0, t = pw.dio(wave, fs)    # raw pitch extractor
    return pw.stonemask(wave, _f0, t, fs)  # pitch refinement


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            text_encoder: BaseTextEncoder,
            sr: int,
            limit=None,
            max_audio_length=None,
            max_text_length=None,
            return_pitch=True,
            return_energy=True,
    ):
        self.text_encoder = text_encoder

        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, max_audio_length, max_text_length, limit)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index: List[dict] = index
        self.return_pitch = return_pitch
        self.return_energy = return_energy
        self.sr = sr

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        text = self.text_encoder.encode(data_dict['text'])
        audio_wave = self.load_audio(audio_path)
        audio_spec = Tensor(np.load(data_dict['mel'])).unsqueeze(0)
        assert(len(audio_spec.shape) == 3)
        alignment = Tensor(np.load(data_dict['alignment']))

        out_dict = {
            "audio": audio_wave,
            "spectrogram": audio_spec,
            "duration": audio_wave.size(1) / self.sr,
            "text": data_dict["text"],
            "text_encoded": text,
            "audio_path": audio_path,
            "alignment": alignment,
            "idx": data_dict['idx'],
        }
        assert(alignment.shape[-1] == text.shape[-1])
        if self.return_energy:
            out_dict['energy'] = Tensor(np.load(data_dict['energy'])).reshape(1, -1)
        if self.return_pitch:
            out_dict['pitch'] = out_dict['energy'].clone()
            # out_dict['pitch'] = Tensor(np.load(data_dict['pitch'])).reshape(1, -1)

        return out_dict

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length, max_text_length, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = np.array([el["audio_len"] for el in index]) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        if max_text_length is not None:
            exceeds_text_length = (
                    np.array(
                        [len(BaseTextEncoder.normalize_text(el["text"])) for el in index]
                    )
                    >= max_text_length
            )
            _total = exceeds_text_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_text_length} characters. Excluding them."
            )
        else:
            exceeds_text_length = False

        records_to_filter = exceeds_text_length | exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - duration of audio (in seconds)."
            )
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
            assert "text" in entry, (
                "Each dataset item should include field 'text'"
                " - text transcription of the audio."
            )
            assert "mel" in entry, (
                "Each dataset item should include field 'mel'"
                " - mel spectrogram."
            )
            assert "alignment" in entry, (
                "Each dataset item should include field 'alignment'"
                " - alignment of the audio."
            )
            assert "energy" in entry, (
                "Each dataset item should include field 'energy'"
                " - energy of the audio."
            )
            assert "pitch" in entry, (
                "Each dataset item should include field 'pitch'"
                " - pitch of the audio."
            )
            assert "idx" in entry, (
                "Each dataset item should include field 'idx'"
                " - index of the audio."
            )
