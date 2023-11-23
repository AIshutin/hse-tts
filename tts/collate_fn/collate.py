import logging
from typing import List, Iterable
import torch
from torch.nn import functional as F
import numpy as np

logger = logging.getLogger(__name__)


def pad_1D(inputs, PAD=0):
    assert(len(inputs[0].shape) == 1)

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs, PAD=0):
    assert(len(inputs[0].shape) == 1)

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    assert(len(inputs[0].shape) == 2)

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_2D_tensor(inputs, maxlen=None):
    assert(len(inputs[0].shape) == 2)

    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["text_encoded"].squeeze(0) for ind in cut_list]
    mel_targets = [batch[ind]["spectrogram"].squeeze(0) for ind in cut_list]
    durations = [batch[ind]["alignment"] for ind in cut_list]
    audio_paths = [batch[ind]["audio_path"] for ind in cut_list]
    energy = [batch[ind]["energy"].squeeze(0) for ind in cut_list]
    pitch  = [batch[ind]["pitch"].squeeze(0) for ind in cut_list]
    raw_texts = [batch[ind]["text"] for ind in cut_list]
    audio_durations = [batch[ind]["duration"] for ind in cut_list]
    audios = [batch[ind]['audio'].squeeze(0) for ind in cut_list]
    alpha_durations = [batch[ind].get('alpha_duration', 1.0) for ind in cut_list]
    alpha_pitch = [batch[ind].get('alpha_pitch', 1.0) for ind in cut_list]
    alpha_energy = [batch[ind].get('alpha_energy', 1.0) for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.shape[0])

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.shape[0])

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    mel_targets = pad_2D_tensor(mel_targets)
    pitch = pad_1D_tensor(pitch)
    energy = pad_1D_tensor(energy)
    audios = pad_1D(audios)


    out = {"text_encoded": texts,
           "spectrogram": mel_targets,
           "duration": audio_durations,
           "alignment": durations.to(torch.long),
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "max_mel_len": max_mel_len,
           "spectrogram_length": torch.from_numpy(length_mel).to(torch.long),
           "pitch": pitch,
           "energy": energy,
           "text": raw_texts,
           "audio_path": audio_paths,
           "audio": audios,
           "alpha_duration": torch.tensor(alpha_durations),
           "alpha_pitch": torch.tensor(alpha_pitch),
           "alpha_energy": torch.tensor(alpha_energy)}

    return out


def collate_fn_tensor(batch: List[dict]):
    batchsize = len(batch)
    cut_list = list(range(batchsize))
    output = reprocess_tensor(batch, cut_list)
    return output