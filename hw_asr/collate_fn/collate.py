import logging
from typing import List, Iterable
import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def pad_merge_tensors(tensors: List[torch.Tensor], mx_length: int, fill_with=0):
    tensors = [F.pad(el, (0, mx_length - el.shape[-1]), value=fill_with) for el in tensors]
    # print('t length', len(tensors))
    return torch.cat(tensors)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}
    for key in 'duration', 'text', 'audio_path':
        result_batch[key] = [el[key] for el in dataset_items]
    

    # print('DI', len(dataset_items))
    #print('spectr', dataset_items[0]['spectrogram'].min(), dataset_items[0]['spectrogram'].max(), dataset_items[0]['spectrogram'].mean())
    #print('audio', dataset_items[0]['audio'].min(), dataset_items[0]['audio'].max(), dataset_items[0]['audio'].mean())

    for key in 'audio', 'spectrogram', 'text_encoded':
        tensors = [el[key] for el in dataset_items]
        lengths = [el.shape[-1] for el in tensors]
        # print(key, 'length', lengths)
        
        result_batch[key + '_length'] = torch.tensor(lengths)
        value = 0
        if key == 'spectrogram':
            value = -20
        result_batch[key] = pad_merge_tensors(tensors, mx_length=max(lengths), fill_with=value)

    return result_batch