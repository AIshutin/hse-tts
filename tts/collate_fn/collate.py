import logging
from typing import List, Iterable
import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def pad_merge_tensors(tensors: List[torch.Tensor], mx_length: int, fill_with=0):
    if len(tensors[0].shape) <= 2:
        tensors = [F.pad(el, (0, mx_length - el.shape[-1]), 
                        value=fill_with) for el in tensors]
    else:
        tensors = [F.pad(el, (0, 0, 0, mx_length - el.shape[1]), 
                        value=fill_with) for el in tensors]

    return torch.cat(tensors)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    aux_keys = {"duration", "text", "audio_path"}
    unused_keys = {'idx'}
    all_keys = set(dataset_items[0].keys())
    result_batch = {}
    for key in aux_keys:
        result_batch[key] = [el[key] for el in dataset_items]
    
    for key in all_keys.difference(aux_keys).difference(unused_keys):
        tensors = [el[key] for el in dataset_items]
        lengths = [el.shape[0] if len(el.shape) == 1 else el.shape[1] for el in tensors]
        
        result_batch[key + '_length'] = torch.tensor(lengths)
        value = 0

        # should pitch be zero for padding purpose?
        if key == 'spectrogram':
            value = -20

        result_batch[key] = pad_merge_tensors(tensors, mx_length=max(lengths), fill_with=value)

    return result_batch