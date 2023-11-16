import argparse
import collections
import warnings

import numpy as np
import torch

import tts.loss as module_loss
import tts.metric as module_metric
import tts.model as module_arch
from tts.trainer import Trainer
from tts.utils import prepare_device
from tts.utils.object_loading import get_dataloaders
from tts.utils.parse_config import ConfigParser
import pyworld as pw
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path
import json
import scipy


def extract_pitch(wave, fs):
    _f0, t = pw.dio(wave, fs)    # raw pitch extractor
    return pw.stonemask(wave, _f0, t, fs)  # pitch refinement


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def wave2pitch(wave, sr):
    pitch = extract_pitch(wave.view(-1).double().numpy(), sr).reshape(-1)
    nonzero_pitch_y = pitch[pitch > 0]
    nonzero_pitch_x = np.arange(0, pitch.shape[-1])[pitch > 0]
    interpolate_f = scipy.interpolate.interp1d(nonzero_pitch_y, nonzero_pitch_x, kind="linear",
                                               fill_value=(nonzero_pitch_y[0], nonzero_pitch_y[-1]),
                                               bounds_error=False)
    fixed_pitch = np.log(interpolate_f(np.arange(0, pitch.shape[-1])))
    return fixed_pitch


def spectrogram2energy(spectrogram):
    return spectrogram.norm(2, dim=-1).reshape(-1).numpy()


def main(config):
    # text_encoder
    text_encoder = config.get_text_encoder()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)
    sr = dataloaders['train'].dataset.config_parser["preprocessing"]["sr"]

    pitch_scaler = StandardScaler()
    min_pitch    = 1e9
    max_pitch    = -1e9
    min_energy   = 1e9
    max_energy   = -1e9 

    '''
    for el in tqdm(dataloaders['train'].dataset):
        energy = spectrogram2energy(el['spectrogram'])
        pitch = wave2pitch(el['audio'], sr).reshape(-1)
        pitch_scaler.partial_fit(pitch.reshape(-1, 1))
        max_pitch = max(max_pitch, pitch.max())
        min_pitch = min(min_pitch, pitch.min())
        max_energy = max(max_energy, energy.max().item())
        min_energy = min(min_energy, energy.min().item())
    
    print(f"Min pitch: {min_pitch} . Max pitch: {max_pitch}")
    print(f"Min energy: {min_energy} . Max energy: {max_energy}")
    print(f"Mean Pitch : {pitch_scaler.mean_[0]} . Std pitch: {pitch_scaler.scale_[0]}")
    with open(config['name'] + '.json', 'w') as file:
        json.dump({
            'min_pitch': min_pitch,
            'max_pitch': max_pitch,
            "min_energy": min_energy,
            "max_energy": max_energy,
            "mean_pitch": pitch_scaler.mean_[0],
            'std_pitch': pitch_scaler.scale_[0]
        }, file)
    '''
    kwargs = json.load(open(config['name'] + '.json'))
    pitch_scaler.mean_ = np.array([kwargs['mean_pitch']])
    pitch_scaler.scale_ = np.array([kwargs['std_pitch']])

    for t in ['val', 'train']:
        for el in tqdm(dataloaders[t].dataset):
            path = Path(el['audio_path'])
            pitch_path = path.parent / (str(el['idx']) + '-pitch.npy')
            energy_path = path.parent / (str(el['idx']) + '-energy.npy')
            fixed_pitch = pitch_scaler.transform(wave2pitch(el['audio'], sr).reshape(-1, 1)).reshape(-1)
            np.save(pitch_path, fixed_pitch)
            np.save(energy_path, spectrogram2energy(el['spectrogram']))
    

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
