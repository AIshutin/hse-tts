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
from tts.audio.hparams_audio import filter_length, hop_length, win_length, sampling_rate, \
                                    n_mel_channels, mel_fmax
from torchaudio.transforms import InverseMelScale


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
    fixed_pitch = np.log1p(interpolate_f(np.arange(0, pitch.shape[-1])))
    return fixed_pitch


inverseMel = InverseMelScale(n_stft=filter_length, n_mels=n_mel_channels, f_max=mel_fmax)

def spectrogram2energy(spectrogram):
    spectrogram = spectrogram[0]
    out = inverseMel(spectrogram.T).T
    return out.norm(2, dim=-1).reshape(-1).numpy()


def main(config):
    # text_encoder
    text_encoder = config.get_text_encoder()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)
    sr = dataloaders['train'].dataset.config_parser["preprocessing"]["sr"]

    # pitch_scaler = StandardScaler()
    min_pitch    = 1e9
    max_pitch    = -1e9
    min_energy   = 0.09309670329093933
    max_energy   = 1.549446702003479

    for dtype in ['train', 'val']:
        for i, el in enumerate(tqdm(dataloaders[dtype].dataset)):
            assert(el['spectrogram'].shape[-1] == 80)
            path = Path(el['audio_path'])
            energy_path = path.parent / (str(el['idx']) + '-energy.npy')
            pitch_path = path.parent / (str(el['idx']) + '-pitch.npy')

            pitch = wave2pitch(el['audio'], sr).reshape(-1)
            np.save(pitch_path, pitch)

            #energy = spectrogram2energy(el['spectrogram'])
            #np.save(energy_path, energy)

            if dtype == "train":
                #max_energy = max(max_energy, energy.max().item())
                #min_energy = min(min_energy, energy.min().item())
                max_pitch = max(max_pitch, pitch.max())
                min_pitch = min(min_pitch, pitch.min())

                if i % 100 == 0:
                    print(f"Min energy: {min_energy} . Max energy: {max_energy}")

    print(f"Min pitch: {min_pitch} . Max pitch: {max_pitch}")
    print(f"Min energy: {min_energy} . Max energy: {max_energy}")
    with open(config['name'] + '.json', 'w') as file:
        json.dump({
            'min_pitch': min_pitch,
            'max_pitch': max_pitch,
            "min_energy": min_energy,
            "max_energy": max_energy,
        }, file)
    

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
