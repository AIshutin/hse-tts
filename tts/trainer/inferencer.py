from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from tts.utils import get_WaveGlow
from .. import waveglow
import json


class Inferencer:
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            dataloaders,
            text_encoder,
            logger,
            out_dir,
            device,
            checkpoint_path,
            **kwargs
    ):
        self.text_encoder = text_encoder
        self.dataloader = dataloaders["test"]
        model.load_state_dict(torch.load(checkpoint_path, device)["state_dict"])
        self.model = model
        self.device = device
        self.logger = logger
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.waveglow = get_WaveGlow().to(device)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "text_encoded", "src_pos", "mel_pos", "alignment", 
                               "pitch", "energy", "alpha_duration", "alpha_pitch", "alpha_energy"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def train(self):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.eval()
        with torch.no_grad():
            batches = []
            for batch in tqdm(
                    self.dataloader,
                    desc="test",
                    total=len(self.dataloader),
            ):
                batch = self.process_test_batch(batch)
                batches.append(batch)

            table = []

            for i, batch in enumerate(batches):
                spectrogram_hat = batch['spectrogram_hat'].T.squeeze(-1).unsqueeze(0)
                text = batch['text']
                alpha_dur = batch['alpha_duration'][0].item()
                alpha_pitch = batch['alpha_pitch'][0].item()
                alpha_energy = batch['alpha_energy'][0].item()
                filename = f"{i + 1}-dur={alpha_dur:.2f}-pitch={alpha_pitch:.2f}-energy={alpha_energy:.2f}.wav"
                waveglow.inference.inference(spectrogram_hat, self.waveglow, 
                                            audio_path=self.out_dir / filename)
                table.append(
                     {
                          "text": text,
                          "filename": filename,
                          "pitch": alpha_pitch,
                          "duration": alpha_dur,
                          "energy": alpha_energy
                     }
                )
            
            with open(self.out_dir / "description.json", "w") as file:
                json.dump(table, file, indent=4)
    
    def process_test_batch(self, batch):
        batch = self.move_batch_to_device(batch, self.device)
        outputs = self.model(text_encoded=batch['text_encoded'],
                             src_pos=batch['src_pos'],
                             alpha_duration=batch['alpha_duration'],
                             alpha_pitch=batch['alpha_pitch'],
                             alpha_energy=batch['alpha_energy'])
        assert(type(outputs) is dict)
        batch.update(outputs)
        return batch

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "text_encoded", "src_pos", "mel_pos", "alignment", 
                               "pitch", "energy", "alpha_duration", "alpha_pitch", "alpha_energy"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch
