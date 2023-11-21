import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from tts.base import BaseTrainer
from tts.logger.utils import plot_spectrogram_to_buf
from tts.utils import inf_loop, MetricTracker
from tts.utils import get_WaveGlow
from .. import waveglow
from ..logger.wandb import WanDBWriter
from numpy import inf
import json
import os


def make_audio_item(wave, writer, config):
    if not isinstance(writer, WanDBWriter):
        raise NotImplemented(f"{str(type(writer))};{writer} is not supported here")
    return writer.wandb.Audio(wave, sample_rate=config['preprocessing']['sr'])


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            device,
            dataloaders,
            text_encoder,
            main_config,
            logger,
            run_id,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
            **kwargs
    ):
        self.lr_scheduler = lr_scheduler
        config = json.loads(main_config)
        config['trainer']['save_dir'] = os.path.join(config['trainer']['save_dir'], run_id)
        super().__init__(model=model, criterion=criterion, metrics=metrics,
                         optimizer=optimizer, device=device, logger=logger, 
                         config=config)
        self.skip_oom = skip_oom
        self.text_encoder = text_encoder
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.log_step = 100

        self.train_metrics = MetricTracker(
            "loss", "loss_spectrogram", "loss_pitch", "loss_align", "loss_energy",
            "grad norm", *[m.name for m in self.metrics if '(bs)' not in m.name], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", "loss_spectrogram", "loss_pitch", "loss_align", "loss_energy",
            *[m.name for m in self.metrics], writer=self.writer
        )
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

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        with torch.autograd.set_detect_anomaly(True):
            for batch_idx, batch in enumerate(
                    tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
            ):
                try:
                    batch = self.process_batch(
                        batch,
                        is_train=True,
                        metrics=self.train_metrics,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                self.train_metrics.update("grad norm", self.get_grad_norm())
                if batch_idx % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(batch_idx), batch["loss"].item()
                        )
                    )
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                    self._log_predictions(**batch)
                    self._log_spectrogram(batch["spectrogram"], batch['spectrogram_hat'])
                    self._log_scalars(self.train_metrics)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()
                if batch_idx >= self.len_epoch:
                    break
            log = last_train_metrics

            for part, dataloader in self.evaluation_dataloaders.items():
                if part == "test":
                    val_log = self._evaluation_epoch_test(epoch, part, dataloader)
                else:
                    val_log = self._evaluation_epoch(epoch, part, dataloader)
                    log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

            return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        assert(type(outputs) is dict)
        batch.update(outputs)

        loss_dict = self.criterion(**batch)
        assert(type(loss_dict) is dict and 'loss' in loss_dict)
        batch.update(loss_dict)
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.step()
        
        metrics.update("loss", batch["loss"].item())
        metrics.update("loss_spectrogram", batch["loss_spectrogram"].item())
        metrics.update("loss_pitch", batch["loss_pitch"].item())
        metrics.update("loss_align", batch["loss_align"].item())
        metrics.update("loss_energy", batch["loss_energy"].item())
        for met in self.metrics:
            if met.name in metrics.tracked_metrics:
                metrics.update(met.name, met(**batch))
        return batch
    
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


    def _evaluation_epoch_test(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            batches = []
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_test_batch(batch)
                batches.append(batch)
            
            self.writer.set_step(epoch * self.len_epoch, part)
            rows = {}
            for i, batch in enumerate(batches):
                with torch.no_grad():
                    spectrogram_hat = batch['spectrogram_hat'].T.squeeze(-1).unsqueeze(0)
                    synthesized_hat = waveglow.inference.inference(spectrogram_hat, self.waveglow)
                text = batch['text']
                audio_path = batch['audio_path']
                self.writer
                rows[str(i) + audio_path[0]] = {
                    "text": text,
                    "synthesized_hat": make_audio_item(synthesized_hat, self.writer, self.config),
                    "duration": batch['alpha_duration'][0].item(),
                    "pitch": batch['alpha_pitch'][0].item(),
                    "energy": batch['alpha_energy'][0].item(),
                    "name": audio_path
                }
            self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))


    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            if part != "test":
                self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch, is_test=part == "test")
            self._log_spectrogram(batch["spectrogram"], batch['spectrogram_hat'])

        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            text,
            audio,
            spectrogram,
            spectrogram_hat,
            spectrogram_length,
            audio_path,
            examples_to_log=10,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return
        
        tuples = list(zip(text, audio, spectrogram, spectrogram_hat,
            spectrogram_length, audio_path))
        shuffle(tuples)
        rows = {}
        for text, audio, spectrogram, spectrogram_hat, spectrogram_length, \
            audio_path in tuples[:examples_to_log]:
            with torch.no_grad():
                spectrogram_hat = spectrogram_hat[:spectrogram_length].T.unsqueeze(0)
                spectrogram     = spectrogram[:spectrogram_length].T.unsqueeze(0)
                synthesized_hat = waveglow.inference.inference(spectrogram_hat, self.waveglow)
                synthesized = waveglow.inference.inference(spectrogram, self.waveglow)
            
            self.writer
            rows[Path(audio_path).name] = {
                "text": text,
                "audio": make_audio_item(audio, self.writer, self.config),
                "synthesized": make_audio_item(synthesized, self.writer, self.config),
                "synthesized_hat": make_audio_item(synthesized_hat, self.writer, self.config),
                "name": Path(audio_path).name
            }
        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch, spectrogram_hat_batch):
        idx = random.randint(0, spectrogram_batch.shape[0] - 1)
        spectrogram = spectrogram_batch[idx].T.detach().cpu()
        spectrogram_hat = spectrogram_hat_batch[idx].T.detach().cpu()
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        image_hat = PIL.Image.open(plot_spectrogram_to_buf(spectrogram_hat))
        self.writer.add_image("spectrogram", ToTensor()(image))
        self.writer.add_image("spectrogram_hat", ToTensor()(image_hat))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
