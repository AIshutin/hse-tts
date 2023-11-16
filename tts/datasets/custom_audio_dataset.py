import logging
from pathlib import Path

import torchaudio

from tts.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, data, *args, **kwargs):
        index = data
        for entry in data:
            assert "path" in entry
            assert Path(entry["path"]).exists(), f"Path {entry['path']} doesn't exist"
            entry["path"] = str(Path(entry["path"]).absolute().resolve())
            entry["text"] = entry.get("text", "")
            t_info = torchaudio.info(entry["path"])
            entry["audio_len"] = t_info.num_frames / t_info.sample_rate

        super().__init__(index, *args, **kwargs)


class CustomDirAudioDataset(CustomAudioDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + '.txt')
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip()
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
