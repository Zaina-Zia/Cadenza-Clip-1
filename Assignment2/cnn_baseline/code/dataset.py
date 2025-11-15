import json
import os
import warnings
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

try:
    torchaudio.set_audio_backend("ffmpeg")
except Exception:
    pass


class MFCCDataset(Dataset):
    def __init__(self, data_root, split="train", n_mfcc=40, target_sr=16000, n_mels=64):
        self.data_root = Path(data_root)
        self.split = split
        self.n_mfcc = n_mfcc
        self.target_sr = int(target_sr)
        self.n_mels = n_mels

        # audio directories
        if split in ("train", "val"):
            # both train and val audio files are physically in this folder
            self.audio_dir = self.data_root / "audio" / "train" / "signals"
            meta_file = {
                "train": self.data_root / "metadata" / "cadenza_data.trainval_train_metadata.json",
                "val":   self.data_root / "metadata" / "cadenza_data.trainval_val_metadata.json"
            }[split]
        elif split == "eval":
            self.audio_dir = self.data_root / "audio" / "eval" / "signals"
            meta_file = self.data_root / "metadata" / "eval_metadata.json"
        else:
            raise ValueError(f"Unknown split: {split}")

        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        with open(meta_file, "r", encoding="utf-8") as fh:
            self.metadata = json.load(fh)

        # extract ids
        self.signal_ids = [item["signal"] for item in self.metadata]

        # MFCC transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.target_sr,
            n_mfcc=self.n_mfcc,
            melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": self.n_mels},
        )

    def __len__(self):
        return len(self.signal_ids)

    def _safe_path_str(self, p: Path) -> str:
        pstr = str(p)
        if os.name == "nt":
            if not pstr.startswith("\\\\?\\"):
                pstr = "\\\\?\\" + pstr
        return pstr

    def _load_audio(self, audio_path: Path):
        path_str = self._safe_path_str(audio_path)
        try:
            waveform, sr = torchaudio.load(path_str)
        except Exception as e:
            warnings.warn(f"Could not load {audio_path}: {e}")
            return None, None

        if waveform.numel() == 0:
            warnings.warn(f"Empty waveform: {audio_path}")
            return None, None

        # mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != self.target_sr:
            try:
                waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
            except Exception as e:
                warnings.warn(f"Resample failed for {audio_path}: {e}")
                return None, None

        return waveform, self.target_sr

    def __getitem__(self, idx):
        signal = self.signal_ids[idx]
        audio_path = self.audio_dir / f"{signal}.flac"

        waveform, sr = self._load_audio(audio_path)
        if waveform is None:
            if self.split == "eval":
                return None
            else:
                return None

        try:
            mfcc = self.mfcc_transform(waveform)  # [1, n_mfcc, T]
        except Exception as e:
            warnings.warn(f"MFCC failed for {audio_path}: {e}")
            return None

        # keep channel dim (1) so model gets [B,1,n_mfcc,T]
        # normalize per-sample
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std(unbiased=False) + 1e-9)

        if self.split in ("train", "val"):
            corr = float(self.metadata[idx].get("correctness", 0.0))
            target = torch.tensor([max(0.0, min(1.0, corr))], dtype=torch.float32)  # shape [1]
            return mfcc, target, signal

        # eval
        return mfcc, signal


def collate_train(batch):
    # batch entries: (mfcc, target, signal) or None
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys, ids = zip(*batch)   # xs: tensors [1,n_mfcc,T]
    # pad to max time length in batch
    maxT = max(x.shape[2] for x in xs)
    padded = []
    for x in xs:
        pad = maxT - x.shape[2]
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        padded.append(x)
    x = torch.stack(padded, dim=0)   # [B,1,n_mfcc,T]
    y = torch.stack(ys, dim=0)       # [B,1]
    return x, y, list(ids)


def collate_eval(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    # normalize shape to (mfcc, id)
    norm_batch = []
    for b in batch:
        if len(b) == 2:
            norm_batch.append((b[0], b[1]))
        elif len(b) == 3:
            norm_batch.append((b[0], b[2]))
        else:
            continue
    if not norm_batch:
        return None
    xs, ids = zip(*norm_batch)
    maxT = max(x.shape[2] for x in xs)
    padded = []
    for x in xs:
        pad = maxT - x.shape[2]
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        padded.append(x)
    x = torch.stack(padded, dim=0)  # [B,1,n_mfcc,T]
    return x, list(ids)
