from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def read_csv_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _uniform_indices(total: int, num_frames: int, random_sample: bool = False) -> np.ndarray:
    if total <= 0:
        return np.zeros(num_frames, dtype=np.int64)
    if total >= num_frames:
        if random_sample and total > num_frames:
            # random temporal crop, then uniform sample inside the crop
            crop = np.random.randint(num_frames, total + 1)
            start = np.random.randint(0, total - crop + 1)
            return np.linspace(start, start + crop - 1, num_frames).round().astype(np.int64)
        return np.linspace(0, total - 1, num_frames).round().astype(np.int64)
    base = np.arange(total, dtype=np.int64)
    pad = np.full(num_frames - total, total - 1, dtype=np.int64)
    return np.concatenate([base, pad])


def read_video_frames(
    path: str | Path,
    num_frames: int = 16,
    image_size: int = 224,
    random_sample: bool = False,
) -> torch.Tensor:
    """Read video as float tensor [T, 3, H, W] in [0, 1]."""
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = _uniform_indices(total, num_frames, random_sample=random_sample)
    frames: list[np.ndarray] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            if frames:
                frame = frames[-1].copy()
            else:
                frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()

    arr = np.stack(frames).astype(np.float32) / 255.0  # [T,H,W,3]
    arr = np.transpose(arr, (0, 3, 1, 2))  # [T,3,H,W]
    return torch.from_numpy(arr)


class GenVideoValDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        num_frames: int = 16,
        image_size: int = 224,
        train: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.image_size = image_size
        self.train = train
        self.rows = read_csv_rows(self.csv_path)
        if max_samples is not None and max_samples > 0:
            self.rows = self.rows[:max_samples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows[idx]
        video_path = self.data_root / row["rel_path"]
        frames = read_video_frames(
            video_path,
            num_frames=self.num_frames,
            image_size=self.image_size,
            random_sample=self.train,
        )
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return {
            "frames": frames,
            "label": label,
            "rel_path": row["rel_path"],
            "generator": row.get("generator", "unknown"),
            "video_id": row.get("video_id", str(idx)),
        }
