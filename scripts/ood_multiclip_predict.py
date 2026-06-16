from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import read_csv_rows
from src.metrics import compute_binary_metrics, logits_to_fake_prob
from src.models.model_factory import build_model
from src.utils import ensure_dir, get_device, save_json


def _clip_indices(total: int, num_frames: int, clip_count: int, clip_idx: int, crop_ratio: float) -> np.ndarray:
    if total <= 0:
        return np.zeros(num_frames, dtype=np.int64)
    if total < num_frames:
        base = np.arange(total, dtype=np.int64)
        pad = np.full(num_frames - total, total - 1, dtype=np.int64)
        return np.concatenate([base, pad])
    if clip_count <= 1:
        return np.linspace(0, total - 1, num_frames).round().astype(np.int64)

    crop = int(round(total * crop_ratio))
    crop = min(total, max(num_frames, crop))
    max_start = max(0, total - crop)
    starts = np.linspace(0, max_start, clip_count).round().astype(np.int64)
    start = int(starts[min(clip_idx, len(starts) - 1)])
    return np.linspace(start, start + crop - 1, num_frames).round().astype(np.int64)


def read_video_clip(
    path: str | Path,
    num_frames: int,
    image_size: int,
    clip_count: int,
    clip_idx: int,
    crop_ratio: float,
) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = _clip_indices(total, num_frames, clip_count, clip_idx, crop_ratio)
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
    arr = np.stack(frames).astype(np.float32) / 255.0
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(arr)


class MultiClipDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        num_frames: int,
        image_size: int,
        clip_count: int,
        crop_ratio: float,
        max_samples: int | None = None,
    ) -> None:
        self.rows = read_csv_rows(csv_path)
        if max_samples is not None and max_samples > 0:
            self.rows = self.rows[:max_samples]
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.image_size = image_size
        self.clip_count = clip_count
        self.crop_ratio = crop_ratio

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows[idx]
        video_path = self.data_root / row["rel_path"]
        clips = [
            read_video_clip(video_path, self.num_frames, self.image_size, self.clip_count, clip_idx, self.crop_ratio)
            for clip_idx in range(self.clip_count)
        ]
        return {
            "clips": torch.stack(clips, dim=0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "meta": {
                "rel_path": row["rel_path"],
                "generator": row.get("generator", "unknown"),
                "video_id": row.get("video_id", str(idx)),
            },
        }


def collate_multiclip(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "clips": torch.stack([item["clips"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
        "meta": [item["meta"] for item in batch],
    }


def aggregate(values: torch.Tensor, mode: str, topk: int) -> torch.Tensor:
    if mode == "mean":
        return values.mean(dim=1)
    if mode == "topk_mean":
        k = min(topk, values.shape[1])
        return values.topk(k, dim=1).values.mean(dim=1)
    raise ValueError(f"Unknown aggregate mode: {mode}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/GenVideo-Val")
    parser.add_argument("--csv", default="data/GenVideo-Val/splits/ood_test.csv")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=0, help="0 means use checkpoint args")
    parser.add_argument("--image-size", type=int, default=0, help="0 means use checkpoint args")
    parser.add_argument("--clip-count", type=int, default=5)
    parser.add_argument("--clip-crop-ratio", type=float, default=0.6)
    parser.add_argument("--aggregate", choices=["mean", "topk_mean"], default="mean")
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out-dir", default="outputs/eval_multiclip")
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    model_mode = ckpt_args.get("model", "stf3_new")
    num_frames = args.num_frames or int(ckpt_args.get("num_frames", 8))
    image_size = args.image_size or int(ckpt_args.get("image_size", 224))

    model = build_model(
        mode=model_mode,
        pretrained_spatial=bool(ckpt_args.get("pretrained_spatial", False)),
        foundation_backbone=ckpt_args.get("foundation_backbone", "dinov2_vits14"),
        freeze_foundation=not bool(ckpt_args.get("finetune_foundation", False)),
        local_files_only=bool(ckpt_args.get("local_files_only", False)),
        image_size=image_size,
        wavelet_aug_prob=float(ckpt_args.get("wavelet_aug_prob", 0.0)),
        branch_dropout=float(ckpt_args.get("branch_dropout", 0.0)),
        d3_loss=ckpt_args.get("d3_loss", "l2"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = MultiClipDataset(
        args.csv,
        args.data_root,
        num_frames,
        image_size,
        args.clip_count,
        args.clip_crop_ratio,
        max_samples=args.max_samples or None,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_multiclip,
    )
    out_dir = ensure_dir(args.out_dir)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    rows: list[dict[str, object]] = []
    labels: list[int] = []
    probs: list[float] = []
    start = time.time()

    for batch in tqdm(loader, desc="predict"):
        clips = batch["clips"].to(device, non_blocking=True)
        b, c, t, ch, h, w = clips.shape
        flat = clips.view(b * c, t, ch, h, w)
        with torch.amp.autocast(autocast_device, enabled=args.amp and device.type == "cuda"):
            out = model(flat, return_dict=True) if hasattr(model, "branch_names") else model(flat)
            logits = out["logits"] if isinstance(out, dict) else out
        clip_probs = logits_to_fake_prob(logits).view(b, c)
        video_probs = aggregate(clip_probs, args.aggregate, args.topk).detach().cpu().numpy().tolist()

        aux_video_probs: dict[str, list[float]] = {}
        branch_video_weights: dict[str, list[float]] = {}
        if isinstance(out, dict):
            for name, aux_logits in out.get("aux_logits", {}).items():
                aux_values = logits_to_fake_prob(aux_logits).view(b, c)
                aux_video_probs[name] = aggregate(aux_values, args.aggregate, args.topk).detach().cpu().numpy().tolist()
            details = out.get("details", {})
            if "branch_weights" in details:
                branch_names = list(out.get("branch_names", []))
                weights = details["branch_weights"].view(b, c, -1).mean(dim=1).detach().cpu().numpy().tolist()
                for j, name in enumerate(branch_names):
                    branch_video_weights[name] = [float(item[j]) for item in weights if j < len(item)]

        y = batch["labels"].cpu().numpy().tolist()
        for i, (meta, yy, pp) in enumerate(zip(batch["meta"], y, video_probs)):
            pred = 1 if pp >= 0.5 else 0
            row = dict(meta)
            row.update({"label": yy, "fake_prob": pp, "pred": pred, "correct": int(yy == pred)})
            for name, values in aux_video_probs.items():
                row[f"fake_prob_{name}"] = values[i]
            for name, values in branch_video_weights.items():
                if i < len(values):
                    row[f"branch_weight_{name}"] = values[i]
            rows.append(row)
        labels.extend(int(v) for v in y)
        probs.extend(float(v) for v in video_probs)

    metrics = compute_binary_metrics(labels, probs).to_dict()
    metrics.update(
        {
            "seconds": time.time() - start,
            "num_samples": len(labels),
            "model": model_mode,
            "clip_count": args.clip_count,
            "clip_crop_ratio": args.clip_crop_ratio,
            "aggregate": args.aggregate,
            "topk": args.topk if args.aggregate == "topk_mean" else 0,
        }
    )
    save_json(metrics, out_dir / "metrics.json")
    with (out_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else ["label", "fake_prob", "pred"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(metrics)
    print(f"[write] {out_dir / 'metrics.json'}")
    print(f"[write] {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
