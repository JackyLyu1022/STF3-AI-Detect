from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import GenVideoValDataset
from src.metrics import compute_binary_metrics, logits_to_fake_prob
from src.models.model_factory import build_model
from src.train import collate_fn
from src.utils import ensure_dir, get_device, save_json


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/GenVideo-Val")
    parser.add_argument("--csv", default="data/GenVideo-Val/splits/random_test.csv")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=0, help="0 means use checkpoint args")
    parser.add_argument("--image-size", type=int, default=0, help="0 means use checkpoint args")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out-dir", default="outputs/eval")
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    model_mode = ckpt_args.get("model", "spatial")
    pretrained_spatial = bool(ckpt_args.get("pretrained_spatial", False))
    num_frames = args.num_frames or int(ckpt_args.get("num_frames", 16))
    image_size = args.image_size or int(ckpt_args.get("image_size", 224))

    model = build_model(
        mode=model_mode,
        pretrained_spatial=pretrained_spatial,
        foundation_backbone=ckpt_args.get("foundation_backbone", "dinov2_vits14"),
        freeze_foundation=not bool(ckpt_args.get("finetune_foundation", False)),
        local_files_only=bool(ckpt_args.get("local_files_only", False)),
        image_size=image_size,
        wavelet_aug_prob=float(ckpt_args.get("wavelet_aug_prob", 0.0)),
        wavelet_aug_mode=ckpt_args.get("wavelet_aug_mode", "batch"),
        branch_dropout=float(ckpt_args.get("branch_dropout", 0.0)),
        d3_loss=ckpt_args.get("d3_loss", "l2"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = GenVideoValDataset(args.csv, args.data_root, num_frames, image_size, train=False, max_samples=args.max_samples or None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate_fn)
    out_dir = ensure_dir(args.out_dir)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    rows = []
    labels, probs = [], []
    start = time.time()
    for batch in tqdm(loader, desc="predict"):
        frames = batch["frames"].to(device, non_blocking=True)
        with torch.amp.autocast(autocast_device, enabled=args.amp and device.type == "cuda"):
            out = model(frames, return_dict=True) if hasattr(model, "branch_names") else model(frames)
            logits = out["logits"] if isinstance(out, dict) else out
        p = logits_to_fake_prob(logits).detach().cpu().numpy().tolist()
        pred = [1 if x >= 0.5 else 0 for x in p]
        y = batch["labels"].cpu().numpy().tolist()
        aux_probs: dict[str, list[float]] = {}
        branch_weights = None
        branch_names = []
        if isinstance(out, dict):
            for name, aux_logits in out.get("aux_logits", {}).items():
                aux_probs[name] = logits_to_fake_prob(aux_logits).detach().cpu().numpy().tolist()
            details = out.get("details", {})
            if "branch_weights" in details:
                branch_weights = details["branch_weights"].detach().cpu().numpy().tolist()
                branch_names = list(out.get("branch_names", []))
        for i, (meta, yy, pp, pr) in enumerate(zip(batch["meta"], y, p, pred)):
            row = dict(meta)
            row.update({"label": yy, "fake_prob": pp, "pred": pr, "correct": int(yy == pr)})
            for name, vals in aux_probs.items():
                row[f"fake_prob_{name}"] = vals[i]
            if branch_weights is not None:
                for j, name in enumerate(branch_names):
                    if j < len(branch_weights[i]):
                        row[f"branch_weight_{name}"] = branch_weights[i][j]
            rows.append(row)
        labels.extend(y)
        probs.extend(p)

    metrics = compute_binary_metrics(labels, probs).to_dict()
    metrics["seconds"] = time.time() - start
    metrics["num_samples"] = len(labels)
    metrics["model"] = model_mode
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
