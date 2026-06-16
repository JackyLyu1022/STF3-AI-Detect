from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torchvision import transforms
from tqdm import tqdm


OBJECTIVES = ("max_f1", "max_balanced_acc", "precision_0.95_recall_max", "max_acc")
WAVEREP_ARC = "vit_base_patch14_reg4_dinov2.lvd142m"


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def uniform_indices(total: int, num_frames: int) -> list[int]:
    if total <= 0:
        raise ValueError(f"invalid frame count: {total}")
    if num_frames <= 0:
        raise ValueError(f"invalid num_frames: {num_frames}")
    return np.linspace(0, total - 1, num_frames).round().astype(int).tolist()


def make_transform(crop_size: int):
    return transforms.Compose(
        [
            transforms.CenterCrop((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_uniform_frames(video_path: Path, num_frames: int, transform) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise RuntimeError(f"invalid frame count: {video_path}")
    frames: list[torch.Tensor] = []
    for idx in uniform_indices(total, num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"failed to read frame {idx}: {video_path}")
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(transform(frame))
    cap.release()
    return torch.stack(frames, dim=0)


def create_waverep_model(weights: Path, crop_size: int, device: str) -> torch.nn.Module:
    # The official demo uses timm's DINOv2 ViT-B/14-reg4 backbone with a 1-logit head.
    # We set pretrained=False because the official WaveRep checkpoint fully defines
    # the model weights; this avoids an unnecessary extra download.
    model = timm.create_model(WAVEREP_ARC, num_classes=1, pretrained=False, img_size=crop_size)
    dat = torch.load(weights, map_location="cpu")
    if "state_dict" in dat:
        dat = {k[6:]: dat["state_dict"][k] for k in dat["state_dict"] if k.startswith("model")}
    missing, unexpected = model.load_state_dict(dat, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.to(device).eval()


@torch.no_grad()
def score_one_video(model, video_path: Path, transform, device: str, *, num_frames: int, batch_size: int) -> dict[str, Any]:
    frames = load_uniform_frames(video_path, num_frames, transform)
    logits: list[float] = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size].to(device, non_blocking=True)
        pred = model(batch)[:, -1].detach().float().cpu().numpy()
        logits.extend(float(x) for x in pred)
    score_logit = float(np.mean(logits))
    return {
        "waverep_logit": score_logit,
        "waverep_fake_score": float(expit(score_logit)),
        "frame_logits": json.dumps(logits),
    }


def score_split(
    *,
    model,
    rows: list[dict[str, str]],
    data_root: Path,
    transform,
    device: str,
    num_frames: int,
    batch_size: int,
    max_samples: int,
    split_name: str,
) -> list[dict[str, Any]]:
    if max_samples and max_samples > 0:
        rows = rows[:max_samples]
    scored: list[dict[str, Any]] = []
    for row in tqdm(rows, desc=f"WaveRep scoring {split_name}", unit="video"):
        rel_path = row["rel_path"]
        video_path = data_root / rel_path
        item: dict[str, Any] = {
            "rel_path": rel_path,
            "video_id": row.get("video_id", ""),
            "filename": row.get("filename", video_path.name),
            "generator": row.get("generator", "unknown"),
            "label": int(row["label"]),
            "error": "",
        }
        try:
            item.update(score_one_video(model, video_path, transform, device, num_frames=num_frames, batch_size=batch_size))
        except Exception as exc:
            item.update(
                {
                    "waverep_logit": float("nan"),
                    "waverep_fake_score": float("nan"),
                    "frame_logits": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        scored.append(item)
    return scored


def valid_arrays(rows: list[dict[str, Any]], score_key: str = "waverep_fake_score") -> tuple[np.ndarray, np.ndarray]:
    y, s = [], []
    for row in rows:
        try:
            score = float(row[score_key])
        except Exception:
            continue
        if math.isfinite(score):
            y.append(int(row["label"]))
            s.append(score)
    return np.asarray(y, dtype=np.int64), np.asarray(s, dtype=np.float64)


def metrics_at(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "acc": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def objective_score(metrics: dict[str, Any], objective: str) -> tuple[float, ...]:
    if objective == "max_f1":
        return (float(metrics["f1"]), float(metrics["acc"]))
    if objective == "max_balanced_acc":
        return (float(metrics["balanced_acc"]), float(metrics["f1"]))
    if objective == "precision_0.95_recall_max":
        if float(metrics["precision"]) < 0.95:
            return (-1.0, -1.0, -1.0)
        return (float(metrics["recall"]), float(metrics["f1"]), float(metrics["acc"]))
    if objective == "max_acc":
        return (float(metrics["acc"]), float(metrics["f1"]))
    raise ValueError(objective)


def best_threshold(y: np.ndarray, score: np.ndarray, objective: str) -> tuple[float, dict[str, Any]]:
    finite = np.isfinite(score)
    y = y[finite]
    score = score[finite]
    candidates = np.unique(np.concatenate([score, np.asarray([score.min() - 1e-9, score.max() + 1e-9])]))
    best_t = float(candidates[0])
    best_m = metrics_at(y, score, best_t)
    best_s = objective_score(best_m, objective)
    for threshold in candidates:
        current = metrics_at(y, score, float(threshold))
        current_s = objective_score(current, objective)
        if current_s > best_s:
            best_t = float(threshold)
            best_m = current
            best_s = current_s
    return best_t, best_m


def apply_predictions(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        score = float(item["waverep_fake_score"])
        pred = 1 if score >= threshold else 0
        item["pred"] = pred
        item["correct"] = int(pred == int(item["label"]))
        out.append(item)
    return out


def write_markdown(path: Path, summary: dict[str, Any], result_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# WaveRep OOD comparison on GenVideo-Val",
        "",
        "WaveRep is evaluated with the official pretrained checkpoint as an inference-only external baseline. "
        "No training is performed on GenVideo-Val. Per-frame logits are averaged into a video-level logit; "
        "score thresholds are selected on OOD validation and fixed on OOD test.",
        "",
        "## Configuration",
        "",
        f"- WaveRep root: `{summary['waverep_root']}`",
        f"- Weights: `{summary['weights']}`",
        f"- Architecture: `{summary['architecture']}`",
        f"- Sampling: `{summary['sampling']}`",
        f"- Frames per video: `{summary['num_frames']}`",
        f"- Crop/input size: `{summary['crop_size']}`",
        f"- Val valid/error: `{summary['val_valid']}` / `{summary['val_errors']}`",
        f"- Test valid/error: `{summary['test_valid']}` / `{summary['test_errors']}`",
        "",
        "## Test results",
        "",
        "| Objective | Threshold | ACC | AUC | AP | F1 | Precision | Recall | Balanced ACC | FN | FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            "| {objective} | {threshold:.6g} | {acc:.4f} | {auc:.4f} | {ap:.4f} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {balanced_acc:.4f} | {fn} | {fp} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--waverep-root", default="comparison_experiment/WaveRep")
    parser.add_argument("--data-root", default="data/GenVideo-Val")
    parser.add_argument("--val-csv", default="data/GenVideo-Val/splits/ood_val.csv")
    parser.add_argument("--test-csv", default="data/GenVideo-Val/splits/ood_test.csv")
    parser.add_argument("--out-dir", default="comparison_experiment/results/WaveRep/ood_g4_uniform8")
    parser.add_argument("--weights", default="comparison_experiment/WaveRep/demo/weights/weights_dinov2_G4.ckpt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=504)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    waverep_root = (project_root / args.waverep_root).resolve()
    data_root = (project_root / args.data_root).resolve()
    val_csv = (project_root / args.val_csv).resolve()
    test_csv = (project_root / args.test_csv).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    weights = (project_root / args.weights).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not weights.exists():
        raise FileNotFoundError(f"WaveRep weights not found: {weights}")
    device = "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    transform = make_transform(args.crop_size)
    model = create_waverep_model(weights, args.crop_size, device)

    start = time.time()
    val_rows = score_split(
        model=model,
        rows=read_rows(val_csv),
        data_root=data_root,
        transform=transform,
        device=device,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        max_samples=args.max_val_samples,
        split_name="val",
    )
    test_rows = score_split(
        model=model,
        rows=read_rows(test_csv),
        data_root=data_root,
        transform=transform,
        device=device,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        max_samples=args.max_test_samples,
        split_name="test",
    )
    write_csv(out_dir / "val_scores.csv", val_rows)
    write_csv(out_dir / "test_scores.csv", test_rows)

    val_y, val_s = valid_arrays(val_rows)
    test_y, test_s = valid_arrays(test_rows)
    test_auc = float(roc_auc_score(test_y, test_s)) if len(np.unique(test_y)) == 2 else float("nan")
    test_ap = float(average_precision_score(test_y, test_s)) if len(np.unique(test_y)) == 2 else float("nan")

    result_rows: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        threshold, val_metrics = best_threshold(val_y, val_s, objective)
        test_metrics = metrics_at(test_y, test_s, threshold)
        result = {
            "objective": objective,
            "threshold": threshold,
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "acc": test_metrics["acc"],
            "auc": test_auc,
            "ap": test_ap,
            "f1": test_metrics["f1"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "balanced_acc": test_metrics["balanced_acc"],
            "tn": test_metrics["tn"],
            "fp": test_metrics["fp"],
            "fn": test_metrics["fn"],
            "tp": test_metrics["tp"],
        }
        result_rows.append(result)
        objective_dir = out_dir / objective
        write_csv(objective_dir / "predictions.csv", apply_predictions(test_rows, threshold))
        save_json(
            objective_dir / "metrics.json",
            {
                "acc": result["acc"],
                "auc": result["auc"],
                "ap": result["ap"],
                "f1": result["f1"],
                "precision": result["precision"],
                "recall": result["recall"],
                "balanced_acc": result["balanced_acc"],
                "confusion_matrix": [[result["tn"], result["fp"]], [result["fn"], result["tp"]]],
                "model": "WaveRep_DINOv2_G4",
                "objective": objective,
                "threshold": threshold,
            },
        )

    summary = {
        "model": "WaveRep_DINOv2_G4",
        "waverep_root": str(waverep_root),
        "weights": str(weights),
        "architecture": WAVEREP_ARC,
        "sampling": "uniform",
        "num_frames": args.num_frames,
        "crop_size": args.crop_size,
        "batch_size": args.batch_size,
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "val_valid": int(len(val_y)),
        "test_valid": int(len(test_y)),
        "val_errors": int(sum(1 for row in val_rows if row.get("error"))),
        "test_errors": int(sum(1 for row in test_rows if row.get("error"))),
        "seconds": time.time() - start,
        "result_rows": result_rows,
    }
    write_csv(out_dir / "results.csv", result_rows)
    save_json(out_dir / "results.json", summary)
    write_markdown(out_dir / "results.md", summary, result_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[write] {out_dir}")


if __name__ == "__main__":
    main()
