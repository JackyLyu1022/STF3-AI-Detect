from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
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
from tqdm import tqdm


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
OBJECTIVES = ("max_f1", "max_balanced_acc", "precision_0.95_recall_max", "max_acc")


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def uniform_indices(total: int, num_frames: int) -> np.ndarray:
    if total <= 0:
        return np.zeros(num_frames, dtype=np.int64)
    if total >= num_frames:
        return np.linspace(0, total - 1, num_frames).round().astype(np.int64)
    base = np.arange(total, dtype=np.int64)
    pad = np.full(num_frames - total, total - 1, dtype=np.int64)
    return np.concatenate([base, pad])


def crop_center_by_percentage(image: np.ndarray, percentage: float = 0.1) -> np.ndarray:
    height, width = image.shape[:2]
    if width > height:
        left = int(width * percentage)
        right = int(width * percentage)
        return image[:, left : width - right]
    up = int(height * percentage)
    down = int(height * percentage)
    return image[up : height - down, :]


def read_video_for_d3(
    video_path: Path,
    num_frames: int = 16,
    image_size: int = 224,
    rgb: bool = False,
) -> torch.Tensor:
    """Read a video into D3-style normalized frames [T,3,H,W].

    The official D3 dataset reads jpg frames through cv2 and normalizes with
    ImageNet mean/std. By default this keeps cv2's BGR channel order to mimic
    the official code path. Pass --rgb to convert BGR->RGB.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    for idx in uniform_indices(total, num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            if frames:
                frame = frames[-1].copy()
            else:
                frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = crop_center_by_percentage(frame, 0.1)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        x = frame.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = (x - MEAN) / STD
        frames.append(x)
    cap.release()
    arr = np.stack(frames, axis=0).astype(np.float32)
    return torch.from_numpy(arr)


def d3_model_import(d3_root: Path):
    sys.path.insert(0, str(d3_root.resolve()))
    from models import D3_model  # type: ignore

    return D3_model


@torch.no_grad()
def score_split(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    data_root: Path,
    device: torch.device,
    num_frames: int,
    image_size: int,
    rgb: bool,
    max_samples: int = 0,
) -> list[dict[str, object]]:
    if max_samples and max_samples > 0:
        rows = rows[:max_samples]
    scored: list[dict[str, object]] = []
    for row in tqdm(rows, desc="D3 scoring"):
        rel_path = row["rel_path"]
        video_path = data_root / rel_path
        item: dict[str, object] = {
            "rel_path": rel_path,
            "video_id": row.get("video_id", ""),
            "generator": row.get("generator", "unknown"),
            "label": int(row["label"]),
        }
        try:
            frames = read_video_for_d3(video_path, num_frames=num_frames, image_size=image_size, rgb=rgb)
            frames = frames.unsqueeze(0).to(device)
            _, _, dis_std = model(frames)
            item["d3_raw_score"] = float(dis_std.detach().cpu().flatten()[0].item())
            item["error"] = ""
        except Exception as exc:
            item["d3_raw_score"] = float("nan")
            item["error"] = f"{type(exc).__name__}: {exc}"
        scored.append(item)
    return scored


def valid_arrays(rows: list[dict[str, object]], score_key: str) -> tuple[np.ndarray, np.ndarray]:
    y, s = [], []
    for row in rows:
        score = row.get(score_key)
        try:
            score_f = float(score)  # type: ignore[arg-type]
        except Exception:
            continue
        if not math.isfinite(score_f):
            continue
        y.append(int(row["label"]))
        s.append(score_f)
    return np.asarray(y, dtype=np.int64), np.asarray(s, dtype=np.float64)


def metrics_at(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, object] = {
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
    return out


def objective_score(metrics: dict[str, object], objective: str) -> tuple[float, ...]:
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


def best_threshold(y: np.ndarray, score: np.ndarray, objective: str) -> tuple[float, dict[str, object]]:
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


def add_oriented_scores(rows: list[dict[str, object]], orientation: str) -> None:
    for row in rows:
        raw = float(row["d3_raw_score"])
        row["d3_fake_score"] = raw if orientation == "fake_high" else -raw


def choose_orientation(val_rows: list[dict[str, object]], requested: str) -> str:
    if requested in {"fake_high", "real_high"}:
        return requested
    y, raw = valid_arrays(val_rows, "d3_raw_score")
    if len(np.unique(y)) < 2:
        return "fake_high"
    auc_fake_high = roc_auc_score(y, raw)
    auc_real_high = roc_auc_score(y, -raw)
    return "fake_high" if auc_fake_high >= auc_real_high else "real_high"


def apply_predictions(rows: list[dict[str, object]], threshold: float) -> list[dict[str, object]]:
    out = []
    for row in rows:
        item = dict(row)
        score = float(item["d3_fake_score"])
        pred = 1 if score >= threshold else 0
        item["pred"] = pred
        item["correct"] = int(pred == int(item["label"]))
        out.append(item)
    return out


def write_markdown(path: Path, summary: dict[str, object], result_rows: list[dict[str, object]]) -> None:
    lines = [
        "# D3 OOD comparison on GenVideo-Val",
        "",
        "D3 is evaluated as a training-free external baseline. Score orientation and thresholds are selected on OOD validation split only, then fixed on OOD test split.",
        "",
        "## Configuration",
        "",
        f"- Encoder: `{summary['encoder']}`",
        f"- D3 loss: `{summary['loss']}`",
        f"- Frames: `{summary['num_frames']}`",
        f"- Image size: `{summary['image_size']}`",
        f"- Channel mode: `{summary['channel_mode']}`",
        f"- Orientation selected on val: `{summary['orientation']}`",
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
    parser.add_argument("--d3-root", default="comparison_experiment/D3")
    parser.add_argument("--data-root", default="data/GenVideo-Val")
    parser.add_argument("--val-csv", default="data/GenVideo-Val/splits/ood_val.csv")
    parser.add_argument("--test-csv", default="data/GenVideo-Val/splits/ood_test.csv")
    parser.add_argument("--out-dir", default="comparison_experiment/results/D3/ood_xclip16_l2")
    parser.add_argument("--encoder", default="XCLIP-16", choices=["CLIP-16", "CLIP-32", "XCLIP-16", "XCLIP-32", "DINO-base", "DINO-large", "ResNet-18", "VGG-16", "EfficientNet-b4", "MobileNet-v3"])
    parser.add_argument("--loss", default="l2", choices=["l2", "cos"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--rgb", action="store_true", help="Convert cv2 BGR frames to RGB. Default keeps BGR to mimic D3's official cv2.imread path.")
    parser.add_argument("--orientation", default="auto", choices=["auto", "fake_high", "real_high"])
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    d3_root = (project_root / args.d3_root).resolve()
    data_root = (project_root / args.data_root).resolve()
    val_csv = (project_root / args.val_csv).resolve()
    test_csv = (project_root / args.test_csv).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    D3_model = d3_model_import(d3_root)
    model = D3_model(encoder_type=args.encoder, loss_type=args.loss).to(device).eval()

    start = time.time()
    val_rows = score_split(model, read_rows(val_csv), data_root, device, args.num_frames, args.image_size, args.rgb, args.max_val_samples)
    test_rows = score_split(model, read_rows(test_csv), data_root, device, args.num_frames, args.image_size, args.rgb, args.max_test_samples)

    orientation = choose_orientation(val_rows, args.orientation)
    add_oriented_scores(val_rows, orientation)
    add_oriented_scores(test_rows, orientation)

    val_y, val_s = valid_arrays(val_rows, "d3_fake_score")
    test_y, test_s = valid_arrays(test_rows, "d3_fake_score")
    test_auc = float(roc_auc_score(test_y, test_s)) if len(np.unique(test_y)) == 2 else float("nan")
    test_ap = float(average_precision_score(test_y, test_s)) if len(np.unique(test_y)) == 2 else float("nan")

    result_rows: list[dict[str, object]] = []
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
        pred_rows = apply_predictions(test_rows, threshold)
        objective_dir = out_dir / objective
        write_csv(objective_dir / "predictions.csv", pred_rows)
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
                "model": f"D3_{args.encoder}_{args.loss}",
                "objective": objective,
                "threshold": threshold,
                "orientation": orientation,
            },
        )

    summary = {
        "encoder": args.encoder,
        "loss": args.loss,
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "channel_mode": "RGB" if args.rgb else "BGR_official_cv2",
        "orientation": orientation,
        "val_valid": int(len(val_y)),
        "test_valid": int(len(test_y)),
        "val_errors": int(sum(1 for row in val_rows if row.get("error"))),
        "test_errors": int(sum(1 for row in test_rows if row.get("error"))),
        "seconds": time.time() - start,
        "result_rows": result_rows,
    }
    write_csv(out_dir / "val_scores.csv", val_rows)
    write_csv(out_dir / "test_scores.csv", test_rows)
    write_csv(out_dir / "results.csv", result_rows)
    save_json(out_dir / "results.json", summary)
    write_markdown(out_dir / "results.md", summary, result_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[write] {out_dir}")


if __name__ == "__main__":
    main()
