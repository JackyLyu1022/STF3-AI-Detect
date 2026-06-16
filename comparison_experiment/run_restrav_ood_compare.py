from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


OBJECTIVES = ("max_f1", "max_balanced_acc", "precision_0.95_recall_max", "max_acc")


class MLP(nn.Module):
    """Same lightweight classifier shape as the official ReStraV train.py."""

    def __init__(self, in_dim: int = 21, h1: int = 64, h2: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_restrav_feature_module(restrav_root: Path):
    sys.path.insert(0, str(restrav_root.resolve()))
    import dinov2_features as d2  # type: ignore

    return d2


def safe_string_array(values: list[str]) -> np.ndarray:
    return np.asarray(values, dtype=object)


def save_feature_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_feature_cache(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


@torch.no_grad()
def extract_split_features(
    *,
    d2: Any,
    rows: list[dict[str, str]],
    data_root: Path,
    device: torch.device,
    cache_path: Path,
    split_name: str,
    num_frames: int,
    window_sec: float,
    max_samples: int,
    force: bool,
) -> dict[str, Any]:
    """Extract ReStraV 21-D temporal-geometry features and cache them.

    The official ReStraV code decodes with TorchCodec, samples a short clip,
    embeds frames with DINOv2 ViT-S/14, and converts the trajectory to 21-D
    distances/angles/statistics. We reuse that implementation directly and only
    add caching/error handling for our OOD train/val/test protocol.
    """

    if cache_path.exists() and not force:
        print(f"[cache] load {split_name}: {cache_path}")
        return load_feature_cache(cache_path)

    if max_samples and max_samples > 0:
        rows = rows[:max_samples]

    features: list[np.ndarray] = []
    labels: list[int] = []
    rel_paths: list[str] = []
    video_ids: list[str] = []
    generators: list[str] = []
    errors: list[dict[str, Any]] = []

    for row in tqdm(rows, desc=f"ReStraV features [{split_name}]"):
        rel_path = row["rel_path"]
        video_path = data_root / rel_path
        try:
            z = d2.extract_dinov2_embeddings(
                [str(video_path)],
                device=device,
                T=num_frames,
                window_sec=window_sec,
            )
            feat = d2.features_from_Z(z)
            if isinstance(feat, torch.Tensor):
                feat_np = feat.detach().cpu().numpy()
            else:
                feat_np = np.asarray(feat)
            feat_np = feat_np.reshape(1, -1).astype(np.float32)
            if feat_np.shape[1] != 21:
                raise RuntimeError(f"Unexpected ReStraV feature dim: {feat_np.shape}")
            features.append(feat_np[0])
            labels.append(int(row["label"]))  # project convention: 1=fake, 0=real
            rel_paths.append(rel_path)
            video_ids.append(row.get("video_id", ""))
            generators.append(row.get("generator", "unknown"))
        except Exception as exc:
            errors.append(
                {
                    "rel_path": rel_path,
                    "video_id": row.get("video_id", ""),
                    "generator": row.get("generator", "unknown"),
                    "label": int(row["label"]),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if features:
        x = np.stack(features, axis=0).astype(np.float32)
    else:
        x = np.empty((0, 21), dtype=np.float32)

    payload = {
        "features": x,
        "labels": np.asarray(labels, dtype=np.int64),
        "rel_paths": safe_string_array(rel_paths),
        "video_ids": safe_string_array(video_ids),
        "generators": safe_string_array(generators),
        "errors_json": np.asarray(json.dumps(errors, ensure_ascii=False), dtype=object),
        "num_frames": np.asarray(num_frames, dtype=np.int64),
        "window_sec": np.asarray(window_sec, dtype=np.float32),
    }
    save_feature_cache(cache_path, payload)
    print(f"[cache] write {split_name}: {cache_path} valid={len(labels)} errors={len(errors)}")
    if errors:
        write_csv(cache_path.with_suffix(".errors.csv"), errors)
    return payload


def parse_errors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("errors_json", "[]")
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    try:
        return json.loads(str(raw))
    except Exception:
        return []


def standardize_features(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True).astype(np.float32)
    std = (train_x.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
    return (
        ((train_x - mean) / std).astype(np.float32),
        ((val_x - mean) / std).astype(np.float32),
        ((test_x - mean) / std).astype(np.float32),
        mean,
        std,
    )


def normalize_with_stats(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean = mean.reshape(1, -1).astype(np.float32)
    std = std.reshape(1, -1).astype(np.float32)
    return ((x - mean) / (std + 1e-8)).astype(np.float32)


def load_pretrained_restrav(pretrained_dir: Path, device: torch.device) -> tuple[MLP, np.ndarray, np.ndarray, float | None]:
    """Load official-style ReStraV artifacts: model.pt, mean.npy, std.npy, optional best_tau.npy."""

    required = [pretrained_dir / "model.pt", pretrained_dir / "mean.npy", pretrained_dir / "std.npy"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing ReStraV pretrained artifact(s):\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\nPut the official ReStraV files `model.pt`, `mean.npy`, `std.npy` "
            "and optionally `best_tau.npy` into --pretrained-dir, then rerun."
        )

    model = MLP().to(device)
    state = torch.load(pretrained_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    mean = np.load(pretrained_dir / "mean.npy").astype(np.float32)
    std = np.load(pretrained_dir / "std.npy").astype(np.float32)
    best_tau_path = pretrained_dir / "best_tau.npy"
    best_tau = float(np.load(best_tau_path)) if best_tau_path.exists() else None
    return model, mean, std, best_tau


def train_restrav_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    pos_weight_mode: str,
) -> MLP:
    ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")

    model = MLP().to(device)
    if pos_weight_mode == "auto":
        n_pos = max(1, int((train_y == 1).sum()))
        n_neg = max(1, int((train_y == 0).sum()))
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
        criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[train] BCEWithLogitsLoss pos_weight={float(pos_weight.item()):.4f}")
    elif pos_weight_mode == "none":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(pos_weight_mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(loader, desc=f"ReStraV MLP epoch {epoch}/{epochs}"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * xb.size(0)
        print(f"[train] epoch={epoch} loss={total_loss / max(1, len(ds)):.6f}")
    return model


@torch.no_grad()
def predict_scores(model: MLP, x: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    ds = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    logits_all: list[np.ndarray] = []
    model.eval()
    for (xb,) in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).detach().cpu().numpy().reshape(-1)
        logits_all.append(logits)
    logits_np = np.concatenate(logits_all, axis=0) if logits_all else np.empty((0,), dtype=np.float32)
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))
    return logits_np.astype(np.float64), probs_np.astype(np.float64)


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


def rows_from_scores(payload: dict[str, Any], logits: np.ndarray, probs: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = payload["labels"].astype(int)
    rel_paths = payload["rel_paths"].astype(str)
    video_ids = payload["video_ids"].astype(str)
    generators = payload["generators"].astype(str)
    for i in range(len(labels)):
        rows.append(
            {
                "rel_path": rel_paths[i],
                "video_id": video_ids[i],
                "generator": generators[i],
                "label": int(labels[i]),
                "restrav_logit": float(logits[i]),
                "restrav_fake_prob": float(probs[i]),
                "error": "",
            }
        )
    return rows


def apply_predictions(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        score = float(item["restrav_fake_prob"])
        pred = 1 if score >= threshold else 0
        item["pred"] = pred
        item["correct"] = int(pred == int(item["label"]))
        out.append(item)
    return out


def write_markdown(path: Path, summary: dict[str, Any], result_rows: list[dict[str, Any]]) -> None:
    if summary.get("eval_only"):
        protocol = (
            "ReStraV is evaluated in eval-only mode with pretrained MLP artifacts. "
            "No ReStraV classifier is trained on the STF3 OOD train split. "
            "Features are extracted with the official DINOv2 feature code; thresholds are selected on the OOD validation split unless the row is marked as official_best_tau."
        )
        train_line = f"- Train CSV: `{summary['train_csv']}` (not used in eval-only mode)"
        epoch_line = f"- MLP epochs: `0` (pretrained eval-only)"
    else:
        protocol = (
            "ReStraV is evaluated as an external trainable baseline. "
            "The 21-D ReStraV temporal-geometry features are extracted with the official DINOv2 feature code. "
            "The MLP is trained on the OOD train split, thresholds are selected on OOD validation split only, and the fixed thresholds are evaluated on OOD test split."
        )
        train_line = f"- Train CSV: `{summary['train_csv']}`"
        epoch_line = f"- MLP epochs: `{summary['epochs']}`"
    lines = [
        "# ReStraV OOD comparison on GenVideo-Val",
        "",
        protocol,
        "",
        "## Configuration",
        "",
        f"- ReStraV root: `{summary['restrav_root']}`",
        train_line,
        f"- Val CSV: `{summary['val_csv']}`",
        f"- Test CSV: `{summary['test_csv']}`",
        f"- DINOv2 backbone: `dinov2_vits14`",
        f"- ReStraV frames: `{summary['num_frames']}`",
        f"- Window seconds: `{summary['window_sec']}`",
        epoch_line,
        f"- Batch size: `{summary['batch_size']}`",
        f"- Pos weight mode: `{summary['pos_weight_mode']}`",
        f"- Pretrained dir: `{summary.get('pretrained_dir', '')}`",
        f"- Pretrained positive class: `{summary.get('pretrained_positive_class', '')}`",
        f"- Train valid/error: `{summary['train_valid']}` / `{summary['train_errors']}`",
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
    parser.add_argument("--restrav-root", default="comparison_experiment/ReStraV")
    parser.add_argument("--data-root", default="data/GenVideo-Val")
    parser.add_argument("--train-csv", default="data/GenVideo-Val/splits/ood_train.csv")
    parser.add_argument("--val-csv", default="data/GenVideo-Val/splits/ood_val.csv")
    parser.add_argument("--test-csv", default="data/GenVideo-Val/splits/ood_test.csv")
    parser.add_argument("--out-dir", default="comparison_experiment/results/ReStraV/ood_dinov2_t24_w2")
    parser.add_argument("--feature-cache-dir", default="", help="Optional shared feature cache dir. Useful for eval-only reruns with the same T/window.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=24, help="Official ReStraV default is 24 frames.")
    parser.add_argument("--window-sec", type=float, default=2.0, help="Official ReStraV default is a 2-second clip.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight-mode", default="auto", choices=["auto", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--force-features", action="store_true", help="Recompute cached ReStraV features.")
    parser.add_argument("--eval-only", action="store_true", help="Do not train MLP; load pretrained ReStraV artifacts and evaluate val/test.")
    parser.add_argument("--pretrained-dir", default="comparison_experiment/ReStraV/pretrained", help="Directory containing model.pt, mean.npy, std.npy, optional best_tau.npy.")
    parser.add_argument("--pretrained-positive-class", default="real", choices=["real", "fake"], help="Official ReStraV train.py uses prob_1=REAL. Use real for official weights; use fake for weights trained with project labels.")
    args = parser.parse_args()

    set_seed(args.seed)
    project_root = Path(args.project_root).resolve()
    restrav_root = (project_root / args.restrav_root).resolve()
    data_root = (project_root / args.data_root).resolve()
    train_csv = (project_root / args.train_csv).resolve()
    val_csv = (project_root / args.val_csv).resolve()
    test_csv = (project_root / args.test_csv).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = (project_root / args.feature_cache_dir).resolve() if args.feature_cache_dir else out_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    pretrained_dir = (project_root / args.pretrained_dir).resolve()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")

    start = time.time()
    d2 = load_restrav_feature_module(restrav_root)

    if args.eval_only:
        model, mean, std, official_best_tau = load_pretrained_restrav(pretrained_dir, device)
        val_payload = extract_split_features(
            d2=d2,
            rows=read_rows(val_csv),
            data_root=data_root,
            device=device,
            cache_path=feature_dir / "val_features.npz",
            split_name="val",
            num_frames=args.num_frames,
            window_sec=args.window_sec,
            max_samples=args.max_val_samples,
            force=args.force_features,
        )
        test_payload = extract_split_features(
            d2=d2,
            rows=read_rows(test_csv),
            data_root=data_root,
            device=device,
            cache_path=feature_dir / "test_features.npz",
            split_name="test",
            num_frames=args.num_frames,
            window_sec=args.window_sec,
            max_samples=args.max_test_samples,
            force=args.force_features,
        )
        val_x = val_payload["features"].astype(np.float32)
        test_x = test_payload["features"].astype(np.float32)
        val_y = val_payload["labels"].astype(np.int64)
        test_y = test_payload["labels"].astype(np.int64)
        if len(val_x) == 0 or len(test_x) == 0:
            raise RuntimeError("Empty val/test features. Check decoding errors and input CSV paths.")

        val_xs = normalize_with_stats(val_x, mean, std)
        test_xs = normalize_with_stats(test_x, mean, std)
        val_logits, val_prob_model_positive = predict_scores(model, val_xs, device, args.batch_size)
        test_logits, test_prob_model_positive = predict_scores(model, test_xs, device, args.batch_size)
        if args.pretrained_positive_class == "real":
            # Official ReStraV labels are 1=REAL, 0=FAKE. Our project positive class is FAKE=1.
            val_probs = 1.0 - val_prob_model_positive
            test_probs = 1.0 - test_prob_model_positive
        else:
            val_probs = val_prob_model_positive
            test_probs = test_prob_model_positive

        val_score_rows = rows_from_scores(val_payload, val_logits, val_probs)
        test_score_rows = rows_from_scores(test_payload, test_logits, test_probs)
        write_csv(out_dir / "val_scores.csv", val_score_rows)
        write_csv(out_dir / "test_scores.csv", test_score_rows)

        test_auc = float(roc_auc_score(test_y, test_probs)) if len(np.unique(test_y)) == 2 else float("nan")
        test_ap = float(average_precision_score(test_y, test_probs)) if len(np.unique(test_y)) == 2 else float("nan")

        result_rows: list[dict[str, Any]] = []
        for objective in OBJECTIVES:
            threshold, val_metrics = best_threshold(val_y, val_probs, objective)
            test_metrics = metrics_at(test_y, test_probs, threshold)
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
            pred_rows = apply_predictions(test_score_rows, threshold)
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
                    "model": "ReStraV_DINOv2_21D_MLP_pretrained_eval_only",
                    "objective": objective,
                    "threshold": threshold,
                    "pretrained_positive_class": args.pretrained_positive_class,
                },
            )

        if official_best_tau is not None:
            if args.pretrained_positive_class == "real":
                official_fake_threshold = 1.0 - official_best_tau
            else:
                official_fake_threshold = official_best_tau
            val_metrics = metrics_at(val_y, val_probs, official_fake_threshold)
            test_metrics = metrics_at(test_y, test_probs, official_fake_threshold)
            result = {
                "objective": "official_best_tau",
                "threshold": official_fake_threshold,
                "official_raw_tau": official_best_tau,
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
            pred_rows = apply_predictions(test_score_rows, official_fake_threshold)
            objective_dir = out_dir / "official_best_tau"
            write_csv(objective_dir / "predictions.csv", pred_rows)
            save_json(objective_dir / "metrics.json", result)

        summary = {
            "model": "ReStraV_DINOv2_21D_MLP_pretrained_eval_only",
            "eval_only": True,
            "pretrained_dir": str(pretrained_dir),
            "pretrained_positive_class": args.pretrained_positive_class,
            "official_best_tau": official_best_tau,
            "restrav_root": str(restrav_root),
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "test_csv": str(test_csv),
            "num_frames": args.num_frames,
            "window_sec": args.window_sec,
            "epochs": 0,
            "batch_size": args.batch_size,
            "lr": None,
            "pos_weight_mode": "not_used_eval_only",
            "seed": args.seed,
            "train_valid": 0,
            "val_valid": int(len(val_y)),
            "test_valid": int(len(test_y)),
            "train_errors": 0,
            "val_errors": int(len(parse_errors(val_payload))),
            "test_errors": int(len(parse_errors(test_payload))),
            "seconds": time.time() - start,
            "result_rows": result_rows,
        }
        write_csv(out_dir / "results.csv", result_rows)
        save_json(out_dir / "results.json", summary)
        write_markdown(out_dir / "results.md", summary, result_rows)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[write] {out_dir}")
        return

    train_payload = extract_split_features(
        d2=d2,
        rows=read_rows(train_csv),
        data_root=data_root,
        device=device,
        cache_path=feature_dir / "train_features.npz",
        split_name="train",
        num_frames=args.num_frames,
        window_sec=args.window_sec,
        max_samples=args.max_train_samples,
        force=args.force_features,
    )
    val_payload = extract_split_features(
        d2=d2,
        rows=read_rows(val_csv),
        data_root=data_root,
        device=device,
        cache_path=feature_dir / "val_features.npz",
        split_name="val",
        num_frames=args.num_frames,
        window_sec=args.window_sec,
        max_samples=args.max_val_samples,
        force=args.force_features,
    )
    test_payload = extract_split_features(
        d2=d2,
        rows=read_rows(test_csv),
        data_root=data_root,
        device=device,
        cache_path=feature_dir / "test_features.npz",
        split_name="test",
        num_frames=args.num_frames,
        window_sec=args.window_sec,
        max_samples=args.max_test_samples,
        force=args.force_features,
    )

    train_x = train_payload["features"].astype(np.float32)
    val_x = val_payload["features"].astype(np.float32)
    test_x = test_payload["features"].astype(np.float32)
    train_y = train_payload["labels"].astype(np.int64)
    val_y = val_payload["labels"].astype(np.int64)
    test_y = test_payload["labels"].astype(np.int64)
    if len(train_x) == 0 or len(val_x) == 0 or len(test_x) == 0:
        raise RuntimeError("Empty train/val/test features. Check decoding errors and input CSV paths.")

    train_xs, val_xs, test_xs, mean, std = standardize_features(train_x, val_x, test_x)
    np.save(out_dir / "mean.npy", mean)
    np.save(out_dir / "std.npy", std)

    model = train_restrav_mlp(
        train_xs,
        train_y,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pos_weight_mode=args.pos_weight_mode,
    )
    torch.save(model.state_dict(), out_dir / "model.pt")

    train_logits, train_probs = predict_scores(model, train_xs, device, args.batch_size)
    val_logits, val_probs = predict_scores(model, val_xs, device, args.batch_size)
    test_logits, test_probs = predict_scores(model, test_xs, device, args.batch_size)

    train_score_rows = rows_from_scores(train_payload, train_logits, train_probs)
    val_score_rows = rows_from_scores(val_payload, val_logits, val_probs)
    test_score_rows = rows_from_scores(test_payload, test_logits, test_probs)
    write_csv(out_dir / "train_scores.csv", train_score_rows)
    write_csv(out_dir / "val_scores.csv", val_score_rows)
    write_csv(out_dir / "test_scores.csv", test_score_rows)

    test_auc = float(roc_auc_score(test_y, test_probs)) if len(np.unique(test_y)) == 2 else float("nan")
    test_ap = float(average_precision_score(test_y, test_probs)) if len(np.unique(test_y)) == 2 else float("nan")

    result_rows: list[dict[str, Any]] = []
    for objective in OBJECTIVES:
        threshold, val_metrics = best_threshold(val_y, val_probs, objective)
        test_metrics = metrics_at(test_y, test_probs, threshold)
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
        pred_rows = apply_predictions(test_score_rows, threshold)
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
                "model": "ReStraV_DINOv2_21D_MLP",
                "objective": objective,
                "threshold": threshold,
            },
        )

    summary = {
        "model": "ReStraV_DINOv2_21D_MLP",
        "restrav_root": str(restrav_root),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "num_frames": args.num_frames,
        "window_sec": args.window_sec,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "pos_weight_mode": args.pos_weight_mode,
        "seed": args.seed,
        "train_valid": int(len(train_y)),
        "val_valid": int(len(val_y)),
        "test_valid": int(len(test_y)),
        "train_errors": int(len(parse_errors(train_payload))),
        "val_errors": int(len(parse_errors(val_payload))),
        "test_errors": int(len(parse_errors(test_payload))),
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
