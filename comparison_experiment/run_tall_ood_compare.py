from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


OBJECTIVES = ("max_f1", "max_balanced_acc", "precision_0.95_recall_max", "max_acc")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


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


def uniform_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError(f"invalid total_frames={total_frames}")
    return np.linspace(0, total_frames - 1, num_frames).round().astype(int).tolist()


def load_video_tensor(video_path: Path, num_frames: int, frame_size: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        arr = torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0)
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        frames.append(arr)
    cap.release()
    # TALL expects B x (T*C) x H x W.
    return torch.cat(frames, dim=0)


class TallCsvDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        data_root: Path,
        *,
        num_frames: int,
        frame_size: int,
        max_samples: int = 0,
    ) -> None:
        self.rows = rows[:max_samples] if max_samples and max_samples > 0 else rows
        self.data_root = data_root
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        rel_path = row["rel_path"]
        video_path = self.data_root / rel_path
        label = int(row["label"])
        try:
            x = load_video_tensor(video_path, self.num_frames, self.frame_size)
            valid = True
            error = ""
        except Exception as exc:
            # Keep metadata so the run can finish and write an error CSV instead
            # of dying on a small number of bad/corrupt videos.
            x = torch.empty(0)
            valid = False
            error = f"{type(exc).__name__}: {exc}"
        return {
            "x": x,
            "label": torch.tensor(label, dtype=torch.long),
            "rel_path": rel_path,
            "video_id": row.get("video_id", ""),
            "filename": row.get("filename", video_path.name),
            "generator": row.get("generator", "unknown"),
            "valid": valid,
            "error": error,
        }


def collate_keep_meta(batch: list[dict[str, Any]]) -> dict[str, Any]:
    valid_batch = [b for b in batch if b.get("valid", True)]
    error_batch = [b for b in batch if not b.get("valid", True)]
    if not valid_batch:
        return {
            "x": torch.empty(0),
            "label": torch.empty(0, dtype=torch.long),
            "rel_path": [],
            "video_id": [],
            "filename": [],
            "generator": [],
            "errors": error_batch,
        }
    return {
        "x": torch.stack([b["x"] for b in valid_batch], dim=0),
        "label": torch.stack([b["label"] for b in valid_batch], dim=0),
        "rel_path": [b["rel_path"] for b in valid_batch],
        "video_id": [b["video_id"] for b in valid_batch],
        "filename": [b["filename"] for b in valid_batch],
        "generator": [b["generator"] for b in valid_batch],
        "errors": error_batch,
    }


def import_tall(tall_root: Path):
    sys.path.insert(0, str(tall_root.resolve()))
    import my_models  # noqa: F401
    from timm.models import create_model

    return create_model


def build_model(
    *,
    create_model,
    num_frames: int,
    num_classes: int,
    thumbnail_rows: int,
    pretrained: bool,
    device: str,
) -> nn.Module:
    model = create_model(
        "TALL_SWIN",
        pretrained=pretrained,
        duration=num_frames,
        num_classes=num_classes,
        thumbnail_rows=thumbnail_rows,
        hpe_to_token=False,
        token_mask=True,
        use_checkpoint=False,
    )
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    *,
    amp: bool,
    epoch: int,
) -> dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.startswith("cuda"))
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    pbar = tqdm(loader, desc=f"TALL train e{epoch}", unit="batch")
    for batch in pbar:
        if batch["x"].numel() == 0:
            continue
        x = batch["x"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.startswith("cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = int(y.numel())
        total_loss += float(loss.detach().cpu()) * bs
        total_correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
        total_n += bs
        pbar.set_postfix(loss=total_loss / max(1, total_n), acc=total_correct / max(1, total_n))
    return {"loss": total_loss / max(1, total_n), "acc": total_correct / max(1, total_n)}


@torch.no_grad()
def infer_rows(model: nn.Module, loader: DataLoader, device: str, *, amp: bool, split_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc=f"TALL infer {split_name}", unit="batch"):
        for bad in batch.get("errors", []):
            errors.append(
                {
                    "rel_path": bad.get("rel_path", ""),
                    "video_id": bad.get("video_id", ""),
                    "filename": bad.get("filename", ""),
                    "generator": bad.get("generator", ""),
                    "label": int(bad["label"]) if "label" in bad else "",
                    "error": bad.get("error", "invalid sample"),
                }
            )
        if batch["x"].numel() == 0:
            continue
        try:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["label"].numpy().astype(int)
            with torch.amp.autocast("cuda", enabled=amp and device.startswith("cuda")):
                logits = model(x)
                prob = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            for i in range(len(y)):
                rows.append(
                    {
                        "rel_path": batch["rel_path"][i],
                        "video_id": batch["video_id"][i],
                        "filename": batch["filename"][i],
                        "generator": batch["generator"][i],
                        "label": int(y[i]),
                        "tall_fake_score": float(prob[i]),
                        "default_pred": int(pred[i]),
                        "error": "",
                    }
                )
        except Exception as exc:
            errors.append({"error": f"{type(exc).__name__}: {exc}"})
    return rows, errors


def valid_arrays(rows: list[dict[str, Any]], score_key: str = "tall_fake_score") -> tuple[np.ndarray, np.ndarray]:
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
        pred = 1 if float(item["tall_fake_score"]) >= threshold else 0
        item["pred"] = pred
        item["correct"] = int(pred == int(item["label"]))
        out.append(item)
    return out


def write_markdown(path: Path, summary: dict[str, Any], result_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# TALL OOD comparison on GenVideo-Val",
        "",
        "TALL is evaluated as an external thumbnail-layout spatiotemporal baseline. "
        "Unlike STALL, this run trains/fine-tunes TALL on the same OOD train split used by STF3, "
        "selects thresholds on OOD validation, and reports fixed-threshold OOD test metrics.",
        "",
        "## Configuration",
        "",
        f"- TALL root: `{summary['tall_root']}`",
        f"- Train CSV: `{summary['train_csv']}`",
        f"- Val CSV: `{summary['val_csv']}`",
        f"- Test CSV: `{summary['test_csv']}`",
        f"- Pretrained ImageNet/Swin init: `{summary['pretrained']}`",
        f"- Epochs: `{summary['epochs']}`",
        f"- Input frames: `{summary['num_frames']}` uniform frames",
        f"- Per-frame resize before thumbnail: `{summary['frame_size']}`",
        f"- Thumbnail rows: `{summary['thumbnail_rows']}`",
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
    parser.add_argument("--tall-root", default="comparison_experiment/TALL")
    parser.add_argument("--data-root", default="data/GenVideo-Val")
    parser.add_argument("--train-csv", default="data/GenVideo-Val/splits/ood_train.csv")
    parser.add_argument("--val-csv", default="data/GenVideo-Val/splits/ood_val.csv")
    parser.add_argument("--test-csv", default="data/GenVideo-Val/splits/ood_test.csv")
    parser.add_argument("--out-dir", default="comparison_experiment/results/TALL/ood_tall_uniform8_e5")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--pretrained", action="store_true", help="Use TALL's ImageNet/Swin pretrained initialization.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--frame-size", type=int, default=112)
    parser.add_argument("--thumbnail-rows", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    tall_root = (project_root / args.tall_root).resolve()
    data_root = (project_root / args.data_root).resolve()
    train_csv = (project_root / args.train_csv).resolve()
    val_csv = (project_root / args.val_csv).resolve()
    test_csv = (project_root / args.test_csv).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    create_model = import_tall(tall_root)
    model = build_model(
        create_model=create_model,
        num_frames=args.num_frames,
        num_classes=2,
        thumbnail_rows=args.thumbnail_rows,
        pretrained=args.pretrained and not args.eval_only,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_ds = TallCsvDataset(read_rows(train_csv), data_root, num_frames=args.num_frames, frame_size=args.frame_size, max_samples=args.max_train_samples)
    val_ds = TallCsvDataset(read_rows(val_csv), data_root, num_frames=args.num_frames, frame_size=args.frame_size, max_samples=args.max_val_samples)
    test_ds = TallCsvDataset(read_rows(test_csv), data_root, num_frames=args.num_frames, frame_size=args.frame_size, max_samples=args.max_test_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device == "cuda", collate_fn=collate_keep_meta)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device == "cuda", collate_fn=collate_keep_meta)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device == "cuda", collate_fn=collate_keep_meta)

    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_ckpt = out_dir / "best.pt"
    start = time.time()

    if args.checkpoint:
        ckpt = torch.load((project_root / args.checkpoint).resolve(), map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
        print(f"[load] checkpoint={args.checkpoint}")

    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            train_stats = train_one_epoch(model, train_loader, optimizer, device, amp=args.amp, epoch=epoch)
            val_rows_epoch, val_errors_epoch = infer_rows(model, val_loader, device, amp=args.amp, split_name=f"val_e{epoch}")
            val_y_epoch, val_s_epoch = valid_arrays(val_rows_epoch)
            val_metrics_epoch = metrics_at(val_y_epoch, val_s_epoch, 0.5)
            entry = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_metrics_epoch.items()}, "val_errors": len(val_errors_epoch)}
            history.append(entry)
            save_json(out_dir / "history.json", history)
            epoch_ckpt = {"model": model.state_dict(), "epoch": epoch, "args": vars(args), "history": history}
            torch.save(epoch_ckpt, out_dir / f"epoch_{epoch}.pt")
            torch.save(epoch_ckpt, out_dir / "last.pt")
            if val_metrics_epoch["f1"] > best_val_f1:
                best_val_f1 = float(val_metrics_epoch["f1"])
                torch.save(epoch_ckpt, best_ckpt)
                print(f"[best] epoch={epoch} val_f1@0.5={best_val_f1:.4f}")

        ckpt = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)

    val_rows, val_errors = infer_rows(model, val_loader, device, amp=args.amp, split_name="val")
    test_rows, test_errors = infer_rows(model, test_loader, device, amp=args.amp, split_name="test")
    write_csv(out_dir / "val_scores.csv", val_rows)
    write_csv(out_dir / "test_scores.csv", test_rows)
    write_csv(out_dir / "val_errors.csv", val_errors)
    write_csv(out_dir / "test_errors.csv", test_errors)

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
                "model": "TALL_SWIN",
                "objective": objective,
                "threshold": threshold,
            },
        )

    summary = {
        "model": "TALL_SWIN",
        "tall_root": str(tall_root),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "out_dir": str(out_dir),
        "pretrained": bool(args.pretrained),
        "eval_only": bool(args.eval_only),
        "checkpoint": args.checkpoint,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_frames": args.num_frames,
        "frame_size": args.frame_size,
        "thumbnail_rows": args.thumbnail_rows,
        "device": device,
        "amp": bool(args.amp),
        "train_samples": len(train_ds),
        "val_valid": int(len(val_y)),
        "test_valid": int(len(test_y)),
        "val_errors": int(len(val_errors)),
        "test_errors": int(len(test_errors)),
        "seconds": time.time() - start,
        "history": history,
        "result_rows": result_rows,
    }
    write_csv(out_dir / "results.csv", result_rows)
    save_json(out_dir / "results.json", summary)
    write_markdown(out_dir / "results.md", summary, result_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[write] {out_dir}")


if __name__ == "__main__":
    main()
