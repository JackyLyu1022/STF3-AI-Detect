from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
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


OBJECTIVES = ("max_f1", "max_balanced_acc", "precision_0.95_recall_max", "max_acc")


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


def import_stall(stall_root: Path):
    src = stall_root / "src"
    sys.path.insert(0, str(src.resolve()))
    from dataset_utils import count_cache_misses, load_csv_with_emb_cache, prefill_emb_cache  # type: ignore
    from stall import STALL  # type: ignore
    from video_index import compute_windows, downsample_frames, get_video_metadata  # type: ignore

    return STALL, count_cache_misses, load_csv_with_emb_cache, prefill_emb_cache, compute_windows, downsample_frames, get_video_metadata


def _probe_one(
    row: dict[str, str],
    *,
    data_root: Path,
    target_fps: float,
    compute_windows,
    downsample_frames,
    get_video_metadata,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    rel_path = row["rel_path"]
    video_path = data_root / rel_path
    item_base = {
        "video_id": row.get("video_id", ""),
        "rel_path": rel_path,
        "video_path": str(video_path),
        "subset": "real" if int(row["label"]) == 0 else "annotated",
        "source_model": row.get("generator", "unknown"),
        "label": int(row["label"]),
        "label_name": row.get("label_name", ""),
        "filename": row.get("filename", video_path.name),
    }
    try:
        meta = get_video_metadata(str(video_path))
        fps = meta["fps"]
        duration_seconds = meta["duration_seconds"]
        num_frames = meta["num_frames"]
        if fps is None or duration_seconds is None or num_frames is None:
            raise RuntimeError(f"ffprobe failed: {meta}")
        if float(fps) < target_fps:
            raise RuntimeError(f"fps<{target_fps}: fps={fps}")
        if float(duration_seconds) < 1.0:
            raise RuntimeError(f"duration<1s: duration={duration_seconds}")
        idxs = downsample_frames(int(num_frames), float(fps), target_fps)
        windows = compute_windows(idxs, target_fps=target_fps)
        out = dict(item_base)
        out.update(
            {
                "fps": float(fps),
                "duration_seconds": float(duration_seconds),
                "num_frames": int(num_frames),
                "downsample_idxs": json.dumps(idxs),
                "1_sec_idxs": json.dumps(windows["1_sec_idxs"]) if windows["1_sec_idxs"] is not None else None,
                "2_sec_idxs": json.dumps(windows["2_sec_idxs"]) if windows["2_sec_idxs"] is not None else None,
                "3_sec_idxs": json.dumps(windows["3_sec_idxs"]) if windows["3_sec_idxs"] is not None else None,
                "4_sec_idxs": json.dumps(windows["4_sec_idxs"]) if windows["4_sec_idxs"] is not None else None,
            }
        )
        return out, None
    except Exception as exc:
        err = dict(item_base)
        err["error"] = f"{type(exc).__name__}: {exc}"
        return None, err


def build_stall_index(
    *,
    split_rows: list[dict[str, str]],
    data_root: Path,
    out_csv: Path,
    error_csv: Path,
    target_fps: float,
    workers: int,
    max_samples: int,
    force: bool,
    compute_windows,
    downsample_frames,
    get_video_metadata,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if out_csv.exists() and not force:
        df = pd.read_csv(out_csv)
        errors: list[dict[str, Any]] = []
        if error_csv.exists():
            errors = list(pd.read_csv(error_csv).to_dict("records"))
        print(f"[index] load {out_csv} rows={len(df)} errors={len(errors)}")
        return df, errors

    rows = split_rows[:max_samples] if max_samples and max_samples > 0 else split_rows
    valid: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [
            ex.submit(
                _probe_one,
                row,
                data_root=data_root,
                target_fps=target_fps,
                compute_windows=compute_windows,
                downsample_frames=downsample_frames,
                get_video_metadata=get_video_metadata,
            )
            for row in rows
        ]
        for fut in tqdm(futures, desc=f"STALL index {out_csv.stem}"):
            item, err = fut.result()
            if item is not None:
                valid.append(item)
            if err is not None:
                errors.append(err)
    df = pd.DataFrame(valid)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    write_csv(error_csv, errors)
    print(f"[index] write {out_csv} rows={len(df)} errors={len(errors)}")
    return df, errors


def score_index(
    *,
    model,
    csv_path: Path,
    emb_cache_dir: Path,
    duration: int,
    compact: bool,
    workers: int,
    video_batch: int,
    frame_batch: int,
    count_cache_misses,
    prefill_emb_cache,
    load_csv_with_emb_cache,
) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    metadata = {
        (Path(str(row.video_path)).name, str(row.source_model), str(row.subset)): row
        for row in df.itertuples(index=False)
    }
    n_miss = count_cache_misses(str(csv_path), str(emb_cache_dir), duration_sec=duration, compact=compact)
    print(f"[STALL] cache misses={n_miss}, cached={len(df)-n_miss}, csv={csv_path.name}")
    for _ in tqdm(
        prefill_emb_cache(
            str(csv_path),
            emb_cache_dir=str(emb_cache_dir),
            model=model,
            batch_size=frame_batch,
            duration_sec=duration,
            compact=compact,
            num_workers=workers,
            video_batch=video_batch,
        ),
        desc=f"STALL extracting {csv_path.stem}",
        unit="video",
        total=n_miss,
    ):
        pass

    rows: list[dict[str, Any]] = []
    iterable = load_csv_with_emb_cache(
        str(csv_path),
        emb_cache_dir=str(emb_cache_dir),
        model=model,
        batch_size=frame_batch,
        duration_sec=duration,
        compact=compact,
    )
    for sample in tqdm(iterable, desc=f"STALL scoring {csv_path.stem}", unit="video", total=len(df)):
        result = model._scores_from_embs(sample["embs"])
        filename = sample["filename"]
        meta = metadata.get((filename, str(sample["source_model"]), str(sample["subset"])))
        if meta is None:
            # fallback should almost never happen
            label = 0 if sample["subset"] == "real" else 1
            rel_path = ""
            video_id = ""
        else:
            label = int(meta.label)
            rel_path = str(meta.rel_path)
            video_id = str(meta.video_id)
        final_score = float(np.asarray(result["final_score"]).reshape(-1)[0])  # higher = real
        spat_score = float(np.asarray(result["spat_percentile"]).reshape(-1)[0])
        temp_score = float(np.asarray(result["temp_percentile"]).reshape(-1)[0])
        rows.append(
            {
                "rel_path": rel_path,
                "video_id": video_id,
                "filename": filename,
                "generator": sample["source_model"],
                "subset": sample["subset"],
                "label": label,
                "stall_real_score": final_score,
                "stall_fake_score": 1.0 - final_score,
                "stall_spatial_real_score": spat_score,
                "stall_temporal_real_score": temp_score,
                "stall_spatial_fake_score": 1.0 - spat_score,
                "stall_temporal_fake_score": 1.0 - temp_score,
                "error": "",
            }
        )
    return rows


def uniform_indices(total_frames: int, num_frames: int) -> list[int]:
    """Return exactly ``num_frames`` uniformly spaced native frame indices.

    If a video has fewer native frames than requested, indices are repeated.
    This matches the fixed-length evaluation idea used by STF3/D3-style test
    scripts and avoids dropping low-FPS real videos.
    """
    if total_frames <= 0:
        raise ValueError(f"invalid total_frames={total_frames}")
    if num_frames <= 0:
        raise ValueError(f"invalid num_frames={num_frames}")
    return np.linspace(0, total_frames - 1, num_frames).round().astype(int).tolist()


def _safe_source_model(row: dict[str, str]) -> str:
    return row.get("generator") or row.get("source_model") or "unknown"


def uniform_cache_path(cache_root: Path, row: dict[str, str], num_frames: int) -> Path:
    rel_path = row.get("rel_path") or row.get("video_path") or row.get("filename") or ""
    digest = hashlib.sha1(rel_path.encode("utf-8", errors="ignore")).hexdigest()[:16]
    subset = "real" if int(row["label"]) == 0 else "annotated"
    source_model = _safe_source_model(row)
    stem = Path(rel_path).stem or row.get("video_id") or digest
    return cache_root / subset / source_model / f"{stem}_{digest}_uniform{num_frames}.pt"


def load_video_frames_exact(video_path: Path, frame_indices: list[int]) -> np.ndarray:
    """Load frames in the requested order, preserving duplicate indices."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video: {video_path}")
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"failed to read frame {idx} from {video_path}")
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {video_path}")
    return np.stack(frames, axis=0)


def _uniform_item_base(row: dict[str, str], data_root: Path) -> dict[str, Any]:
    rel_path = row["rel_path"]
    video_path = data_root / rel_path
    label = int(row["label"])
    return {
        "video_id": row.get("video_id", ""),
        "rel_path": rel_path,
        "video_path": str(video_path),
        "filename": row.get("filename", video_path.name),
        "generator": _safe_source_model(row),
        "subset": "real" if label == 0 else "annotated",
        "label": label,
        "label_name": row.get("label_name", ""),
    }


def _decode_uniform_job(job: tuple[dict[str, str], Path, int]) -> tuple[dict[str, Any] | None, np.ndarray | None, dict[str, Any] | None]:
    row, data_root, num_frames = job
    item = _uniform_item_base(row, data_root)
    video_path = Path(item["video_path"])
    try:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if total_frames <= 0:
            raise RuntimeError(f"invalid frame count: {total_frames}")
        idxs = uniform_indices(total_frames, num_frames)
        frames = load_video_frames_exact(video_path, idxs)
        item.update(
            {
                "fps": fps,
                "num_frames_native": total_frames,
                "uniform_num_frames": num_frames,
                "uniform_idxs": json.dumps(idxs),
            }
        )
        return item, frames, None
    except Exception as exc:
        item["error"] = f"{type(exc).__name__}: {exc}"
        return None, None, item


def _score_uniform_emb(item: dict[str, Any], emb: np.ndarray, model) -> dict[str, Any]:
    result = model._scores_from_embs(emb[np.newaxis])
    final_score = float(np.asarray(result["final_score"]).reshape(-1)[0])  # higher = real
    spat_score = float(np.asarray(result["spat_percentile"]).reshape(-1)[0])
    temp_score = float(np.asarray(result["temp_percentile"]).reshape(-1)[0])
    return {
        "rel_path": item["rel_path"],
        "video_id": item["video_id"],
        "filename": item["filename"],
        "generator": item["generator"],
        "subset": item["subset"],
        "label": int(item["label"]),
        "fps": item.get("fps", ""),
        "num_frames_native": item.get("num_frames_native", ""),
        "uniform_num_frames": item.get("uniform_num_frames", ""),
        "uniform_idxs": item.get("uniform_idxs", ""),
        "stall_real_score": final_score,
        "stall_fake_score": 1.0 - final_score,
        "stall_spatial_real_score": spat_score,
        "stall_temporal_real_score": temp_score,
        "stall_spatial_fake_score": 1.0 - spat_score,
        "stall_temporal_fake_score": 1.0 - temp_score,
        "error": "",
    }


def score_split_uniform(
    *,
    model,
    split_rows: list[dict[str, str]],
    data_root: Path,
    emb_cache_dir: Path,
    split_name: str,
    num_frames: int,
    max_samples: int,
    workers: int,
    video_batch: int,
    frame_batch: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score a split with STF3-matched uniform-N frame sampling.

    This path intentionally bypasses STALL's official 8-fps/1-second filtering,
    because GenVideo-Val real MSR-VTT videos are often 3 fps and would otherwise
    be removed from the binary comparison.
    """
    selected = split_rows[:max_samples] if max_samples and max_samples > 0 else split_rows
    emb_cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, str], Path]] = []

    def flush_pending() -> None:
        nonlocal pending, rows, errors
        if not pending:
            return
        jobs = [(row, data_root, num_frames) for row, _ in pending]
        cache_paths = [cache_path for _, cache_path in pending]
        pending = []

        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as ex:
            decoded = list(ex.map(_decode_uniform_job, jobs))

        ok_items: list[dict[str, Any]] = []
        ok_frames: list[np.ndarray] = []
        ok_cache_paths: list[Path] = []
        for (item, frames, err), cache_path in zip(decoded, cache_paths):
            if err is not None:
                errors.append(err)
                continue
            if item is None or frames is None:
                continue
            ok_items.append(item)
            ok_frames.append(frames)
            ok_cache_paths.append(cache_path)

        if not ok_items:
            return

        lengths = [len(frames) for frames in ok_frames]
        flat_frames = [frame for video in ok_frames for frame in video]
        flat_embs = model._embed_flat_frames(flat_frames, frame_batch)
        cursor = 0
        for item, length, cache_path in zip(ok_items, lengths, ok_cache_paths):
            emb = flat_embs[cursor : cursor + length]
            cursor += length
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp.pt")
            torch.save(torch.from_numpy(emb), tmp)
            tmp.rename(cache_path)
            rows.append(_score_uniform_emb(item, emb, model))

    for row in tqdm(selected, desc=f"STALL uniform-{num_frames} {split_name}", unit="video"):
        item = _uniform_item_base(row, data_root)
        cache_path = uniform_cache_path(emb_cache_dir, row, num_frames)
        if cache_path.exists():
            try:
                emb = torch.load(cache_path, weights_only=True).cpu().numpy()
                item.update(
                    {
                        "uniform_num_frames": num_frames,
                        "uniform_idxs": "",
                    }
                )
                rows.append(_score_uniform_emb(item, emb, model))
                continue
            except Exception as exc:
                # Bad/partial cache should not poison the run; recompute below.
                try:
                    cache_path.unlink()
                except OSError:
                    pass
                print(f"[warn] cache reload failed, recomputing {cache_path}: {exc}")

        pending.append((row, cache_path))
        if len(pending) >= max(1, video_batch):
            flush_pending()
    flush_pending()
    return rows, errors


def valid_arrays(rows: list[dict[str, Any]], score_key: str = "stall_fake_score") -> tuple[np.ndarray, np.ndarray]:
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


def apply_predictions(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        score = float(item["stall_fake_score"])
        pred = 1 if score >= threshold else 0
        item["pred"] = pred
        item["correct"] = int(pred == int(item["label"]))
        out.append(item)
    return out


def write_markdown(path: Path, summary: dict[str, Any], result_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# STALL OOD comparison on GenVideo-Val",
        "",
        "STALL is evaluated as a recent training-free spatial-temporal dual-branch external baseline. "
        "The official VATEX-calibrated STALL parameters are used; score thresholds are selected on the OOD validation split and fixed on OOD test.",
        "",
        "## Configuration",
        "",
        f"- STALL root: `{summary['stall_root']}`",
        f"- Params: `{summary['params']}`",
        f"- DINOv3 repo: `{summary['dino_repo']}`",
        f"- DINOv3 weights: `{summary['dino_weights']}`",
        f"- Sampling mode: `{summary.get('sampling_mode', 'official_fps')}`",
        f"- Duration: `{summary['duration']}` second(s)",
        f"- Target FPS: `{summary['target_fps']}`",
        f"- Uniform frames: `{summary.get('num_frames')}`",
        f"- Effective frames: `{summary['effective_frames']}`",
        f"- Compact cache: `{summary['compact']}`",
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
    parser.add_argument("--stall-root", default="comparison_experiment/STALL")
    parser.add_argument("--data-root", default="data/GenVideo-Val")
    parser.add_argument("--val-csv", default="data/GenVideo-Val/splits/ood_val.csv")
    parser.add_argument("--test-csv", default="data/GenVideo-Val/splits/ood_test.csv")
    parser.add_argument("--out-dir", default="comparison_experiment/results/STALL/ood_vatex_dinov3_t8")
    parser.add_argument("--params", default="comparison_experiment/STALL/precomputed/stall_params_vatex_dino_v3.npz")
    parser.add_argument("--dino-repo", default="comparison_experiment/STALL/dinov3")
    parser.add_argument("--dino-weights", default="comparison_experiment/STALL/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--duration", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument(
        "--sampling-mode",
        choices=["official_fps", "uniform"],
        default="official_fps",
        help="official_fps keeps STALL's original target-FPS window filtering; uniform uses STF3-matched uniform-N frames.",
    )
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames for --sampling-mode uniform.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--video-batch", type=int, default=4)
    parser.add_argument("--frame-batch", type=int, default=8)
    parser.add_argument("--compact", action="store_true", default=True)
    parser.add_argument("--no-compact", dest="compact", action="store_false")
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    stall_root = (project_root / args.stall_root).resolve()
    data_root = (project_root / args.data_root).resolve()
    val_csv = (project_root / args.val_csv).resolve()
    test_csv = (project_root / args.test_csv).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    params = (project_root / args.params).resolve()
    dino_repo = (project_root / args.dino_repo).resolve()
    dino_weights = (project_root / args.dino_weights).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not params.exists():
        raise FileNotFoundError(f"STALL params not found: {params}")
    if not dino_repo.exists():
        raise FileNotFoundError(f"DINOv3 repo not found: {dino_repo}")
    if not dino_weights.exists():
        raise FileNotFoundError(
            "DINOv3 weights not found:\n"
            f"  {dino_weights}\n\n"
            "Download `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` "
            "from Meta DINOv3 downloads and place it there before running."
        )

    start = time.time()
    STALL, count_cache_misses, load_csv_with_emb_cache, prefill_emb_cache, compute_windows, downsample_frames, get_video_metadata = import_stall(stall_root)
    data = np.load(params)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    elif args.device.startswith("cuda"):
        device = "cuda"
    else:
        device = args.device
    model = STALL(device=device, data_dict=data, load_dino=True, dino_repo=str(dino_repo), dino_weights=str(dino_weights))

    if args.sampling_mode == "uniform":
        cache_dir = out_dir / f"emb_cache_uniform{args.num_frames}"
        val_split_rows = read_rows(val_csv)
        test_split_rows = read_rows(test_csv)
        val_rows, val_errors = score_split_uniform(
            model=model,
            split_rows=val_split_rows,
            data_root=data_root,
            emb_cache_dir=cache_dir,
            split_name="val",
            num_frames=args.num_frames,
            max_samples=args.max_val_samples,
            workers=args.workers,
            video_batch=args.video_batch,
            frame_batch=args.frame_batch,
        )
        test_rows, test_errors = score_split_uniform(
            model=model,
            split_rows=test_split_rows,
            data_root=data_root,
            emb_cache_dir=cache_dir,
            split_name="test",
            num_frames=args.num_frames,
            max_samples=args.max_test_samples,
            workers=args.workers,
            video_batch=args.video_batch,
            frame_batch=args.frame_batch,
        )
        write_csv(out_dir / "uniform_val_errors.csv", val_errors)
        write_csv(out_dir / "uniform_test_errors.csv", test_errors)
        val_index_len = min(len(val_split_rows), args.max_val_samples) if args.max_val_samples and args.max_val_samples > 0 else len(val_split_rows)
        test_index_len = min(len(test_split_rows), args.max_test_samples) if args.max_test_samples and args.max_test_samples > 0 else len(test_split_rows)
        val_error_count = len(val_errors) + max(0, val_index_len - len(val_rows) - len(val_errors))
        test_error_count = len(test_errors) + max(0, test_index_len - len(test_rows) - len(test_errors))
    else:
        index_dir = out_dir / "indexes"
        cache_dir = out_dir / "emb_cache"
        val_index, val_errors = build_stall_index(
            split_rows=read_rows(val_csv),
            data_root=data_root,
            out_csv=index_dir / "val_index.csv",
            error_csv=index_dir / "val_index_errors.csv",
            target_fps=args.target_fps,
            workers=args.workers,
            max_samples=args.max_val_samples,
            force=args.force_index,
            compute_windows=compute_windows,
            downsample_frames=downsample_frames,
            get_video_metadata=get_video_metadata,
        )
        test_index, test_errors = build_stall_index(
            split_rows=read_rows(test_csv),
            data_root=data_root,
            out_csv=index_dir / "test_index.csv",
            error_csv=index_dir / "test_index_errors.csv",
            target_fps=args.target_fps,
            workers=args.workers,
            max_samples=args.max_test_samples,
            force=args.force_index,
            compute_windows=compute_windows,
            downsample_frames=downsample_frames,
            get_video_metadata=get_video_metadata,
        )
        val_rows = score_index(
            model=model,
            csv_path=index_dir / "val_index.csv",
            emb_cache_dir=cache_dir,
            duration=args.duration,
            compact=args.compact,
            workers=args.workers,
            video_batch=args.video_batch,
            frame_batch=args.frame_batch,
            count_cache_misses=count_cache_misses,
            prefill_emb_cache=prefill_emb_cache,
            load_csv_with_emb_cache=load_csv_with_emb_cache,
        )
        test_rows = score_index(
            model=model,
            csv_path=index_dir / "test_index.csv",
            emb_cache_dir=cache_dir,
            duration=args.duration,
            compact=args.compact,
            workers=args.workers,
            video_batch=args.video_batch,
            frame_batch=args.frame_batch,
            count_cache_misses=count_cache_misses,
            prefill_emb_cache=prefill_emb_cache,
            load_csv_with_emb_cache=load_csv_with_emb_cache,
        )
        val_error_count = len(val_errors) + (len(val_index) - len(val_rows))
        test_error_count = len(test_errors) + (len(test_index) - len(test_rows))

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
                "model": "STALL_VATEX_DINOv3",
                "objective": objective,
                "threshold": threshold,
            },
        )

    summary = {
        "model": "STALL_VATEX_DINOv3",
        "stall_root": str(stall_root),
        "params": str(params),
        "dino_repo": str(dino_repo),
        "dino_weights": str(dino_weights),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "sampling_mode": args.sampling_mode,
        "duration": args.duration,
        "target_fps": args.target_fps,
        "num_frames": args.num_frames if args.sampling_mode == "uniform" else None,
        "effective_frames": int(args.num_frames if args.sampling_mode == "uniform" else round(args.duration * args.target_fps)),
        "compact": args.compact,
        "workers": args.workers,
        "video_batch": args.video_batch,
        "frame_batch": args.frame_batch,
        "val_valid": int(len(val_y)),
        "test_valid": int(len(test_y)),
        "val_errors": int(val_error_count),
        "test_errors": int(test_error_count),
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
