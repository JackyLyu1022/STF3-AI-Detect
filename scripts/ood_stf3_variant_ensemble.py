from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ood_threshold_calibration import metrics_at, objective_score, candidate_thresholds
from src.utils import save_json


OBJECTIVES = ("max_f1", "max_balanced_acc", "precision_0.95_recall_max", "max_acc")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
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


def path_key(rows: list[dict[str, str]]) -> str:
    if "rel_path" in rows[0]:
        return "rel_path"
    for key in rows[0].keys():
        if "path" in key.lower():
            return key
    raise KeyError("No path-like key found.")


def align_rows(a_rows: list[dict[str, str]], b_rows: list[dict[str, str]], weight_a: float, a_name: str, b_name: str) -> list[dict[str, object]]:
    key = path_key(a_rows)
    b_by_path = {row[key]: row for row in b_rows}
    aligned: list[dict[str, object]] = []
    for a in a_rows:
        b = b_by_path.get(a[key])
        if b is None:
            continue
        pa = float(a["fake_prob"])
        pb = float(b["fake_prob"])
        aligned.append(
            {
                key: a[key],
                "generator": a.get("generator", ""),
                "video_id": a.get("video_id", ""),
                "label": int(a["label"]),
                f"fake_prob_{a_name}": pa,
                f"fake_prob_{b_name}": pb,
                "fake_prob": weight_a * pa + (1.0 - weight_a) * pb,
            }
        )
    return aligned


def best_threshold(rows: list[dict[str, object]], objective: str) -> tuple[float, dict[str, object]]:
    as_strings = [{**row, "fake_prob": str(row["fake_prob"]), "label": str(row["label"])} for row in rows]
    best_t = 0.5
    best_m = metrics_at(as_strings, best_t)
    best_s = objective_score(best_m, objective)
    for threshold in candidate_thresholds(as_strings):
        current = metrics_at(as_strings, threshold)
        score = objective_score(current, objective)
        if score > best_s:
            best_t = threshold
            best_m = current
            best_s = score
    return best_t, best_m


def metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, object]:
    total = tn + fp + fn + tp
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced_acc = 0.5 * (recall + specificity)
    return {
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_acc": balanced_acc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def fast_best_threshold(rows: list[dict[str, object]], objective: str) -> tuple[float, dict[str, object]]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probs = np.asarray([float(row["fake_prob"]) for row in rows], dtype=np.float64)
    pos = int(labels.sum())
    neg = int(len(labels) - pos)

    candidates: list[tuple[float, dict[str, object]]] = []
    for threshold in (0.0, 0.5, 1.0):
        pred = probs >= threshold
        tp = int(((labels == 1) & pred).sum())
        fp = int(((labels == 0) & pred).sum())
        fn = pos - tp
        tn = neg - fp
        candidates.append((threshold, metrics_from_counts(tn, fp, fn, tp)))

    order = np.argsort(-probs)
    sorted_probs = probs[order]
    sorted_labels = labels[order]
    tp = fp = 0
    i = 0
    n = len(sorted_probs)
    while i < n:
        threshold = float(sorted_probs[i])
        j = i
        while j < n and sorted_probs[j] == threshold:
            if sorted_labels[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        fn = pos - tp
        tn = neg - fp
        candidates.append((threshold, metrics_from_counts(tn, fp, fn, tp)))
        i = j

    best_t, best_m = candidates[0]
    best_s = objective_score(best_m, objective)
    for threshold, current in candidates[1:]:
        score = objective_score(current, objective)
        if score > best_s:
            best_t = threshold
            best_m = current
            best_s = score
    return best_t, best_m


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, object]], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "Weight and threshold are selected on validation predictions, then fixed on test predictions.",
        "",
        "| Objective | w_A | w_B | Threshold | Test ACC | Test F1 | Test Precision | Test Recall | Test FN | Test FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {objective} | {wa} | {wb} | {threshold} | {acc} | {f1} | {precision} | {recall} | {fn} | {fp} |".format(
                objective=row["objective"],
                wa=fmt(row["weight_a"]),
                wb=fmt(row["weight_b"]),
                threshold=fmt(row["threshold"]),
                acc=fmt(row["test_acc"]),
                f1=fmt(row["test_f1"]),
                precision=fmt(row["test_precision"]),
                recall=fmt(row["test_recall"]),
                fn=row["test_fn"],
                fp=row["test_fp"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--a-id", required=True)
    parser.add_argument("--a-method", required=True)
    parser.add_argument("--a-out-dir", required=True)
    parser.add_argument("--b-id", required=True)
    parser.add_argument("--b-method", required=True)
    parser.add_argument("--b-out-dir", required=True)
    parser.add_argument("--val-suffix", default="_val")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--out-dir", default="outputs/ood_followup/optimization_runs")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    root = Path(args.project_root)
    out_dir = root / args.out_dir
    prefix = args.prefix or f"{args.a_id}_plus_{args.b_id}".lower()

    a_val = read_csv(root / f"{args.a_out_dir}{args.val_suffix}" / "predictions.csv")
    b_val = read_csv(root / f"{args.b_out_dir}{args.val_suffix}" / "predictions.csv")
    a_test = read_csv(root / args.a_out_dir / "predictions.csv")
    b_test = read_csv(root / args.b_out_dir / "predictions.csv")

    steps = int(round(1.0 / args.weight_step))
    weights = [round(i * args.weight_step, 10) for i in range(steps + 1)]
    result_rows: list[dict[str, object]] = []
    prediction_sets: dict[str, list[dict[str, object]]] = {}
    for objective in OBJECTIVES:
        best: dict[str, object] | None = None
        for weight_a in weights:
            val_rows = align_rows(a_val, b_val, weight_a, args.a_id, args.b_id)
            threshold, val_metrics = fast_best_threshold(val_rows, objective)
            score = objective_score(val_metrics, objective)
            if best is None or score > best["score"]:
                best = {"weight_a": weight_a, "threshold": threshold, "score": score, "val_metrics": val_metrics}
        assert best is not None
        test_rows = align_rows(a_test, b_test, float(best["weight_a"]), args.a_id, args.b_id)
        test_as_strings = [{**row, "fake_prob": str(row["fake_prob"]), "label": str(row["label"])} for row in test_rows]
        test_metrics = metrics_at(test_as_strings, float(best["threshold"]))
        for row in test_rows:
            row["pred"] = 1 if float(row["fake_prob"]) >= float(best["threshold"]) else 0
            row["correct"] = int(int(row["label"]) == int(row["pred"]))
        prediction_sets[objective] = test_rows
        result_rows.append(
            {
                "objective": objective,
                "a_id": args.a_id,
                "a_method": args.a_method,
                "b_id": args.b_id,
                "b_method": args.b_method,
                "weight_a": best["weight_a"],
                "weight_b": 1.0 - float(best["weight_a"]),
                "threshold": best["threshold"],
                "val_acc": best["val_metrics"]["acc"],
                "val_f1": best["val_metrics"]["f1"],
                "val_precision": best["val_metrics"]["precision"],
                "val_recall": best["val_metrics"]["recall"],
                "val_balanced_acc": best["val_metrics"]["balanced_acc"],
                "val_fn": best["val_metrics"]["fn"],
                "val_fp": best["val_metrics"]["fp"],
                "test_acc": test_metrics["acc"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_balanced_acc": test_metrics["balanced_acc"],
                "test_tn": test_metrics["tn"],
                "test_fn": test_metrics["fn"],
                "test_fp": test_metrics["fp"],
                "test_tp": test_metrics["tp"],
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / f"{prefix}_ensemble_results.csv", result_rows)
    write_markdown(out_dir / f"{prefix}_ensemble_results.md", result_rows, f"OOD STF3 Variant Ensemble: {args.a_id} + {args.b_id}")
    for objective, rows in prediction_sets.items():
        write_csv(out_dir / f"{prefix}_ensemble_predictions_{objective}.csv", rows)
        objective_dir = out_dir / f"{prefix}_{objective}"
        write_csv(objective_dir / "predictions.csv", rows)
        metrics = next(row for row in result_rows if row["objective"] == objective)
        save_json(
            {
                "acc": metrics["test_acc"],
                "f1": metrics["test_f1"],
                "precision": metrics["test_precision"],
                "recall": metrics["test_recall"],
                "balanced_acc": metrics["test_balanced_acc"],
                "confusion_matrix": [[metrics["test_tn"], metrics["test_fp"]], [metrics["test_fn"], metrics["test_tp"]]],
                "model": f"{args.a_id}+{args.b_id}",
                "objective": objective,
                "threshold": metrics["threshold"],
            },
            objective_dir / "metrics.json",
        )
    (out_dir / f"{prefix}_ensemble_results.json").write_text(json.dumps(result_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_dir / f'{prefix}_ensemble_results.csv'}")
    print(f"[write] {out_dir / f'{prefix}_ensemble_results.md'}")
    print(f"[write] {out_dir / f'{prefix}_ensemble_results.json'}")


if __name__ == "__main__":
    main()
