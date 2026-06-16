from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ood_threshold_calibration import candidate_thresholds, metrics_at, objective_score
from src.utils import save_json


OBJECTIVES = ("max_f1", "max_balanced_acc", "precision_0.95_recall_max", "max_acc")
FEATURES = (
    "fake_prob",
    "fake_prob_spatial",
    "fake_prob_temporal",
    "fake_prob_frequency",
    "branch_weight_spatial",
    "branch_weight_temporal",
    "branch_weight_frequency",
)


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


def matrix(rows: list[dict[str, str]], features: tuple[str, ...]) -> np.ndarray:
    values: list[list[float]] = []
    for row in rows:
        item: list[float] = []
        for feature in features:
            raw = row.get(feature, "")
            try:
                item.append(float(raw))
            except (TypeError, ValueError):
                item.append(np.nan)
        values.append(item)
    return np.asarray(values, dtype=np.float64)


def labels(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


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


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, object]], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "Logistic regression is fitted on validation predictions only, then fixed on test predictions.",
        "",
        "| Objective | Threshold | Test ACC | Test F1 | Test Precision | Test Recall | Test AUC | Test AP | Test FN | Test FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {objective} | {threshold} | {acc} | {f1} | {precision} | {recall} | {auc} | {ap} | {fn} | {fp} |".format(
                objective=row["objective"],
                threshold=fmt(row["threshold"]),
                acc=fmt(row["test_acc"]),
                f1=fmt(row["test_f1"]),
                precision=fmt(row["test_precision"]),
                recall=fmt(row["test_recall"]),
                auc=fmt(row["test_auc"]),
                ap=fmt(row["test_ap"]),
                fn=row["test_fn"],
                fp=row["test_fp"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out-dir-input", required=True)
    parser.add_argument("--val-suffix", default="_val")
    parser.add_argument("--out-dir", default="outputs/ood_followup/optimization_runs")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    root = Path(args.project_root)
    out_dir = root / args.out_dir
    prefix = args.prefix or f"{args.id.lower()}_branch_lr"
    val_rows = read_csv(root / f"{args.out_dir_input}{args.val_suffix}" / "predictions.csv")
    test_rows_raw = read_csv(root / args.out_dir_input / "predictions.csv")

    available = tuple(feature for feature in FEATURES if feature in val_rows[0] and feature in test_rows_raw[0])
    if not available:
        raise ValueError("No calibration features found in prediction CSV files.")

    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    pipe.fit(matrix(val_rows, available), labels(val_rows))
    val_prob = pipe.predict_proba(matrix(val_rows, available))[:, 1]
    test_prob = pipe.predict_proba(matrix(test_rows_raw, available))[:, 1]

    val_calib_rows: list[dict[str, object]] = []
    for row, prob in zip(val_rows, val_prob):
        val_calib_rows.append({**row, "fake_prob": float(prob), "label": int(row["label"])})

    test_calib_rows: list[dict[str, object]] = []
    for row, prob in zip(test_rows_raw, test_prob):
        test_calib_rows.append({**row, "fake_prob": float(prob), "label": int(row["label"])})

    result_rows: list[dict[str, object]] = []
    prediction_sets: dict[str, list[dict[str, object]]] = {}
    test_y = labels(test_rows_raw)
    test_auc = float(roc_auc_score(test_y, test_prob))
    test_ap = float(average_precision_score(test_y, test_prob))
    for objective in OBJECTIVES:
        threshold, val_metrics = best_threshold(val_calib_rows, objective)
        test_as_strings = [{**row, "fake_prob": str(row["fake_prob"]), "label": str(row["label"])} for row in test_calib_rows]
        test_metrics = metrics_at(test_as_strings, threshold)
        rows_for_objective: list[dict[str, object]] = []
        for row in test_calib_rows:
            out_row = dict(row)
            out_row["pred"] = 1 if float(out_row["fake_prob"]) >= threshold else 0
            out_row["correct"] = int(int(out_row["label"]) == int(out_row["pred"]))
            rows_for_objective.append(out_row)
        prediction_sets[objective] = rows_for_objective
        result_rows.append(
            {
                "id": args.id,
                "method": args.method,
                "objective": objective,
                "features": ",".join(available),
                "threshold": threshold,
                "val_acc": val_metrics["acc"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_balanced_acc": val_metrics["balanced_acc"],
                "val_fn": val_metrics["fn"],
                "val_fp": val_metrics["fp"],
                "test_acc": test_metrics["acc"],
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_balanced_acc": test_metrics["balanced_acc"],
                "test_auc": test_auc,
                "test_ap": test_ap,
                "test_tn": test_metrics["tn"],
                "test_fn": test_metrics["fn"],
                "test_fp": test_metrics["fp"],
                "test_tp": test_metrics["tp"],
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / f"{prefix}_results.csv", result_rows)
    write_markdown(out_dir / f"{prefix}_results.md", result_rows, f"OOD STF3 Branch Calibration LR: {args.id}")
    for objective, rows in prediction_sets.items():
        write_csv(out_dir / f"{prefix}_predictions_{objective}.csv", rows)
        objective_dir = out_dir / f"{prefix}_{objective}"
        write_csv(objective_dir / "predictions.csv", rows)
        metrics = next(row for row in result_rows if row["objective"] == objective)
        save_json(
            {
                "acc": metrics["test_acc"],
                "auc": metrics["test_auc"],
                "ap": metrics["test_ap"],
                "f1": metrics["test_f1"],
                "precision": metrics["test_precision"],
                "recall": metrics["test_recall"],
                "balanced_acc": metrics["test_balanced_acc"],
                "confusion_matrix": [[metrics["test_tn"], metrics["test_fp"]], [metrics["test_fn"], metrics["test_tp"]]],
                "model": args.method,
                "objective": objective,
                "threshold": metrics["threshold"],
            },
            objective_dir / "metrics.json",
        )
    (out_dir / f"{prefix}_results.json").write_text(json.dumps(result_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_dir / f'{prefix}_results.csv'}")
    print(f"[write] {out_dir / f'{prefix}_results.md'}")
    print(f"[write] {out_dir / f'{prefix}_results.json'}")


if __name__ == "__main__":
    main()
