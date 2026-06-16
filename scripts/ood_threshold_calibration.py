from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPERIMENTS = {
    "R1": {"method": "frequency_wave", "out_dir": "outputs/ood_frequency_wave"},
    "R2": {"method": "spatial_dino", "out_dir": "outputs/ood_spatial_dino"},
    "R3": {"method": "temporal_d3_restrav", "out_dir": "outputs/ood_temporal_d3_restrav"},
    "R4": {"method": "spatial_frequency_new", "out_dir": "outputs/ood_spatial_frequency_new"},
    "R5": {"method": "spatial_temporal_new", "out_dir": "outputs/ood_spatial_temporal_new"},
    "R6": {"method": "stf3_new_concat", "out_dir": "outputs/ood_stf3_new_concat"},
    "R7": {"method": "stf3_new", "out_dir": "outputs/ood_stf3_new"},
}

OBJECTIVES = ("max_f1", "max_balanced_acc", "precision_0.95_recall_max", "max_acc")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_experiments(root: Path, extra_path: str) -> dict[str, dict[str, str]]:
    experiments = {key: dict(value) for key, value in EXPERIMENTS.items()}
    if extra_path:
        path = root / extra_path
        extra = json.loads(path.read_text(encoding="utf-8"))
        for key, value in extra.items():
            if "method" not in value or "out_dir" not in value:
                raise ValueError(f"Extra experiment {key} must include method and out_dir.")
            experiments[key] = {"method": str(value["method"]), "out_dir": str(value["out_dir"])}
    return experiments


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def metrics_at(rows: list[dict[str, str]], threshold: float, prob_key: str = "fake_prob") -> dict[str, object]:
    tn = fp = fn = tp = 0
    for row in rows:
        y = int(row["label"])
        p = float(row[prob_key])
        pred = 1 if p >= threshold else 0
        if y == 0 and pred == 0:
            tn += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 1 and pred == 0:
            fn += 1
        elif y == 1 and pred == 1:
            tp += 1
    acc = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)
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


def candidate_thresholds(rows: list[dict[str, str]], prob_key: str = "fake_prob") -> list[float]:
    values = sorted({round(float(row[prob_key]), 10) for row in rows})
    thresholds = {0.0, 0.5, 1.0}
    thresholds.update(values)
    for a, b in zip(values, values[1:]):
        thresholds.add((a + b) / 2.0)
    return sorted(thresholds)


def objective_score(metrics: dict[str, object], objective: str) -> tuple[float, float, float, float]:
    f1 = float(metrics["f1"])
    precision = float(metrics["precision"])
    recall = float(metrics["recall"])
    balanced_acc = float(metrics["balanced_acc"])
    if objective == "max_f1":
        return (f1, balanced_acc, recall, precision)
    if objective == "max_balanced_acc":
        return (balanced_acc, f1, recall, precision)
    if objective == "max_acc":
        acc = float(metrics["acc"])
        return (acc, f1, balanced_acc, recall)
    if objective == "precision_0.95_recall_max":
        if precision < 0.95:
            return (-1.0, precision, f1, recall)
        return (recall, f1, precision, balanced_acc)
    raise ValueError(f"Unknown objective: {objective}")


def find_threshold(rows: list[dict[str, str]], objective: str) -> tuple[float, dict[str, object]]:
    best_threshold = 0.5
    best_metrics = metrics_at(rows, best_threshold)
    best_score = objective_score(best_metrics, objective)
    for threshold in candidate_thresholds(rows):
        current = metrics_at(rows, threshold)
        current_score = objective_score(current, objective)
        if current_score > best_score:
            best_threshold = threshold
            best_metrics = current
            best_score = current_score
    return best_threshold, best_metrics


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# OOD Val-Calibrated Threshold Results",
        "",
        "Thresholds are selected on `ood_val.csv` predictions and then fixed before scoring `ood_test.csv` predictions.",
        "",
        "| ID | Method | Objective | Threshold | Test ACC | Test F1 | Test Precision | Test Recall | Test FN | Test FP |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {id} | {method} | {objective} | {threshold} | {test_acc} | {test_f1} | {test_precision} | {test_recall} | {test_fn} | {test_fp} |".format(
                id=row["id"],
                method=row["method"],
                objective=row["objective"],
                threshold=fmt(row["threshold"]),
                test_acc=fmt(row["test_acc"]),
                test_f1=fmt(row["test_f1"]),
                test_precision=fmt(row["test_precision"]),
                test_recall=fmt(row["test_recall"]),
                test_fn=row["test_fn"],
                test_fp=row["test_fp"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--ids", default="R1,R2,R3,R4,R5,R6,R7")
    parser.add_argument("--extra-experiments-json", default="", help="Optional JSON dict with method/out_dir entries.")
    parser.add_argument("--val-suffix", default="_val")
    parser.add_argument("--out-dir", default="outputs/ood_followup")
    args = parser.parse_args()

    root = Path(args.project_root)
    out_dir = root / args.out_dir
    experiments = load_experiments(root, args.extra_experiments_json)
    selected_ids = [item.strip() for item in args.ids.split(",") if item.strip()]
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}

    for exp_id in selected_ids:
        cfg = experiments[exp_id]
        test_pred = root / cfg["out_dir"] / "predictions.csv"
        val_pred = root / f"{cfg['out_dir']}{args.val_suffix}" / "predictions.csv"
        if not test_pred.exists() or not val_pred.exists():
            print(f"[skip] {exp_id} missing predictions: val={val_pred.exists()} test={test_pred.exists()}")
            continue
        val_rows = read_csv(val_pred)
        test_rows = read_csv(test_pred)
        default_test = metrics_at(test_rows, 0.5)
        summary[exp_id] = {"method": cfg["method"], "default_test": default_test, "objectives": {}}
        for objective in OBJECTIVES:
            threshold, val_metrics = find_threshold(val_rows, objective)
            test_metrics = metrics_at(test_rows, threshold)
            summary[exp_id]["objectives"][objective] = {
                "threshold": threshold,
                "val": val_metrics,
                "test": test_metrics,
            }
            rows.append(
                {
                    "id": exp_id,
                    "method": cfg["method"],
                    "objective": objective,
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
                    "test_fn": test_metrics["fn"],
                    "test_fp": test_metrics["fp"],
                    "default_test_acc": default_test["acc"],
                    "default_test_f1": default_test["f1"],
                    "default_test_precision": default_test["precision"],
                    "default_test_recall": default_test["recall"],
                    "default_test_fn": default_test["fn"],
                    "default_test_fp": default_test["fp"],
                }
            )

    write_csv(out_dir / "val_calibrated_thresholds.csv", rows)
    write_markdown(out_dir / "val_calibrated_thresholds.md", rows)
    (out_dir / "val_calibrated_thresholds.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_dir / 'val_calibrated_thresholds.csv'}")
    print(f"[write] {out_dir / 'val_calibrated_thresholds.md'}")
    print(f"[write] {out_dir / 'val_calibrated_thresholds.json'}")


if __name__ == "__main__":
    main()
