from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


R5 = {"id": "R5", "method": "spatial_temporal_new", "out_dir": "outputs/ood_spatial_temporal_new"}
R7 = {"id": "R7", "method": "stf3_new", "out_dir": "outputs/ood_stf3_new"}
OBJECTIVES = ("max_f1", "max_balanced_acc", "precision_0.95_recall_max")


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


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def path_key(rows: list[dict[str, str]]) -> str:
    if "rel_path" in rows[0]:
        return "rel_path"
    for key in rows[0].keys():
        if "path" in key.lower():
            return key
    raise KeyError("No path-like key found.")


def align_rows(r5_rows: list[dict[str, str]], r7_rows: list[dict[str, str]], weight_r5: float) -> list[dict[str, object]]:
    key = path_key(r5_rows)
    r7_by_path = {row[key]: row for row in r7_rows}
    aligned: list[dict[str, object]] = []
    for r5 in r5_rows:
        r7 = r7_by_path.get(r5[key])
        if r7 is None:
            continue
        p5 = float(r5["fake_prob"])
        p7 = float(r7["fake_prob"])
        aligned.append(
            {
                key: r5[key],
                "generator": r5.get("generator", ""),
                "video_id": r5.get("video_id", ""),
                "label": int(r5["label"]),
                "fake_prob_r5": p5,
                "fake_prob_r7": p7,
                "fake_prob_ensemble": weight_r5 * p5 + (1.0 - weight_r5) * p7,
            }
        )
    return aligned


def metrics_at(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    tn = fp = fn = tp = 0
    for row in rows:
        y = int(row["label"])
        pred = 1 if float(row["fake_prob_ensemble"]) >= threshold else 0
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


def candidate_thresholds(rows: list[dict[str, object]]) -> list[float]:
    values = sorted({round(float(row["fake_prob_ensemble"]), 10) for row in rows})
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
    if objective == "precision_0.95_recall_max":
        if precision < 0.95:
            return (-1.0, precision, f1, recall)
        return (recall, f1, precision, balanced_acc)
    raise ValueError(f"Unknown objective: {objective}")


def best_threshold(rows: list[dict[str, object]], objective: str) -> tuple[float, dict[str, object]]:
    best_t = 0.5
    best_m = metrics_at(rows, best_t)
    best_s = objective_score(best_m, objective)
    for threshold in candidate_thresholds(rows):
        current = metrics_at(rows, threshold)
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


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# OOD R5 + R7 Probability Ensemble",
        "",
        "Weight and threshold are selected on validation predictions, then fixed on test predictions.",
        "",
        "| Objective | w_R5 | w_R7 | Threshold | Test ACC | Test F1 | Test Precision | Test Recall | Test FN | Test FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {objective} | {w5} | {w7} | {threshold} | {acc} | {f1} | {precision} | {recall} | {fn} | {fp} |".format(
                objective=row["objective"],
                w5=fmt(row["weight_r5"]),
                w7=fmt(1.0 - float(row["weight_r5"])),
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
    parser.add_argument("--val-suffix", default="_val")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--out-dir", default="outputs/ood_followup")
    args = parser.parse_args()

    root = Path(args.project_root)
    out_dir = root / args.out_dir
    r5_val = read_csv(root / f"{R5['out_dir']}{args.val_suffix}" / "predictions.csv")
    r7_val = read_csv(root / f"{R7['out_dir']}{args.val_suffix}" / "predictions.csv")
    r5_test = read_csv(root / R5["out_dir"] / "predictions.csv")
    r7_test = read_csv(root / R7["out_dir"] / "predictions.csv")

    steps = int(round(1.0 / args.weight_step))
    weights = [round(i * args.weight_step, 10) for i in range(steps + 1)]
    result_rows: list[dict[str, object]] = []
    ensemble_predictions: dict[str, list[dict[str, object]]] = {}
    for objective in OBJECTIVES:
        best: dict[str, object] | None = None
        for weight_r5 in weights:
            val_rows = align_rows(r5_val, r7_val, weight_r5)
            threshold, val_metrics = best_threshold(val_rows, objective)
            score = objective_score(val_metrics, objective)
            if best is None or score > best["score"]:
                best = {
                    "weight_r5": weight_r5,
                    "threshold": threshold,
                    "score": score,
                    "val_metrics": val_metrics,
                }
        assert best is not None
        test_rows = align_rows(r5_test, r7_test, float(best["weight_r5"]))
        test_metrics = metrics_at(test_rows, float(best["threshold"]))
        for row in test_rows:
            row["pred"] = 1 if float(row["fake_prob_ensemble"]) >= float(best["threshold"]) else 0
            row["correct"] = int(int(row["label"]) == int(row["pred"]))
        ensemble_predictions[objective] = test_rows
        result_rows.append(
            {
                "objective": objective,
                "weight_r5": best["weight_r5"],
                "weight_r7": 1.0 - float(best["weight_r5"]),
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
                "test_fn": test_metrics["fn"],
                "test_fp": test_metrics["fp"],
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "r5_r7_ensemble_results.csv", result_rows)
    write_markdown(out_dir / "r5_r7_ensemble_results.md", result_rows)
    for objective, rows in ensemble_predictions.items():
        write_csv(out_dir / f"r5_r7_ensemble_predictions_{objective}.csv", rows)
    (out_dir / "r5_r7_ensemble_results.json").write_text(json.dumps(result_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {out_dir / 'r5_r7_ensemble_results.csv'}")
    print(f"[write] {out_dir / 'r5_r7_ensemble_results.md'}")
    print(f"[write] {out_dir / 'r5_r7_ensemble_results.json'}")


if __name__ == "__main__":
    main()
