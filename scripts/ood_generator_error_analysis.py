from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
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


def load_thresholds(path: Path | None, objective: str) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    thresholds: dict[str, float] = {}
    for row in read_csv(path):
        if row.get("objective") == objective:
            thresholds[row["id"]] = float(row["threshold"])
    return thresholds


def per_generator(rows: list[dict[str, str]], threshold: float) -> list[dict[str, object]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0, "real_n": 0, "fake_n": 0, "tn": 0, "fp": 0, "fn": 0, "tp": 0})
    for row in rows:
        generator = row.get("generator", "unknown")
        y = int(row["label"])
        pred = 1 if float(row["fake_prob"]) >= threshold else 0
        bucket = stats[generator]
        bucket["n"] += 1
        bucket["correct"] += int(y == pred)
        if y == 0:
            bucket["real_n"] += 1
            if pred == 0:
                bucket["tn"] += 1
            else:
                bucket["fp"] += 1
        else:
            bucket["fake_n"] += 1
            if pred == 0:
                bucket["fn"] += 1
            else:
                bucket["tp"] += 1
    out: list[dict[str, object]] = []
    for generator, d in sorted(stats.items()):
        precision = safe_div(d["tp"], d["tp"] + d["fp"])
        recall = safe_div(d["tp"], d["tp"] + d["fn"])
        specificity = safe_div(d["tn"], d["tn"] + d["fp"])
        out.append(
            {
                "generator": generator,
                "n": d["n"],
                "real_n": d["real_n"],
                "fake_n": d["fake_n"],
                "acc": safe_div(d["correct"], d["n"]),
                "precision": precision,
                "recall": recall if d["fake_n"] else "",
                "specificity": specificity if d["real_n"] else "",
                "tn": d["tn"],
                "fp": d["fp"],
                "fn": d["fn"],
                "tp": d["tp"],
            }
        )
    return out


def path_key(rows: list[dict[str, str]]) -> str:
    keys = rows[0].keys()
    if "rel_path" in keys:
        return "rel_path"
    for key in keys:
        if "path" in key.lower():
            return key
    raise KeyError(f"No path key found in columns: {list(keys)}")


def compare_fn_fix(
    a_rows: list[dict[str, str]],
    b_rows: list[dict[str, str]],
    a_threshold: float,
    b_threshold: float,
    a_id: str = "R5",
    b_id: str = "R7",
) -> list[dict[str, object]]:
    key = path_key(a_rows)
    b_by_path = {row[key]: row for row in b_rows}
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"a_fn": 0, "b_fixed": 0, "b_still_fn": 0, "b_new_fn": 0})
    for a in a_rows:
        b = b_by_path.get(a[key])
        if b is None or int(a["label"]) != 1:
            continue
        generator = a.get("generator", "unknown")
        a_pred = 1 if float(a["fake_prob"]) >= a_threshold else 0
        b_pred = 1 if float(b["fake_prob"]) >= b_threshold else 0
        if a_pred == 0:
            stats[generator]["a_fn"] += 1
            if b_pred == 1:
                stats[generator]["b_fixed"] += 1
            else:
                stats[generator]["b_still_fn"] += 1
        elif b_pred == 0:
            stats[generator]["b_new_fn"] += 1
    rows: list[dict[str, object]] = []
    for generator, d in sorted(stats.items()):
        rows.append(
            {
                "generator": generator,
                f"{a_id}_fn": d["a_fn"],
                f"{b_id}_fixed_{a_id}_fn": d["b_fixed"],
                f"{b_id}_still_fn_from_{a_id}_fn": d["b_still_fn"],
                f"{b_id}_new_fn_vs_{a_id}": d["b_new_fn"],
                "fix_rate": safe_div(d["b_fixed"], d["a_fn"]),
            }
        )
    return rows


def write_markdown(path: Path, rows: list[dict[str, object]], compare_rows: list[dict[str, object]]) -> None:
    lines = [
        "# OOD Generator-Level Error Analysis",
        "",
        "| ID | Method | Threshold | Generator | N | ACC | Recall | FN | FP |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['method']} | {float(row['threshold']):.4f} | {row['generator']} | {row['n']} | {float(row['acc']):.4f} | {row['recall'] if row['recall'] == '' else f'{float(row['recall']):.4f}'} | {row['fn']} | {row['fp']} |"
        )
    if compare_rows:
        lines.extend(
            [
                "",
                "## R5 FN Fixed by R7",
                "",
                "| Generator | R5 FN | R7 Fixed | R7 Still FN | R7 New FN | Fix Rate |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in compare_rows:
            lines.append(
                f"| {row['generator']} | {row['R5_fn']} | {row['R7_fixed_R5_fn']} | {row['R7_still_fn_from_R5_fn']} | {row['R7_new_fn_vs_R5']} | {float(row['fix_rate']):.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--ids", default="R1,R2,R3,R4,R5,R6,R7")
    parser.add_argument("--extra-experiments-json", default="", help="Optional JSON dict with method/out_dir entries.")
    parser.add_argument("--thresholds-csv", default="")
    parser.add_argument("--objective", default="max_f1")
    parser.add_argument("--out-dir", default="outputs/ood_followup")
    args = parser.parse_args()

    root = Path(args.project_root)
    out_dir = root / args.out_dir
    experiments = load_experiments(root, args.extra_experiments_json)
    threshold_path = root / args.thresholds_csv if args.thresholds_csv else None
    thresholds = load_thresholds(threshold_path, args.objective)
    selected_ids = [item.strip() for item in args.ids.split(",") if item.strip()]

    all_rows: list[dict[str, object]] = []
    prediction_rows: dict[str, list[dict[str, str]]] = {}
    used_thresholds: dict[str, float] = {}
    for exp_id in selected_ids:
        cfg = experiments[exp_id]
        pred_path = root / cfg["out_dir"] / "predictions.csv"
        if not pred_path.exists():
            print(f"[skip] {exp_id} missing {pred_path}")
            continue
        threshold = thresholds.get(exp_id, 0.5)
        rows = read_csv(pred_path)
        prediction_rows[exp_id] = rows
        used_thresholds[exp_id] = threshold
        for gen_row in per_generator(rows, threshold):
            gen_row.update({"id": exp_id, "method": cfg["method"], "threshold": threshold})
            all_rows.append(gen_row)

    compare_rows: list[dict[str, object]] = []
    if "R5" in prediction_rows and "R7" in prediction_rows:
        compare_rows = compare_fn_fix(
            prediction_rows["R5"],
            prediction_rows["R7"],
            used_thresholds.get("R5", 0.5),
            used_thresholds.get("R7", 0.5),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.objective if threshold_path else "default_0.5"
    write_csv(out_dir / f"generator_error_analysis_{suffix}.csv", all_rows)
    write_csv(out_dir / f"r5_r7_fn_comparison_{suffix}.csv", compare_rows)
    write_markdown(out_dir / f"generator_error_analysis_{suffix}.md", all_rows, compare_rows)
    (out_dir / f"generator_error_analysis_{suffix}.json").write_text(
        json.dumps({"per_generator": all_rows, "r5_r7_fn_comparison": compare_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[write] {out_dir / f'generator_error_analysis_{suffix}.csv'}")
    print(f"[write] {out_dir / f'r5_r7_fn_comparison_{suffix}.csv'}")
    print(f"[write] {out_dir / f'generator_error_analysis_{suffix}.md'}")


if __name__ == "__main__":
    main()
