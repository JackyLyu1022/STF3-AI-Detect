from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_metric(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_metrics(outputs_root: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(outputs_root.glob("*/metrics.json")):
        exp = metrics_path.parent.name
        m = read_metric(metrics_path)
        row = {
            "experiment": exp,
            "model": m.get("model", ""),
            "num_samples": m.get("num_samples", ""),
            "acc": m.get("acc", ""),
            "auc": m.get("auc", ""),
            "f1": m.get("f1", ""),
            "precision": m.get("precision", ""),
            "recall": m.get("recall", ""),
            "seconds": m.get("seconds", ""),
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--out", default="outputs/summary.csv")
    args = parser.parse_args()

    rows = find_metrics(Path(args.outputs_root))
    if not rows:
        print("No metrics.json found under outputs/. Run evaluate.py first.")
        return
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
