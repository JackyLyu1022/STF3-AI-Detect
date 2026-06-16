from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
BASE_FIELDS = [
    "video_id",
    "rel_path",
    "label",
    "label_name",
    "generator",
    "filename",
    "ext",
    "size_bytes",
    "split",
]
EXTRA_FIELDS = ["source_dataset", "source_class"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for name in BASE_FIELDS + EXTRA_FIELDS:
        if any(name in r for r in rows) and name not in fieldnames:
            fieldnames.append(name)
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prefix_genvideo_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    out = []
    for r in rows:
        rr = dict(r)
        rel = rr["rel_path"].replace("\\", "/")
        if not rel.startswith("GenVideo-Val/"):
            rr["rel_path"] = f"GenVideo-Val/{rel}"
        rr["split"] = split
        rr.setdefault("source_dataset", "GenVideo-Val")
        rr.setdefault("source_class", rr.get("generator", "unknown"))
        out.append(rr)
    return out


def collect_ucf(ucf_root: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for p in ucf_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            try:
                cls = p.relative_to(ucf_root).parts[0]
            except Exception:
                cls = p.parent.name
            by_class[cls].append(p)
    for cls in by_class:
        by_class[cls].sort(key=lambda x: str(x).lower())
    return dict(sorted(by_class.items()))


def stratified_sample(by_class: dict[str, list[Path]], n: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    classes = list(by_class.keys())
    if n <= 0:
        return []
    total = sum(len(v) for v in by_class.values())
    if n >= total:
        return [p for cls in classes for p in by_class[cls]]

    base = n // len(classes)
    rem = n % len(classes)
    selected: list[Path] = []
    leftovers: list[Path] = []

    for i, cls in enumerate(classes):
        files = list(by_class[cls])
        rng.shuffle(files)
        quota = base + (1 if i < rem else 0)
        take = min(quota, len(files))
        selected.extend(files[:take])
        leftovers.extend(files[take:])

    if len(selected) < n:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])
    selected = selected[:n]
    selected.sort(key=lambda x: str(x).lower())
    return selected


def make_ucf_rows(files: list[Path], data_root: Path, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i, p in enumerate(files):
        rel = p.relative_to(data_root).as_posix()
        cls = p.relative_to(data_root / "UCF-101").parts[0]
        rows.append(
            {
                "video_id": f"UCF101_{i:05d}",
                "rel_path": rel,
                "label": "0",
                "label_name": "real",
                "generator": "real_UCF101",
                "filename": p.name,
                "ext": p.suffix.lower(),
                "size_bytes": str(p.stat().st_size),
                "split": split,
                "source_dataset": "UCF-101",
                "source_class": cls,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--genvideo-root", default="data/GenVideo-Val")
    ap.add_argument("--ucf-root", default="data/UCF-101")
    ap.add_argument("--extra-real", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/STF3-RealRobust/splits")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    gen_root = Path(args.genvideo_root)
    ucf_root = Path(args.ucf_root)
    out_dir = Path(args.out_dir)

    train = read_rows(gen_root / "splits" / "ood_train.csv")
    val = read_rows(gen_root / "splits" / "ood_val.csv")
    test = read_rows(gen_root / "splits" / "ood_test.csv")

    train_pref = prefix_genvideo_rows(train, "train")
    val_pref = prefix_genvideo_rows(val, "val")
    test_pref = prefix_genvideo_rows(test, "test")

    by_class = collect_ucf(ucf_root)
    sampled = stratified_sample(by_class, args.extra_real, args.seed)
    ucf_rows = make_ucf_rows(sampled, data_root, "train")

    combined_train = train_pref + ucf_rows

    write_rows(out_dir / f"ood_train_genvideo_ucf101_real{args.extra_real}.csv", combined_train)
    write_rows(out_dir / "ood_val_genvideo_prefixed.csv", val_pref)
    write_rows(out_dir / "ood_test_genvideo_prefixed.csv", test_pref)
    write_rows(out_dir / f"ucf101_real{args.extra_real}_manifest.csv", ucf_rows)

    summary = {
        "train_total": len(combined_train),
        "train_label_counts": dict(Counter(r["label_name"] for r in combined_train)),
        "genvideo_train_total": len(train_pref),
        "ucf_added_total": len(ucf_rows),
        "ucf_class_count": len(by_class),
        "ucf_sampled_class_counts_minmax": [
            min(Counter(r["source_class"] for r in ucf_rows).values()) if ucf_rows else 0,
            max(Counter(r["source_class"] for r in ucf_rows).values()) if ucf_rows else 0,
        ],
        "val_total": len(val_pref),
        "val_label_counts": dict(Counter(r["label_name"] for r in val_pref)),
        "test_total": len(test_pref),
        "test_label_counts": dict(Counter(r["label_name"] for r in test_pref)),
        "data_root_for_training": str(data_root),
        "train_csv": str(out_dir / f"ood_train_genvideo_ucf101_real{args.extra_real}.csv"),
        "val_csv": str(out_dir / "ood_val_genvideo_prefixed.csv"),
        "test_csv": str(out_dir / "ood_test_genvideo_prefixed.csv"),
    }
    import json
    (out_dir / f"summary_real{args.extra_real}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
