"""Prepare GenVideo-Val for STF3-Detect.

This script is intentionally conservative:
- It can extract GenVideo-Val.zip if present.
- It normalizes the directory layout to data/GenVideo-Val/video/<generator>/*.
- It removes harmless macOS metadata files/directories.
- It creates metadata.csv and train/val/test split CSV files.

Usage:
    python scripts/prepare_genvideo_val.py
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_OOD_FAKE_GENERATORS = ["MorphStudio", "Show_1", "Sora", "WildScrape"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_rmtree(path: Path, allowed_root: Path) -> None:
    path = path.resolve()
    allowed_root = allowed_root.resolve()
    if allowed_root not in path.parents and path != allowed_root:
        raise RuntimeError(f"Refusing to delete outside allowed root: {path}")
    if path.exists():
        shutil.rmtree(path)


def remove_macos_junk(root: Path) -> None:
    for p in list(root.rglob("__MACOSX")):
        if p.is_dir():
            safe_rmtree(p, root)
    for p in root.rglob(".DS_Store"):
        if p.is_file():
            p.unlink()
    for p in root.rglob("._*"):
        if p.is_file():
            p.unlink()


def maybe_extract_zip(data_root: Path) -> None:
    zip_candidates = [
        data_root / "GenVideo-Val.zip",
        data_root.parent / "GenVideo-Val.zip",
        project_root() / "data" / "GenVideo-Val.zip",
    ]
    zip_path = next((p for p in zip_candidates if p.exists()), None)
    nested = data_root / "GenVideo-Val"
    video_dir = data_root / "video"
    if zip_path and not nested.exists() and not video_dir.exists():
        print(f"[extract] {zip_path} -> {data_root}")
        data_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)
    elif zip_path:
        print(f"[extract] zip found but extracted directory already exists, skip: {zip_path}")
    else:
        print("[extract] no GenVideo-Val.zip found, assume already extracted")


def find_raw_root(data_root: Path) -> Path | None:
    candidates = [
        data_root / "GenVideo-Val",
        data_root,
    ]
    for c in candidates:
        if (c / "Fake").is_dir() and (c / "Real").is_dir():
            return c
    return None


def unique_target_dir(base: Path) -> Path:
    if not base.exists():
        return base
    # If it already contains files, we will merge into it. The caller handles file conflicts.
    return base


def move_or_merge_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            move_or_merge_dir(item, target)
            if item.exists() and not any(item.iterdir()):
                item.rmdir()
        else:
            if target.exists():
                # Same file already there from a previous run. Keep destination.
                if item.stat().st_size == target.stat().st_size:
                    item.unlink()
                else:
                    stem, suffix = target.stem, target.suffix
                    i = 1
                    while True:
                        alt = dst / f"{stem}__dup{i}{suffix}"
                        if not alt.exists():
                            shutil.move(str(item), str(alt))
                            break
                        i += 1
            else:
                shutil.move(str(item), str(target))
    if src.exists() and src.is_dir() and not any(src.iterdir()):
        src.rmdir()


def normalize_layout(data_root: Path) -> Path:
    video_dir = data_root / "video"
    raw_root = find_raw_root(data_root)

    if raw_root is None:
        if video_dir.is_dir():
            print(f"[layout] already normalized: {video_dir}")
            return video_dir
        raise FileNotFoundError(
            f"Cannot find GenVideo-Val raw layout. Expected Fake/ and Real/ under {data_root}"
        )

    print(f"[layout] raw root: {raw_root}")
    video_dir.mkdir(parents=True, exist_ok=True)

    real_dir = raw_root / "Real"
    if real_dir.exists():
        print("[layout] move Real -> video/real_MSRVTT")
        move_or_merge_dir(real_dir, video_dir / "real_MSRVTT")

    fake_dir = raw_root / "Fake"
    if fake_dir.exists():
        for gen_dir in sorted([p for p in fake_dir.iterdir() if p.is_dir()]):
            print(f"[layout] move Fake/{gen_dir.name} -> video/{gen_dir.name}")
            move_or_merge_dir(gen_dir, video_dir / gen_dir.name)
        if fake_dir.exists() and not any(fake_dir.iterdir()):
            fake_dir.rmdir()

    # Remove now-empty nested raw directory if possible.
    if raw_root != data_root and raw_root.exists():
        leftover = [p for p in raw_root.iterdir() if p.name not in {"Fake", "Real"}]
        # after junk removal this is usually empty; do not force-delete unexpected files here.
        if not any(raw_root.iterdir()):
            raw_root.rmdir()

    return video_dir


def iter_video_files(video_dir: Path) -> Iterable[Path]:
    for p in sorted(video_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("."):
            yield p


def build_metadata(data_root: Path, video_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for p in iter_video_files(video_dir):
        generator = p.parent.name
        is_real = generator.lower() in {"real", "real_msrvtt", "msrvtt"}
        label = 0 if is_real else 1
        rel_path = p.relative_to(data_root).as_posix()
        rows.append(
            {
                "video_id": f"{len(rows):06d}",
                "rel_path": rel_path,
                "label": str(label),
                "label_name": "real" if label == 0 else "fake",
                "generator": generator,
                "filename": p.name,
                "ext": p.suffix.lower(),
                "size_bytes": str(p.stat().st_size),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_group(rows: list[dict[str, str]], ratios=(0.70, 0.15, 0.15), seed=42):
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]
    return train, val, test


def add_split(rows: list[dict[str, str]], split_name: str) -> list[dict[str, str]]:
    return [dict(r, split=split_name) for r in rows]


def make_random_splits(rows: list[dict[str, str]], seed=42):
    # Stratify by generator so every generator is represented proportionally.
    by_gen = defaultdict(list)
    for r in rows:
        by_gen[r["generator"]].append(r)

    train, val, test = [], [], []
    for gen in sorted(by_gen):
        a, b, c = split_group(by_gen[gen], seed=seed)
        train.extend(a)
        val.extend(b)
        test.extend(c)

    rng = random.Random(seed)
    for part in [train, val, test]:
        rng.shuffle(part)
    return add_split(train, "train"), add_split(val, "val"), add_split(test, "test")


def make_ood_splits(rows: list[dict[str, str]], holdout_generators: list[str], seed=42):
    real = [r for r in rows if r["label"] == "0"]
    fake_seen = [r for r in rows if r["label"] == "1" and r["generator"] not in holdout_generators]
    fake_unseen = [r for r in rows if r["label"] == "1" and r["generator"] in holdout_generators]

    real_train, real_val, real_test = split_group(real, seed=seed)

    # For seen fake generators: train/val only, with a small in-domain test portion kept separate
    # for optional diagnostics. The official OOD test below uses unseen fake generators.
    by_gen = defaultdict(list)
    for r in fake_seen:
        by_gen[r["generator"]].append(r)
    fake_train, fake_val = [], []
    for gen in sorted(by_gen):
        a, b, _ = split_group(by_gen[gen], ratios=(0.85, 0.15, 0.0), seed=seed)
        fake_train.extend(a)
        fake_val.extend(b)

    rng = random.Random(seed)
    train = real_train + fake_train
    val = real_val + fake_val
    test = real_test + fake_unseen
    for part in [train, val, test]:
        rng.shuffle(part)

    return add_split(train, "train"), add_split(val, "val"), add_split(test, "test")


def summarize(name: str, rows: list[dict[str, str]]) -> None:
    label_counts = Counter(r["label_name"] for r in rows)
    gen_counts = Counter(r["generator"] for r in rows)
    print(f"\n[{name}] total={len(rows)} labels={dict(label_counts)}")
    for gen, n in sorted(gen_counts.items()):
        print(f"  {gen:14s} {n}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(project_root() / "data" / "GenVideo-Val"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ood-holdout", nargs="*", default=DEFAULT_OOD_FAKE_GENERATORS)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"[start] data_root={data_root}")
    maybe_extract_zip(data_root)
    remove_macos_junk(data_root)
    video_dir = normalize_layout(data_root)
    remove_macos_junk(data_root)

    rows = build_metadata(data_root, video_dir)
    if not rows:
        raise RuntimeError(f"No video files found under {video_dir}. Supported extensions: {sorted(VIDEO_EXTS)}")

    metadata_path = data_root / "metadata.csv"
    write_csv(metadata_path, rows)
    summarize("metadata", rows)
    print(f"[write] {metadata_path}")

    splits_dir = data_root / "splits"
    r_train, r_val, r_test = make_random_splits(rows, seed=args.seed)
    write_csv(splits_dir / "random_train.csv", r_train)
    write_csv(splits_dir / "random_val.csv", r_val)
    write_csv(splits_dir / "random_test.csv", r_test)
    summarize("random_train", r_train)
    summarize("random_val", r_val)
    summarize("random_test", r_test)

    existing_holdout = sorted({r["generator"] for r in rows if r["generator"] in args.ood_holdout})
    if not existing_holdout:
        raise RuntimeError(f"No requested OOD holdout generators found: {args.ood_holdout}")
    o_train, o_val, o_test = make_ood_splits(rows, existing_holdout, seed=args.seed)
    write_csv(splits_dir / "ood_train.csv", o_train)
    write_csv(splits_dir / "ood_val.csv", o_val)
    write_csv(splits_dir / "ood_test.csv", o_test)
    summarize("ood_train", o_train)
    summarize("ood_val", o_val)
    summarize("ood_test", o_test)
    print(f"[ood] holdout generators: {existing_holdout}")

    print("\n[done]")
    print(f"metadata: {metadata_path}")
    print(f"splits:   {splits_dir}")


if __name__ == "__main__":
    main()
