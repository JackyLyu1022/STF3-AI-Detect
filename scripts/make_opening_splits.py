from __future__ import annotations

"""Create small/medium splits used for the opening-report experiments.

This script generates:

- data/GenVideo-Val/splits_ood_mini/
  - seen-generator train/val, unseen-generator OOD test
- data/GenVideo-Val/splits_medium/
  - medium-sized balanced random train/val/test

It is intentionally separate from prepare_genvideo_val.py so the main dataset
preparation remains unchanged and the opening-report experiments are
reproducible.
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def sample_fake_balanced_by_generator(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample fake videos roughly evenly across available generators."""
    fake = df[df.label == 1]
    gens = sorted(fake.generator.unique())
    if not gens or n <= 0:
        return fake.head(0)

    parts = []
    remaining = min(n, len(fake))
    per_gen = max(1, remaining // len(gens))
    for i, gen in enumerate(gens):
        sub = fake[fake.generator == gen]
        take = remaining if i == len(gens) - 1 else min(per_gen, len(sub), remaining)
        if take > 0:
            parts.append(sub.sample(n=take, random_state=seed + i))
            remaining -= take
        if remaining <= 0:
            break

    sampled = pd.concat(parts) if parts else fake.head(0)
    if len(sampled) < n:
        used = set(sampled.video_id.astype(str))
        extra = fake[~fake.video_id.astype(str).isin(used)]
        if len(extra):
            sampled = pd.concat(
                [sampled, extra.sample(n=min(n - len(sampled), len(extra)), random_state=seed + 999)]
            )
    return sampled


def sample_fake_proportional_by_generator(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample fake videos roughly according to random-split generator ratios."""
    fake = df[df.label == 1]
    gens = sorted(fake.generator.unique())
    if not gens or n <= 0:
        return fake.head(0)

    weights = fake.generator.value_counts(normalize=True).to_dict()
    parts = []
    allocated = 0
    for i, gen in enumerate(gens):
        sub = fake[fake.generator == gen]
        if i == len(gens) - 1:
            take = n - allocated
        else:
            take = max(1, int(round(n * weights[gen])))
            take = min(take, len(sub), n - allocated)
        if take > 0:
            parts.append(sub.sample(n=take, random_state=seed + i))
            allocated += take
        if allocated >= n:
            break

    sampled = pd.concat(parts) if parts else fake.head(0)
    if len(sampled) < n:
        used = set(sampled.video_id.astype(str))
        extra = fake[~fake.video_id.astype(str).isin(used)]
        sampled = pd.concat([sampled, extra.sample(n=min(n - len(sampled), len(extra)), random_state=seed + 999)])
    return sampled


def write_balanced_split(
    df: pd.DataFrame,
    out_path: Path,
    split: str,
    n_real: int,
    n_fake: int,
    seed: int,
    fake_sampler,
) -> None:
    real = df[df.label == 0].sample(n=min(n_real, int((df.label == 0).sum())), random_state=seed)
    fake = fake_sampler(df, n_fake, seed)
    part = pd.concat([real, fake]).sample(frac=1, random_state=seed).assign(split=split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[write] {out_path} total={len(part)} labels={dict(Counter(part.label))}")
    for gen, count in Counter(part.generator).most_common():
        print(f"  {gen:14s} {count}")


def make_ood_mini(data_root: Path, seed: int) -> None:
    out_dir = data_root / "splits_ood_mini"
    sizes = {"train": (160, 160), "val": (40, 40), "test": (80, 80)}
    for split, (n_real, n_fake) in sizes.items():
        df = pd.read_csv(data_root / "splits" / f"ood_{split}.csv")
        write_balanced_split(
            df,
            out_dir / f"ood_mini_{split}.csv",
            split,
            n_real,
            n_fake,
            seed,
            sample_fake_balanced_by_generator,
        )


def make_medium(data_root: Path, seed: int) -> None:
    out_dir = data_root / "splits_medium"
    sizes = {"train": (600, 600), "val": (150, 150), "test": (200, 200)}
    for split, (n_real, n_fake) in sizes.items():
        df = pd.read_csv(data_root / "splits" / f"random_{split}.csv")
        write_balanced_split(
            df,
            out_dir / f"medium_{split}.csv",
            split,
            n_real,
            n_fake,
            seed,
            sample_fake_proportional_by_generator,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/GenVideo-Val")
    parser.add_argument("--ood-seed", type=int, default=2026)
    parser.add_argument("--medium-seed", type=int, default=2027)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    make_ood_mini(data_root, args.ood_seed)
    make_medium(data_root, args.medium_seed)


if __name__ == "__main__":
    main()
