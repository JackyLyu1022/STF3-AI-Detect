from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def sample_group(df, n, seed):
    if len(df) <= n:
        return df.sample(frac=1, random_state=seed)
    return df.sample(n=n, random_state=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='data/GenVideo-Val/metadata.csv')
    parser.add_argument('--out-dir', default='data/GenVideo-Val/splits_mini')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train-real', type=int, default=160)
    parser.add_argument('--train-fake', type=int, default=160)
    parser.add_argument('--val-real', type=int, default=40)
    parser.add_argument('--val-fake', type=int, default=40)
    parser.add_argument('--test-real', type=int, default=80)
    parser.add_argument('--test-fake', type=int, default=80)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real = df[df.label == 0].sample(frac=1, random_state=args.seed).reset_index(drop=True)
    fake = df[df.label == 1].copy()

    # For fake samples, sample roughly uniformly across generators when possible.
    fake_parts = []
    total_fake = args.train_fake + args.val_fake + args.test_fake
    gens = sorted(fake.generator.unique())
    per_gen = max(1, total_fake // len(gens))
    remaining = total_fake
    for i, gen in enumerate(gens):
        sub = fake[fake.generator == gen]
        if i == len(gens) - 1:
            n = remaining
        else:
            n = min(per_gen, len(sub), remaining)
        fake_parts.append(sample_group(sub, n, args.seed + i))
        remaining -= n
        if remaining <= 0:
            break
    fake_sampled = pd.concat(fake_parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if len(fake_sampled) < total_fake:
        used = set(fake_sampled.video_id.astype(str))
        extra = fake[~fake.video_id.astype(str).isin(used)]
        fake_sampled = pd.concat([fake_sampled, sample_group(extra, total_fake - len(fake_sampled), args.seed + 999)])
        fake_sampled = fake_sampled.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    train = pd.concat([
        real.iloc[:args.train_real],
        fake_sampled.iloc[:args.train_fake],
    ]).sample(frac=1, random_state=args.seed).assign(split='train')

    val = pd.concat([
        real.iloc[args.train_real:args.train_real + args.val_real],
        fake_sampled.iloc[args.train_fake:args.train_fake + args.val_fake],
    ]).sample(frac=1, random_state=args.seed).assign(split='val')

    test = pd.concat([
        real.iloc[args.train_real + args.val_real:args.train_real + args.val_real + args.test_real],
        fake_sampled.iloc[args.train_fake + args.val_fake:args.train_fake + args.val_fake + args.test_fake],
    ]).sample(frac=1, random_state=args.seed).assign(split='test')

    for name, part in [('mini_train.csv', train), ('mini_val.csv', val), ('mini_test.csv', test)]:
        part.to_csv(out_dir / name, index=False, encoding='utf-8-sig')
        print('\n', name, len(part))
        print(part.label_name.value_counts().to_string())
        print(part.generator.value_counts().sort_index().to_string())

    print('\n[done]', out_dir)


if __name__ == '__main__':
    main()
