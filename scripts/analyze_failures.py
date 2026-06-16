"""Analyze STF3 medium predictions: failure cases + per-generator difficulty + model size."""
from __future__ import annotations
import csv, sys, io, json
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PRED_PATH = Path('outputs/medium_stf3/predictions.csv')

with PRED_PATH.open(encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

path_key = next((k for k in rows[0].keys() if 'path' in k.lower()), None)
assert path_key, f'no path column found, keys={list(rows[0].keys())}'

fn, fp, tp, tn = [], [], [], []
for r in rows:
    label = int(r['label']); pred = int(r['pred']); prob = float(r['fake_prob'])
    item = (r, prob)
    if   label == 1 and pred == 0: fn.append(item)
    elif label == 0 and pred == 1: fp.append(item)
    elif label == 1 and pred == 1: tp.append(item)
    else: tn.append(item)

print(f'Total: {len(rows)}  TP={len(tp)} TN={len(tn)} FP={len(fp)} FN={len(fn)}')

print('\n=== FN (most-confident wrong, lowest fake_prob, label=1) ===')
for r, p in sorted(fn, key=lambda x: x[1])[:10]:
    print(f'  prob={p:.3f}  gen={r["generator"]:20s}  path={r[path_key]}')

print('\n=== FP (most-confident wrong, highest fake_prob, label=0) ===')
for r, p in sorted(fp, key=lambda x: -x[1])[:10]:
    print(f'  prob={p:.3f}  gen={r["generator"]:20s}  path={r[path_key]}')

# Per-generator stats
by_gen = defaultdict(lambda: {'total': 0, 'correct': 0, 'fake_total': 0, 'fake_correct': 0})
for r in rows:
    g = r['generator']
    by_gen[g]['total']   += 1
    by_gen[g]['correct'] += int(r['correct'])
    if int(r['label']) == 1:
        by_gen[g]['fake_total']   += 1
        by_gen[g]['fake_correct'] += int(int(r['pred']) == 1)

print('\n=== Per-generator accuracy (sorted high → low) ===')
gen_stats = []
for g, d in by_gen.items():
    acc = d['correct'] / max(d['total'], 1)
    rec = d['fake_correct'] / max(d['fake_total'], 1) if d['fake_total'] else None
    gen_stats.append((g, d['total'], acc, rec, d['fake_correct'], d['fake_total']))
gen_stats.sort(key=lambda x: -x[2])
for g, n, acc, rec, fc, ft in gen_stats:
    rec_str = f'{rec:.3f}' if rec is not None else '  -  '
    print(f'  {g:25s}  n={n:3d}  acc={acc:.3f}  fake_recall={rec_str} ({fc}/{ft})')

# Save tier classification for PPT
tiers = {'easy': [], 'mid': [], 'hard': []}
for g, n, acc, rec, fc, ft in gen_stats:
    if ft == 0:  # real bucket
        continue
    if rec >= 0.85:    tiers['easy'].append((g, n, rec))
    elif rec >= 0.50:  tiers['mid'].append((g, n, rec))
    else:              tiers['hard'].append((g, n, rec))

Path('outputs/failure_analysis').mkdir(parents=True, exist_ok=True)
with open('outputs/failure_analysis/per_generator.json', 'w', encoding='utf-8') as f:
    json.dump({
        'gen_stats': [{'gen': g, 'n': n, 'acc': acc, 'recall': rec,
                       'fake_correct': fc, 'fake_total': ft}
                      for g, n, acc, rec, fc, ft in gen_stats],
        'tiers': tiers,
    }, f, ensure_ascii=False, indent=2)
print('\n[write] outputs/failure_analysis/per_generator.json')

# Save FN/FP sample paths for visualization
fn_picks = sorted(fn, key=lambda x: x[1])[:6]
fp_picks = sorted(fp, key=lambda x: -x[1])[:3]
with open('outputs/failure_analysis/cases.json', 'w', encoding='utf-8') as f:
    json.dump({
        'fn_cases': [{'path': r[path_key], 'gen': r['generator'], 'prob': p,
                      'label': int(r['label'])} for r, p in fn_picks],
        'fp_cases': [{'path': r[path_key], 'gen': r['generator'], 'prob': p,
                      'label': int(r['label'])} for r, p in fp_picks],
    }, f, ensure_ascii=False, indent=2)
print('[write] outputs/failure_analysis/cases.json')
