"""Compute params for each branch of STF3-Detect-Lite."""
from __future__ import annotations
import sys, io, json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.spatial_branch import SpatialBranch
from src.models.temporal_branch import TemporalBranch
from src.models.frequency_branch import FrequencyBranch
from src.models.stf3_detect import STF3Detect


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def fmt(n: int) -> str:
    if n >= 1e6:  return f'{n/1e6:.2f}M'
    if n >= 1e3:  return f'{n/1e3:.1f}K'
    return str(n)


modules = {
    'Spatial (ResNet18)': SpatialBranch(out_dim=256, pretrained=False),
    'Temporal (SmallCNN)': TemporalBranch(out_dim=128),
    'Frequency (SmallCNN)': FrequencyBranch(out_dim=128),
}

results = {}
total_branches = 0
for name, m in modules.items():
    n = count_params(m)
    total_branches += n
    results[name] = n
    print(f'{name:30s}  {fmt(n):>10s}  ({n:,})')

# Build full model with all branches
full = STF3Detect(mode='stf3')
full_params = count_params(full)
fusion_params = full_params - total_branches
results['Fusion MLP'] = fusion_params
results['Total STF3'] = full_params

print(f'{"Fusion MLP":30s}  {fmt(fusion_params):>10s}  ({fusion_params:,})')
print(f'{"Total STF3":30s}  {fmt(full_params):>10s}  ({full_params:,})')

# Reference baselines for comparison context
references = {
    'ResNet50': 25_557_032,
    'ViT-Base/16': 86_567_656,
    'VideoMAE-Base': 87_000_000,
    'TimeSformer-Base': 121_000_000,
}

Path('outputs/failure_analysis').mkdir(parents=True, exist_ok=True)
with open('outputs/failure_analysis/model_size.json', 'w', encoding='utf-8') as f:
    json.dump({'modules': results, 'references': references}, f, indent=2)
print('\n[write] outputs/failure_analysis/model_size.json')
