"""Generate 3 deep-dive figures for opening report PPT:
1. Per-generator difficulty (easy/mid/hard tiers)
2. Failure case visualization (FN frames + spectrum)
3. Model size comparison (Lite justification)
"""
from __future__ import annotations
import json, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import cv2
import torch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path('.')
OUT  = Path('docs/ppt_assets')
OUT.mkdir(parents=True, exist_ok=True)
DATA_ROOT = ROOT / 'data' / 'GenVideo-Val'

# Color palette aligned with PPT
C_PRIMARY = '#1E3A5F'
C_RED     = '#EE0000'
C_BLUE    = '#2D9CDB'
C_CYAN    = '#00A6A6'
C_ORANGE  = '#F2994A'
C_GREEN   = '#27AE60'
C_GRAY    = '#607080'

# ============================================================
# Figure 1: per-generator difficulty tiers
# ============================================================
def fig_per_generator_tiers():
    pg = json.load(open('outputs/failure_analysis/per_generator.json', encoding='utf-8'))
    fake_only = [s for s in pg['gen_stats'] if s['fake_total'] > 0]
    fake_only.sort(key=lambda x: -x['recall'])

    fig, ax = plt.subplots(figsize=(13.5, 6.0), dpi=150)
    fig.patch.set_facecolor('white')

    names = [s['gen'] for s in fake_only]
    recalls = [s['recall'] for s in fake_only]
    counts  = [s['fake_total'] for s in fake_only]

    def tier_color(r):
        if r >= 0.85: return C_GREEN
        if r >= 0.50: return C_ORANGE
        return C_RED

    bar_colors = [tier_color(r) for r in recalls]
    x = np.arange(len(names))
    bars = ax.bar(x, recalls, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.72)

    for i, (b, r, n) in enumerate(zip(bars, recalls, counts)):
        ax.text(b.get_x() + b.get_width()/2, r + 0.018,
                f'{r:.3f}', ha='center', va='bottom', fontsize=10.5,
                color=C_PRIMARY, fontweight='bold')
        ax.text(b.get_x() + b.get_width()/2, -0.05,
                f'n={n}', ha='center', va='top', fontsize=9, color=C_GRAY)

    ax.axhline(0.85, color=C_GREEN, ls='--', lw=1.0, alpha=0.55)
    ax.axhline(0.50, color=C_ORANGE, ls='--', lw=1.0, alpha=0.55)
    ax.text(len(names) - 0.4, 0.86, 'EASY ≥ 0.85',  color=C_GREEN,  fontsize=9.5, ha='right')
    ax.text(len(names) - 0.4, 0.51, 'MID 0.50–0.85', color=C_ORANGE, fontsize=9.5, ha='right')
    ax.text(len(names) - 0.4, 0.06, 'HARD < 0.50',  color=C_RED,    fontsize=9.5, ha='right')

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=18, ha='right', fontsize=10.5)
    ax.set_ylabel('Fake Recall (per generator)', fontsize=11.5, color=C_PRIMARY)
    ax.set_ylim(-0.10, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title('Medium-Random STF³ — Per-Generator Difficulty (recall = correctly detected as AI / total)',
                 fontsize=13, color=C_PRIMARY, fontweight='bold', pad=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(C_GRAY); ax.spines['bottom'].set_color(C_GRAY)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    ax.tick_params(colors=C_GRAY)

    plt.tight_layout()
    out = OUT / 'final_per_generator_difficulty.png'
    plt.savefig(out, dpi=160, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'[write] {out}')


# ============================================================
# Figure 2: failure case visualization (FN frames + spectrum)
# ============================================================
def sample_frames(video_path: Path, n: int = 4) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return []
    idxs = np.linspace(0, max(total - 1, 0), n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frm = cap.read()
        if ok:
            frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            frames.append(frm)
    cap.release()
    return frames


def compute_log_spectrum(frame_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    fft = np.fft.fft2(gray)
    mag = np.log1p(np.abs(fft))
    mag = np.fft.fftshift(mag)
    # standardize for visualization
    m, s = mag.mean(), mag.std() + 1e-6
    return (mag - m) / s


def fig_failure_cases():
    cases = json.load(open('outputs/failure_analysis/cases.json', encoding='utf-8'))
    fn_picks = cases['fn_cases'][:3]   # 3 FN
    fp_picks = cases['fp_cases'][:1]   # 1 FP (only 1 exists)
    picks = [('FN', c) for c in fn_picks] + [('FP', c) for c in fp_picks]

    rows = len(picks)
    fig = plt.figure(figsize=(14, 2.5 * rows + 0.6), dpi=150)
    fig.patch.set_facecolor('white')

    gs = fig.add_gridspec(rows, 6, hspace=0.45, wspace=0.18,
                          left=0.04, right=0.98, top=0.92, bottom=0.04)

    fig.suptitle('Medium-Random STF³ — Failure Case Visualization (sampled frames + FFT log-magnitude)',
                 fontsize=13, color=C_PRIMARY, fontweight='bold', y=0.985)

    for r, (kind, case) in enumerate(picks):
        video_path = DATA_ROOT / case['path']
        frames = sample_frames(video_path, n=4)
        if not frames:
            for c in range(6):
                ax = fig.add_subplot(gs[r, c])
                ax.text(0.5, 0.5, '[video unreadable]', ha='center', va='center', fontsize=9)
                ax.axis('off')
            continue

        # 4 frame thumbnails
        for c in range(4):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(frames[c])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_color(C_GRAY); spine.set_linewidth(0.5)
            if c == 0:
                tag_color = C_RED if kind == 'FN' else C_ORANGE
                tag_text  = f'{kind}: {case["gen"]}'
                ax.set_title(tag_text, fontsize=10.5, color=tag_color, fontweight='bold', loc='left', pad=4)

        # spectrum of first frame
        ax = fig.add_subplot(gs[r, 4])
        spec = compute_log_spectrum(frames[0])
        ax.imshow(spec, cmap='magma', vmin=-2, vmax=4)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_color(C_GRAY); spine.set_linewidth(0.5)
        ax.set_title('FFT log-mag', fontsize=9, color=C_PRIMARY, pad=3)

        # text panel
        ax = fig.add_subplot(gs[r, 5])
        ax.axis('off')
        if kind == 'FN':
            txt = (f'label: AI-generated (1)\n'
                   f'pred:  Real (0)  [WRONG]\n'
                   f'fake_prob: {case["prob"]:.3f}\n'
                   f'\n'
                   f'被漏检：模型给出极低伪造概率\n'
                   f'诊断方向：补强 {case["gen"]}\n'
                   f'类生成器的时序/频域特征。')
        else:
            txt = (f'label: Real (0)\n'
                   f'pred:  AI-generated (1)  [WRONG]\n'
                   f'fake_prob: {case["prob"]:.3f}\n'
                   f'\n'
                   f'误判真实视频为 AI：\n'
                   f'real_MSRVTT 中含压缩/抖动\n'
                   f'触发频域伪影类分支响应。')
        ax.text(0.0, 0.95, txt, ha='left', va='top', fontsize=8.8,
                color=C_PRIMARY,
                bbox=dict(facecolor='#F8FBFF', edgecolor=C_GRAY, boxstyle='round,pad=0.35'))

    out = OUT / 'final_failure_cases.png'
    plt.savefig(out, dpi=160, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'[write] {out}')


# ============================================================
# Figure 3: model size comparison (Lite justification)
# ============================================================
def fig_model_size():
    sz = json.load(open('outputs/failure_analysis/model_size.json', encoding='utf-8'))
    branches = {
        'Spatial\n(ResNet18)':  sz['modules']['Spatial (ResNet18)'],
        'Temporal\n(SmallCNN)': sz['modules']['Temporal (SmallCNN)'],
        'Frequency\n(SmallCNN)':sz['modules']['Frequency (SmallCNN)'],
        'Fusion\n(MLP)':        sz['modules']['Fusion MLP'],
    }
    refs = sz['references']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=150,
                              gridspec_kw={'width_ratios': [1, 1.15], 'wspace': 0.32})
    fig.patch.set_facecolor('white')

    # LEFT: STF3 internal breakdown — donut
    ax = axes[0]
    sizes = list(branches.values())
    labels = list(branches.keys())
    colors = [C_BLUE, C_CYAN, C_ORANGE, C_PRIMARY]
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.38, edgecolor='white', linewidth=2))
    total = sum(sizes)
    ax.text(0, 0.10, f'{total/1e6:.2f}M', ha='center', va='center',
            fontsize=22, color=C_PRIMARY, fontweight='bold')
    ax.text(0, -0.18, 'STF³-Lite\ntrainable params', ha='center', va='center',
            fontsize=10.5, color=C_GRAY)
    # external labels with percent
    for w, lab, sz_v in zip(wedges, labels, sizes):
        ang = (w.theta2 + w.theta1) / 2
        x = 1.18 * np.cos(np.deg2rad(ang)); y = 1.18 * np.sin(np.deg2rad(ang))
        pct = sz_v / total * 100
        ax.text(x, y, f'{lab}\n{sz_v/1e6:.2f}M ({pct:.1f}%)' if sz_v >= 1e6 else f'{lab}\n{sz_v/1e3:.1f}K ({pct:.1f}%)',
                ha='center', va='center', fontsize=9.5, color=C_PRIMARY)
    ax.set_title('STF³-Detect-Lite 内部参数分解', fontsize=12.5,
                 color=C_PRIMARY, fontweight='bold', pad=10)

    # RIGHT: comparison with reference video models
    ax = axes[1]
    cmp_data = [('STF³-Lite (ours)', total, C_RED)]
    for name, n in refs.items():
        cmp_data.append((name, n, C_GRAY))
    cmp_data.sort(key=lambda x: x[1])

    names = [x[0] for x in cmp_data]
    vals  = [x[1] / 1e6 for x in cmp_data]
    cols  = [x[2] for x in cmp_data]

    y = np.arange(len(names))
    bars = ax.barh(y, vals, color=cols, edgecolor='white', linewidth=1.5, height=0.65)
    for b, v in zip(bars, vals):
        ax.text(v + 2, b.get_y() + b.get_height()/2,
                f'{v:.1f}M', va='center', fontsize=10.5, color=C_PRIMARY, fontweight='bold')

    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlabel('Parameters (Million)', fontsize=11, color=C_PRIMARY)
    ax.set_title('参数量对比：相比常见视频/视觉骨干显著轻量', fontsize=12.5,
                 color=C_PRIMARY, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(C_GRAY); ax.spines['bottom'].set_color(C_GRAY)
    ax.tick_params(colors=C_GRAY)
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    ax.set_xlim(0, max(vals) * 1.18)

    plt.tight_layout()
    out = OUT / 'final_model_size.png'
    plt.savefig(out, dpi=160, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'[write] {out}')


if __name__ == '__main__':
    fig_per_generator_tiers()
    fig_failure_cases()
    fig_model_size()
