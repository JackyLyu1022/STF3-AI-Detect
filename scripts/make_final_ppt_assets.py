"""Regenerate CCF-A style scientific figures with proper Unicode (no garbled chars)."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd
import numpy as np

ASSETS = Path("docs/ppt_assets")
ASSETS.mkdir(parents=True, exist_ok=True)

PRIMARY = "#1E3A5F"
SECONDARY = "#2D9CDB"
CYAN = "#00A6A6"
ORANGE = "#F2994A"
GREEN = "#27AE60"
RED = "#EE0000"
INK = "#18212F"
MUTED = "#607080"
LIGHT = "#F5F8FC"
GRID = "#DDE6F2"
WHITE = "#FFFFFF"
PB = "#EAF2FF"
PR_ = "#FFF3F3"
PG = "#F1FFF5"
PO = "#FFF7E8"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 200,
})


def save(fig, name: str):
    out = ASSETS / name
    fig.savefig(out, dpi=240, bbox_inches="tight", transparent=False, facecolor="white", pad_inches=0.18)
    plt.close(fig)
    print(f"[write] {out}")


def rounded(ax, xy, w, h, text, fc=WHITE, ec=GRID, lw=1.6, color=INK, fontsize=11,
            weight="normal", radius=1.6, sub=None, sub_color=MUTED, sub_size=None):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.18,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(box)
    cx = x + w / 2
    if sub:
        ax.text(cx, y + h * 0.62, text, ha="center", va="center",
                fontsize=fontsize, color=color, weight=weight)
        ax.text(cx, y + h * 0.30, sub, ha="center", va="center",
                fontsize=sub_size or (fontsize - 2.5), color=sub_color)
    else:
        ax.text(cx, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=color, weight=weight)


def arrow(ax, start, end, color=PRIMARY, lw=2.0, rad=0.0, ms=14):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
                        color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def setup_canvas(w=16, h=8):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    return fig, ax


# ============== ARCHITECTURE ==============
def architecture():
    fig, ax = setup_canvas(16.5, 8.6)
    # Title bar
    ax.text(0, 58.6, "STF³-Detect-Lite Architecture",
            fontsize=22, color=PRIMARY, weight="bold", va="top")
    ax.text(0, 55.2, "Spatial / Temporal / Frequency three-branch fusion for AI-generated video detection",
            fontsize=11, color=MUTED)

    # Stage 1: Input
    for i, off in enumerate([0, 0.9, 1.8]):
        rect = patches.Rectangle((1.5+off, 30+off*0.55), 11, 8.0,
                                  facecolor=PB, edgecolor=PRIMARY, linewidth=1.4, alpha=0.95)
        ax.add_patch(rect)
    ax.text(7.5, 35.5, "Input Video V\n(T frames)",
            ha="center", va="center", fontsize=12, color=PRIMARY, weight="bold")

    # Stage 2: Sampling
    rounded(ax, (17, 31.0), 12, 7.5, "Uniform\nSampling",
            fc="#F0F7FF", ec=SECONDARY, color=PRIMARY, fontsize=12, weight="bold",
            sub="T = 4 / 8 / 16\n112 × 112")
    arrow(ax, (12.6, 35), (17, 35), color=SECONDARY, lw=2.2)

    # Stage 3: Three branches
    bx = 33; bw = 28.0; bh = 7.4
    # Spatial
    rounded(ax, (bx, 44.2), bw, bh, "Spatial Branch",
            fc=WHITE, ec=SECONDARY, color=PRIMARY, fontsize=12.5, weight="bold",
            sub="ResNet18 → mean(t) → F_s (256-d)")
    # Temporal
    rounded(ax, (bx, 30.8), bw, bh, "Temporal Branch",
            fc=WHITE, ec=CYAN, color=PRIMARY, fontsize=12.5, weight="bold",
            sub="|x_t − x_{t-1}| → Small CNN → F_t (128-d)")
    # Frequency
    rounded(ax, (bx, 17.4), bw, bh, "Frequency Branch",
            fc=WHITE, ec=ORANGE, color=PRIMARY, fontsize=12.5, weight="bold",
            sub="log(1+|FFT(gray)|) → fftshift → CNN → F_f (128-d)")
    for sy, col in [(47.9, SECONDARY), (34.5, CYAN), (21.1, ORANGE)]:
        rad = 0.10 if sy > 35 else (-0.10 if sy < 35 else 0)
        arrow(ax, (29.3, 35), (bx, sy), color=col, lw=1.9, rad=rad)

    # Stage 4: Fusion (red highlight - innovation focus)
    fbx = 65; fbw = 13.5; fbh = 14.5
    rounded(ax, (fbx, 27.6), fbw, fbh, "Feature\nFusion",
            fc=PR_, ec=RED, color=RED, fontsize=14, weight="bold", lw=2.4,
            sub="concat[F_s, F_t, F_f]\n→ 512-d", sub_color=RED, sub_size=10)
    for sy, col in [(47.9, SECONDARY), (34.5, CYAN), (21.1, ORANGE)]:
        rad = -0.12 if sy > 34.5 else (0.12 if sy < 34.5 else 0)
        arrow(ax, (61, sy), (fbx, 34.8), color=col, lw=1.8, rad=rad)

    # Stage 5: MLP
    mbx = 82; mbw = 12.0; mbh = 9.0
    rounded(ax, (mbx, 30.2), mbw, mbh, "MLP Head",
            fc=WHITE, ec=PRIMARY, color=PRIMARY, fontsize=13, weight="bold",
            sub="Linear → ReLU → Dropout\n→ Linear → P(real), P(fake)")
    arrow(ax, (78.5, 34.8), (mbx, 34.8), color=RED, lw=2.3)

    # Bottom: emphasis bar (dual-language)
    ax.text(0, 9.2, "Core emphasis (open-stage focus)",
            fontsize=10.5, color=MUTED, style="italic")
    rounded(ax, (10, 3.5), 80, 5.4,
            "完整工程闭环  +  OOD 跨生成器评估协议  +  可解释诊断可视化",
            fc="#FFF8F8", ec=RED, color=RED, fontsize=15, weight="bold", lw=2.2)

    save(fig, "final_architecture_ccf_style.png")


# ============== PROTOCOL ==============
def protocol():
    fig, ax = setup_canvas(16.5, 8.4)
    ax.text(0, 58.6, "Experimental Protocol Overview",
            fontsize=21, color=PRIMARY, weight="bold", va="top")
    ax.text(0, 55.2, "Dataset → Splits → Training → Evaluation → Visualization → Demo",
            fontsize=10.8, color=MUTED)

    # Top: data preparation
    rounded(ax, (3, 41), 18, 8.5, "GenVideo-Val", fc=WHITE, ec=PRIMARY, color=PRIMARY,
            fontsize=13, weight="bold", sub="18,302 videos\nReal 10,000 / Fake 8,302")
    rounded(ax, (28, 41), 18, 8.5, "metadata.csv", fc=WHITE, ec=SECONDARY, color=PRIMARY,
            fontsize=13, weight="bold", sub="rel_path | label | generator")
    arrow(ax, (21.2, 45.2), (28, 45.2), color=SECONDARY, lw=2.1)

    # Five split boxes
    splits = [
        (3, 25.5, "Mini", "320 / 80 / 160\n2 epochs", SECONDARY),
        (22, 25.5, "OOD Mini", "holdout 4 gens\n2 epochs", RED),
        (41, 25.5, "Medium Random", "1200 / 300 / 400\n3 epochs", SECONDARY),
        (60, 25.5, "Random Split", "12,804 / 2,743 / 2,755", PRIMARY),
        (79, 25.5, "OOD Split", "holdout: Morph/Show/\nSora/Wild", RED),
    ]
    for x, y, t, sub, ec in splits:
        is_red = ec == RED
        rounded(ax, (x, y), 17, 9, t, fc=PR_ if is_red else WHITE, ec=ec,
                color=RED if is_red else PRIMARY, fontsize=12, weight="bold", sub=sub)
        arrow(ax, (37, 41), (x+8.5, y+9), color=ec, lw=1.4, rad=0.06)

    # Bottom pipeline
    rounded(ax, (4, 7.5), 17, 7.5, "Training",
            fc="#F8FBFF", ec=SECONDARY, color=PRIMARY, fontsize=12, weight="bold",
            sub="AdamW + CE Loss\nbest AUC checkpoint")
    rounded(ax, (27, 7.5), 17, 7.5, "Evaluation",
            fc="#F8FBFF", ec=SECONDARY, color=PRIMARY, fontsize=12, weight="bold",
            sub="ACC / AUC / F1\nP / R / CM")
    rounded(ax, (50, 7.5), 17, 7.5, "Visualization",
            fc="#F8FBFF", ec=SECONDARY, color=PRIMARY, fontsize=12, weight="bold",
            sub="ROC / CM / per-gen\nFFT / frame diff")
    rounded(ax, (73, 7.5), 17, 7.5, "Gradio Demo",
            fc=PR_, ec=RED, color=RED, fontsize=12, weight="bold",
            sub="Upload → Probability\nframes + spectrum")
    arrow(ax, (21, 11.2), (27, 11.2), color=PRIMARY, lw=1.9)
    arrow(ax, (44, 11.2), (50, 11.2), color=PRIMARY, lw=1.9)
    arrow(ax, (67, 11.2), (73, 11.2), color=RED, lw=2.0)

    save(fig, "final_protocol_ccf_style.png")


# ============== DEMO WORKFLOW ==============
def demo_workflow():
    fig, ax = setup_canvas(16, 7.6)
    ax.text(0, 58.6, "Demo Inference Workflow",
            fontsize=21, color=PRIMARY, weight="bold", va="top")
    steps = [
        (3, "1. Upload", "mp4 / mov", SECONDARY),
        (22, "2. Sampling", "T frames uniform", SECONDARY),
        (41, "3. STF³ Inference", "best.pt @ GPU/CPU", RED),
        (60, "4. Probability", "P(real), P(fake)", PRIMARY),
        (79, "5. Evidence", "frames + FFT", PRIMARY),
    ]
    for i, (x, t, sub, col) in enumerate(steps):
        is_red = col == RED
        rounded(ax, (x, 33), 17, 11, t, fc=PR_ if is_red else WHITE, ec=col,
                color=RED if is_red else PRIMARY, fontsize=12.5, weight="bold", sub=sub)
        if i < len(steps) - 1:
            nx = steps[i+1][0]
            arrow(ax, (x+17, 38.5), (nx, 38.5), color=RED if i == 2 else PRIMARY, lw=2.2)

    # Mock UI
    rounded(ax, (10, 7), 80, 19, "Gradio Concept UI", fc=WHITE, ec=GRID,
            color=PRIMARY, fontsize=12, weight="bold")
    ax.add_patch(patches.Rectangle((14, 11), 22, 11, facecolor=PB, edgecolor=GRID, linewidth=1.2))
    ax.text(25, 16.5, "Video preview", ha="center", va="center", color=MUTED, fontsize=11)
    ax.add_patch(patches.Rectangle((40, 11), 20, 11, facecolor=PG, edgecolor=GRID, linewidth=1.2))
    ax.text(50, 17.6, "AI prob: 0.992", ha="center", va="center", color=GREEN,
            fontsize=12, weight="bold")
    ax.text(50, 14.4, "Real prob: 0.008", ha="center", va="center", color=GREEN, fontsize=11)
    ax.add_patch(patches.Rectangle((64, 11), 22, 11, facecolor=PO, edgecolor=GRID, linewidth=1.2))
    ax.text(75, 17.5, "Sample frames", ha="center", va="center", color=MUTED, fontsize=11)
    ax.text(75, 14.5, "FFT spectrum", ha="center", va="center", color=MUTED, fontsize=11)

    save(fig, "final_demo_workflow.png")


# ============== DATA DISTRIBUTION ==============
def dataset_distribution():
    meta = pd.read_csv("data/GenVideo-Val/metadata.csv")
    counts = meta.groupby(["generator", "label_name"]).size().reset_index(name="count")
    order = counts.sort_values("count", ascending=True)
    gens = order["generator"].tolist()
    vals = order["count"].tolist()
    holdout = {"MorphStudio", "Show_1", "Sora", "WildScrape"}
    colors = [RED if g in holdout else (PRIMARY if g == "real_MSRVTT" else SECONDARY) for g in gens]

    fig, ax = plt.subplots(figsize=(12.6, 7.1))
    bars = ax.barh(gens, vals, color=colors, alpha=0.92)
    ax.set_xlabel("Number of videos", color=MUTED, fontsize=11)
    ax.set_title("GenVideo-Val Generator Distribution",
                 loc="left", color=PRIMARY, fontsize=18, weight="bold", pad=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='y', labelsize=10, colors=INK)
    ax.tick_params(axis='x', labelsize=9, colors=MUTED)
    for y, v in enumerate(vals):
        ax.text(v + max(vals)*0.01, y, f"{v:,}", va="center", fontsize=9, color=INK)

    legend_elems = [
        patches.Patch(color=PRIMARY, label="real_MSRVTT (real)"),
        patches.Patch(color=SECONDARY, label="seen fake generators"),
        patches.Patch(color=RED, label="OOD holdout fake generators"),
    ]
    ax.legend(handles=legend_elems, frameon=False, fontsize=9.5, loc="lower right")
    save(fig, "final_dataset_distribution.png")


# ============== RESULTS SUMMARY ==============
def results_summary():
    df = pd.read_csv("outputs/opening_key_results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.4),
                              gridspec_kw={"width_ratios": [1.45, 0.85, 0.85]})
    fig.suptitle("Opening-stage quantitative evidence",
                 x=0.02, y=1.02, ha="left", fontsize=20, color=PRIMARY, weight="bold")

    # mini ablation
    mini = df[df.group == "mini_ablation"].copy()
    name_map = {"spatial": "Spatial", "frequency": "Frequency", "temporal": "Temporal",
                "spatial_frequency": "S+F", "spatial_temporal": "S+T", "stf3": "STF³"}
    mini["name"] = mini["model"].map(name_map)
    colors = [SECONDARY, SECONDARY, CYAN, ORANGE, ORANGE, RED]
    axes[0].bar(mini["name"], mini["auc"], color=colors, alpha=0.9)
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_title("Mini ablation (AUC)", color=PRIMARY, weight="bold")
    axes[0].grid(axis="y", color=GRID)
    axes[0].tick_params(axis='x', rotation=22)
    for i, v in enumerate(mini["auc"]):
        axes[0].text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)

    # OOD mini
    ood = df[df.group == "ood_mini"].copy()
    ood["name"] = ood["model"].map({"spatial": "Spatial", "stf3": "STF³"})
    x = np.arange(len(ood))
    width = 0.34
    axes[1].bar(x - width/2, ood["acc"], width, label="ACC", color=SECONDARY)
    axes[1].bar(x + width/2, ood["auc"], width, label="AUC", color=RED)
    axes[1].set_xticks(x); axes[1].set_xticklabels(ood["name"])
    axes[1].set_ylim(0.55, 0.9)
    axes[1].set_title("OOD mini", color=PRIMARY, weight="bold")
    axes[1].legend(frameon=False, fontsize=8); axes[1].grid(axis="y", color=GRID)
    for j, metric in enumerate(["acc", "auc"]):
        for i, v in enumerate(ood[metric]):
            axes[1].text(i + (-width/2 if j == 0 else width/2), v + 0.008,
                          f"{v:.3f}", ha="center", fontsize=8)

    # medium random
    med = df[df.group == "medium_random"].copy()
    med["name"] = med["model"].map({"spatial": "Spatial", "stf3": "STF³"})
    x = np.arange(len(med))
    axes[2].bar(x - width/2, med["acc"], width, label="ACC", color=SECONDARY)
    axes[2].bar(x + width/2, med["auc"], width, label="AUC", color=RED)
    axes[2].set_xticks(x); axes[2].set_xticklabels(med["name"])
    axes[2].set_ylim(0.68, 1.0)
    axes[2].set_title("Medium random", color=PRIMARY, weight="bold")
    axes[2].legend(frameon=False, fontsize=8); axes[2].grid(axis="y", color=GRID)
    for j, metric in enumerate(["acc", "auc"]):
        for i, v in enumerate(med[metric]):
            axes[2].text(i + (-width/2 if j == 0 else width/2), v + 0.008,
                          f"{v:.3f}", ha="center", fontsize=8)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=MUTED)
    save(fig, "final_results_summary.png")


# ============== ABLATION DETAILED ==============
def ablation_detail():
    """Detailed ablation: ACC + AUC + F1 stacked."""
    fig, ax = plt.subplots(figsize=(13, 5.6))
    rows = [
        ("Spatial",       0.5563, 0.6641, 0.6537),
        ("Frequency",     0.5500, 0.6547, 0.6400),
        ("Temporal",      0.8187, 0.8671, 0.7972),
        ("S + F",         0.5938, 0.6832, 0.3810),
        ("S + T",         0.6937, 0.8938, 0.5586),
        ("STF³-Lite",0.7625, 0.8764, 0.7164),
    ]
    names = [r[0] for r in rows]
    accs = [r[1] for r in rows]
    aucs = [r[2] for r in rows]
    f1s  = [r[3] for r in rows]
    x = np.arange(len(names))
    w = 0.27
    bars1 = ax.bar(x - w, accs, w, label="ACC", color=SECONDARY)
    bars2 = ax.bar(x,     aucs, w, label="AUC", color=RED)
    bars3 = ax.bar(x + w, f1s,  w, label="F1",  color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylim(0.30, 1.05)
    ax.set_ylabel("Score", color=MUTED)
    ax.set_title("Mini ablation: ACC / AUC / F1 across single, dual, and triple-branch models",
                 color=PRIMARY, weight="bold", fontsize=14, pad=10)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    ax.grid(axis="y", color=GRID)
    for bars in [bars1, bars2, bars3]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
                     f"{b.get_height():.3f}", ha="center", fontsize=8, color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED)
    save(fig, "final_ablation_detail.png")


# ============== MEDIUM COMPARE ==============
def medium_compare():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    names = ["Spatial", "STF³-Lite"]
    metrics = {
        "ACC":       [0.7400, 0.8475],
        "AUC":       [0.8350, 0.9381],
        "F1":        [0.6994, 0.8211],
        "Precision": [0.8288, 0.9929],
        "Recall":    [0.6050, 0.7000],
    }
    x = np.arange(len(metrics))
    w = 0.36
    spat = [v[0] for v in metrics.values()]
    stf3 = [v[1] for v in metrics.values()]
    bars1 = ax.bar(x - w/2, spat, w, label="Spatial baseline", color=SECONDARY)
    bars2 = ax.bar(x + w/2, stf3, w, label="STF³-Lite (ours)", color=RED)
    ax.set_xticks(x); ax.set_xticklabels(metrics.keys(), fontsize=11)
    ax.set_ylim(0.55, 1.05)
    ax.set_title("Medium Random: STF³-Lite vs Spatial baseline",
                 color=PRIMARY, weight="bold", fontsize=15, pad=10)
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    ax.grid(axis="y", color=GRID)
    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
                     f"{b.get_height():.3f}", ha="center", fontsize=9, color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED)
    save(fig, "final_medium_compare.png")


# ============== TIMELINE ==============
def timeline():
    fig, ax = plt.subplots(figsize=(15, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50)
    ax.axis("off")
    ax.text(0, 47, "Project Timeline",
            fontsize=18, color=PRIMARY, weight="bold", va="top")

    stages = [
        (4, "Phase 1\nEnv & Data", "GPU ok / metadata\nsplits CSV ready", GREEN),
        (24, "Phase 2\nModels & Training", "Spatial / Temporal /\nFrequency / STF³", GREEN),
        (44, "Phase 3 (now)\nOpening Eval", "Mini / OOD Mini /\nMedium Random", RED),
        (64, "Phase 4\nFull Experiments", "Random Split + OOD\n+ multi-seed", PRIMARY),
        (84, "Phase 5\nReport & Demo", "Final report + PPT\n+ Gradio demo", PRIMARY),
    ]
    # baseline line
    ax.plot([4, 96], [22, 22], color=GRID, linewidth=2.5, zorder=0)
    for x, t, sub, col in stages:
        is_red = col == RED
        ax.add_patch(patches.Circle((x+7, 22), 1.6, color=col, zorder=2))
        rounded(ax, (x, 28), 14, 16, t, fc=PR_ if is_red else WHITE, ec=col,
                color=col, fontsize=11, weight="bold", sub=sub, sub_color=MUTED)
        rounded(ax, (x+1, 4), 12, 12, "DONE" if col == GREEN else ("OPEN" if is_red else "TODO"),
                fc=("#F2FFF6" if col == GREEN else (PR_ if is_red else PB)), ec=col,
                color=col, fontsize=11, weight="bold")
    save(fig, "final_timeline.png")


if __name__ == "__main__":
    architecture()
    protocol()
    demo_workflow()
    dataset_distribution()
    results_summary()
    ablation_detail()
    medium_compare()
    timeline()
    print("done")
