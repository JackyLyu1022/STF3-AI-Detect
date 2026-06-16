from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
    }
)


COL = {
    "ink": "#17213D",
    "muted": "#687086",
    "line": "#B9C2D6",
    "panel": "#F7F9FD",
    "blue": "#3F7EE8",
    "blue_l": "#EAF1FF",
    "teal": "#159A91",
    "teal_l": "#E8F7F4",
    "orange": "#F27A2E",
    "orange_l": "#FFF0E7",
    "red": "#D84848",
    "red_l": "#FFF1F1",
    "purple": "#6457C8",
    "purple_l": "#F0EEFF",
    "gray_l": "#F4F6FA",
}


def text(ax, x, y, s, size=8, weight="normal", color=None, ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, weight=weight, color=color or COL["ink"], ha=ha, va=va, linespacing=1.12, zorder=5)


def box(ax, x, y, w, h, s="", fc="white", ec=None, lw=1.1, size=8, weight="normal", color=None, r=0.012):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec or COL["line"],
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    if s:
        text(ax, x + w / 2, y + h / 2, s, size=size, weight=weight, color=color)
    return patch


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.2, rad=0.0, dashed=False):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=lw,
        color=color or COL["muted"],
        connectionstyle=f"arc3,rad={rad}",
        linestyle=(0, (3, 2)) if dashed else "solid",
        zorder=4,
    )
    ax.add_patch(patch)
    return patch


def frames_icon(ax, x, y, w=0.055, h=0.055):
    for i in range(4):
        xx = x + i * 0.018
        yy = y - i * 0.002
        ax.add_patch(Rectangle((xx, yy), w, h, facecolor="#1D2C54", edgecolor="white", linewidth=0.7, zorder=4))
        ax.plot([xx + 0.008, xx + 0.022, xx + 0.035, xx + 0.048], [yy + 0.016, yy + 0.035, yy + 0.026, yy + 0.044], color="#9EC1FF", lw=0.9, zorder=5)


def tokens(ax, x, y, color, n=6, scale=1.0):
    for i in range(n):
        box(ax, x + i * 0.026 * scale, y, 0.017 * scale, 0.032 * scale, fc=color, ec=color, lw=0.5, r=0.004)


def branch(ax, y, color, light, title, subtitle, stage1, stage2, token_label, mode_label):
    x0, w = 0.255, 0.515
    box(ax, x0, y, w, 0.175, fc="white", ec=color, lw=1.35, r=0.018)
    text(ax, x0 + 0.022, y + 0.145, title, size=10, weight="bold", color=color, ha="left")
    text(ax, x0 + 0.022, y + 0.122, subtitle, size=7.1, color=COL["muted"], ha="left")
    box(ax, x0 + 0.025, y + 0.040, 0.115, 0.070, stage1, fc=light, ec=color, lw=0.9, size=7.0)
    box(ax, x0 + 0.190, y + 0.040, 0.165, 0.070, stage2, fc=light, ec=color, lw=0.9, size=7.0)
    box(ax, x0 + 0.405, y + 0.040, 0.070, 0.070, token_label, fc=light, ec=color, lw=0.9, size=7.1, weight="bold")
    arrow(ax, x0 + 0.140, y + 0.075, x0 + 0.190, y + 0.075, color=color)
    arrow(ax, x0 + 0.355, y + 0.075, x0 + 0.405, y + 0.075, color=color)
    tokens(ax, x0 + 0.488, y + 0.058, color, n=5)
    text(ax, x0 + 0.545, y + 0.030, mode_label, size=6.6, color=color)
    arrow(ax, x0 + 0.475, y + 0.075, x0 + 0.488, y + 0.074, color=color, lw=1.0)


def main():
    out_dir = Path(__file__).resolve().parent
    fig = plt.figure(figsize=(15.8, 7.2), facecolor="white")
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.92])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    text(ax, 0.5, 0.965, "STF3-New architecture for AI-generated video detection", size=15, weight="bold")
    text(ax, 0.5, 0.930, "A modernized STF3 pipeline: DINOv2 spatial semantics, D3/ReStraV temporal dynamics, wavelet-spectral frequency traces, and branch-token fusion", size=8.5, color=COL["muted"])

    # Left input pipeline
    box(ax, 0.035, 0.455, 0.090, 0.180, fc="white", ec=COL["line"], lw=1.2, r=0.018)
    text(ax, 0.080, 0.600, "Input video", size=9, weight="bold")
    frames_icon(ax, 0.050, 0.505)
    text(ax, 0.080, 0.475, "RGB frames", size=7.0, color=COL["muted"])

    box(ax, 0.150, 0.455, 0.085, 0.180, "Uniform\nsampling\n\nT = 8\n112 x 112", fc=COL["gray_l"], ec=COL["line"], lw=1.2, size=8.0, weight="bold", r=0.018)
    arrow(ax, 0.125, 0.545, 0.150, 0.545)

    # Core frame
    box(ax, 0.245, 0.140, 0.550, 0.720, fc=COL["panel"], ec="#C8D1E4", lw=1.0, r=0.024)
    text(ax, 0.520, 0.830, "STF3-New feature extraction core", size=12, weight="bold")
    box(ax, 0.330, 0.775, 0.350, 0.045, "Shared frozen foundation encoder for spatial/temporal branches: DINOv2 ViT-S/14 -> z_t in R384", fc="white", ec=COL["purple"], lw=1.0, size=7.0, weight="bold", r=0.010)

    branch(
        ax,
        0.590,
        COL["blue"],
        COL["blue_l"],
        "Spatial branch",
        "semantic and local visual artifacts",
        "DINOv2\nframe tokens",
        "temporal attention\npooling + MLP",
        "F_s\n256-d",
        "R2 spatial_dino",
    )
    branch(
        ax,
        0.370,
        COL["teal"],
        COL["teal_l"],
        "Temporal branch",
        "representation dynamics, not pixel differences",
        "DINOv2\ntrajectory",
        "D3 second-order distance\n+ ReStraV 21-D geometry",
        "F_t\n256-d",
        "R3 temporal_d3_restrav",
    )
    branch(
        ax,
        0.150,
        COL["orange"],
        COL["orange_l"],
        "Frequency branch",
        "frequency traces and wavelet-band artifacts",
        "raw RGB\nframes",
        "WaveRep-style Haar DWT\n+ radial FFT statistics",
        "F_f\n256-d",
        "R1 frequency_wave",
    )
    text(ax, 0.505, 0.165, "Wavelet band replacement augmentation during training, p = 0.1", size=6.7, color=COL["orange"])

    # Input fan-out
    arrow(ax, 0.235, 0.545, 0.280, 0.665, color=COL["blue"], rad=0.18)
    arrow(ax, 0.235, 0.545, 0.280, 0.445, color=COL["teal"], rad=-0.10)
    arrow(ax, 0.235, 0.545, 0.280, 0.225, color=COL["orange"], rad=-0.18)
    arrow(ax, 0.470, 0.775, 0.300, 0.700, color=COL["purple"], rad=0.10, lw=1.0)
    arrow(ax, 0.470, 0.775, 0.300, 0.480, color=COL["purple"], rad=0.04, lw=1.0)

    # Fusion
    box(ax, 0.835, 0.410, 0.125, 0.270, fc=COL["red_l"], ec=COL["red"], lw=1.35, r=0.020)
    text(ax, 0.897, 0.652, "R7 final model", size=9, weight="bold", color=COL["red"])
    text(ax, 0.897, 0.625, "Branch-token\nTransformer", size=8.2, weight="bold", color=COL["red"])
    tokens(ax, 0.852, 0.560, COL["ink"], n=1, scale=1.35)
    tokens(ax, 0.882, 0.560, COL["blue"], n=1, scale=1.35)
    tokens(ax, 0.912, 0.560, COL["teal"], n=1, scale=1.35)
    tokens(ax, 0.942, 0.560, COL["orange"], n=1, scale=1.35)
    text(ax, 0.897, 0.532, "[CLS, F_s, F_t, F_f]", size=6.8, color=COL["muted"])
    box(ax, 0.857, 0.470, 0.080, 0.045, "2 layers\n4 heads", fc="white", ec=COL["red"], lw=0.8, size=6.8, r=0.008)
    arrow(ax, 0.897, 0.555, 0.897, 0.515, color=COL["red"], lw=1.0)

    # Branches into fusion
    arrow(ax, 0.790, 0.665, 0.835, 0.603, color=COL["blue"], lw=1.5)
    arrow(ax, 0.790, 0.445, 0.835, 0.545, color=COL["teal"], lw=1.5)
    arrow(ax, 0.790, 0.225, 0.835, 0.485, color=COL["orange"], lw=1.5, rad=0.08)

    # Concat ablation and output
    box(ax, 0.835, 0.725, 0.125, 0.090, "R6 fusion baseline\nstf3_new_concat\nGated weighted concat", fc="white", ec=COL["line"], lw=1.0, size=7.0, r=0.014)
    arrow(ax, 0.875, 0.725, 0.875, 0.680, color=COL["line"], lw=1.0, dashed=True)

    box(ax, 0.835, 0.255, 0.125, 0.080, "Prediction\nP(real), P(fake)", fc="white", ec=COL["ink"], lw=1.25, size=8.3, weight="bold", r=0.016)
    arrow(ax, 0.897, 0.410, 0.897, 0.335, color=COL["ink"], lw=1.3)

    # Experiment map
    box(ax, 0.035, 0.155, 0.200, 0.150, fc="white", ec=COL["line"], lw=1.0, r=0.016)
    text(ax, 0.135, 0.280, "Experiment map", size=8.5, weight="bold")
    text(ax, 0.135, 0.232, "R4: Spatial + Frequency\nR5: Spatial + Temporal\nR6: S + T + F with gated concat\nR7: S + T + F with Transformer", size=7.1)

    text(ax, 0.500, 0.055, "Compared with the old STF3-Lite diagram, ResNet18, pixel difference, SmallCNN and FFT-only modules are replaced by foundation embeddings, D3/ReStraV dynamics, WaveRep-style wavelet-spectral statistics and branch-token fusion.", size=7.0, color=COL["muted"])

    fig.savefig(out_dir / "stf3_new_architecture_clean.svg", bbox_inches="tight")
    fig.savefig(out_dir / "stf3_new_architecture_clean.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "stf3_new_architecture_clean.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
