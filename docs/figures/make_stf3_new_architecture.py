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
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


COL = {
    "navy": "#17213D",
    "muted": "#5E6678",
    "line": "#AEB8CC",
    "panel": "#F7F9FC",
    "blue": "#3F7EE8",
    "blue_l": "#EAF1FF",
    "teal": "#159A91",
    "teal_l": "#E8F7F4",
    "orange": "#F27A2E",
    "orange_l": "#FFF0E7",
    "red": "#D84848",
    "red_l": "#FFF0F0",
    "purple": "#6457C8",
    "purple_l": "#F0EEFF",
    "gray_l": "#F4F5F8",
}


def box(ax, xy, wh, text, fc="white", ec="#AEB8CC", lw=1.2, r=0.035, fontsize=8, weight="normal", color=None):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.01,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color or COL["navy"],
        weight=weight,
        linespacing=1.12,
        zorder=3,
    )
    return patch


def label(ax, x, y, text, size=8, weight="normal", color=None, ha="center", va="center", rotation=0):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        weight=weight,
        color=color or COL["navy"],
        linespacing=1.12,
        rotation=rotation,
        zorder=4,
    )


def arrow(ax, start, end, color="#5E6678", lw=1.3, rad=0.0, dashed=False, ms=10):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=(0, (3, 2)) if dashed else "solid",
        zorder=3,
    )
    ax.add_patch(arr)
    return arr


def mini_frames(ax, x, y, w=0.22, h=0.13, n=4):
    for i in range(n):
        xx = x + i * w * 0.27
        rect = Rectangle((xx, y - i * 0.004), w * 0.42, h, facecolor="#1D2C54", edgecolor="white", linewidth=0.7, zorder=4)
        ax.add_patch(rect)
        ax.plot([xx + w * 0.04, xx + w * 0.17, xx + w * 0.29, xx + w * 0.38], [y + h * 0.25, y + h * 0.6, y + h * 0.42, y + h * 0.72], color="#9EC1FF", lw=0.9, zorder=5)


def token_strip(ax, x, y, color, n=6, label_text=None):
    for i in range(n):
        ax.add_patch(
            FancyBboxPatch(
                (x + i * 0.027, y),
                0.019,
                0.036,
                boxstyle="round,pad=0.002,rounding_size=0.004",
                linewidth=0.5,
                edgecolor=color,
                facecolor=color,
                alpha=0.95,
                zorder=5,
            )
        )
    if label_text:
        label(ax, x + n * 0.013, y - 0.02, label_text, size=7, color=color)


def branch_panel(ax, y, title, color, light, subtitle, left_text, mid_text, out_text, mode_text):
    box(ax, (0.215, y), (0.575, 0.205), "", fc="white", ec=color, lw=1.25, r=0.018)
    label(ax, 0.235, y + 0.178, title, size=10, weight="bold", color=color, ha="left")
    label(ax, 0.235, y + 0.151, subtitle, size=7.2, color=COL["muted"], ha="left")

    box(ax, (0.238, y + 0.047), (0.105, 0.085), left_text, fc=light, ec=color, lw=0.9, r=0.012, fontsize=7.2)
    box(ax, (0.405, y + 0.047), (0.155, 0.085), mid_text, fc=light, ec=color, lw=0.9, r=0.012, fontsize=7.0)
    box(ax, (0.622, y + 0.047), (0.103, 0.085), out_text, fc=light, ec=color, lw=0.9, r=0.012, fontsize=7.0, weight="bold")
    arrow(ax, (0.343, y + 0.089), (0.405, y + 0.089), color=color, lw=1.1, ms=8)
    arrow(ax, (0.560, y + 0.089), (0.622, y + 0.089), color=color, lw=1.1, ms=8)
    token_strip(ax, 0.735, y + 0.081, color, label_text=mode_text)
    arrow(ax, (0.725, y + 0.089), (0.735, y + 0.099), color=color, lw=1.0, ms=7)


def main():
    out_dir = Path(__file__).resolve().parent
    fig = plt.figure(figsize=(15.6, 8.2), facecolor="white")
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    label(ax, 0.5, 0.965, "STF3-New: Foundation Spatiotemporal-Frequency Fusion for AI Video Detection", size=15, weight="bold")
    label(ax, 0.5, 0.935, "Modernized branches for Random/OOD experiments: DINOv2 spatial cues, D3/ReStraV temporal dynamics, WaveRep/FFT frequency traces", size=8.5, color=COL["muted"])

    # Input and sampling
    box(ax, (0.025, 0.42), (0.095, 0.18), "Input video\nV", fc="white", ec=COL["line"], lw=1.2, r=0.02, fontsize=9, weight="bold")
    mini_frames(ax, 0.045, 0.49, w=0.10, h=0.055, n=4)
    label(ax, 0.073, 0.435, "raw RGB frames", size=7, color=COL["muted"])

    box(ax, (0.142, 0.42), (0.085, 0.18), "Uniform\nsampling\n\nT = 8\n112 x 112", fc=COL["gray_l"], ec=COL["line"], lw=1.2, r=0.02, fontsize=8.0, weight="bold")
    arrow(ax, (0.120, 0.51), (0.142, 0.51), color=COL["muted"], lw=1.4)

    # Feature extraction core
    box(ax, (0.235, 0.14), (0.585, 0.75), "", fc=COL["panel"], ec="#C8D1E4", lw=1.0, r=0.025)
    label(ax, 0.528, 0.865, "STF3-New feature extraction core", size=12, weight="bold")

    box(ax, (0.275, 0.800), (0.255, 0.055), "Shared frozen foundation encoder: DINOv2 ViT-S/14 -> frame embeddings z_t in R384", fc="white", ec=COL["purple"], lw=1.2, r=0.014, fontsize=7.3, weight="bold")
    arrow(ax, (0.227, 0.51), (0.275, 0.827), color=COL["purple"], lw=1.0, rad=0.12)
    label(ax, 0.402, 0.782, "spatial and temporal branches share the same frame embedding sequence", size=6.7, color=COL["purple"])

    branch_panel(
        ax,
        0.580,
        "Spatial branch",
        COL["blue"],
        COL["blue_l"],
        "semantic and local visual artifacts",
        "DINOv2\nframe tokens",
        "temporal attention\npooling + adapter",
        "F_s\n256-d",
        "spatial_dino",
    )
    arrow(ax, (0.402, 0.800), (0.343, 0.669), color=COL["blue"], lw=1.0, rad=0.10)

    branch_panel(
        ax,
        0.355,
        "Temporal branch",
        COL["teal"],
        COL["teal_l"],
        "representation dynamics instead of pixel differences",
        "DINOv2\ntrajectory z_1...z_T",
        "D3 second-order distance\n+ ReStraV 21-D geometry",
        "F_t\n256-d",
        "temporal_d3_restrav",
    )
    arrow(ax, (0.402, 0.800), (0.343, 0.444), color=COL["teal"], lw=1.0, rad=0.08)

    branch_panel(
        ax,
        0.130,
        "Frequency branch",
        COL["orange"],
        COL["orange_l"],
        "frequency traces and wavelet-band artifacts",
        "raw RGB\nframes",
        "WaveRep-style Haar DWT\n+ radial FFT statistics",
        "F_f\n256-d",
        "frequency_wave",
    )
    arrow(ax, (0.227, 0.51), (0.343, 0.219), color=COL["orange"], lw=1.0, rad=-0.08)
    label(ax, 0.455, 0.153, "training augmentation: wavelet band replacement, p = 0.1", size=6.7, color=COL["orange"])

    # Fusion
    box(ax, (0.845, 0.43), (0.13, 0.28), "", fc=COL["red_l"], ec=COL["red"], lw=1.4, r=0.022)
    label(ax, 0.910, 0.685, "R7 final fusion", size=10, weight="bold", color=COL["red"])
    label(ax, 0.910, 0.660, "Branch-token Transformer", size=8.5, weight="bold", color=COL["red"])
    token_strip(ax, 0.865, 0.606, COL["navy"], n=1)
    token_strip(ax, 0.900, 0.606, COL["blue"], n=1)
    token_strip(ax, 0.935, 0.606, COL["teal"], n=1)
    token_strip(ax, 0.970, 0.606, COL["orange"], n=1)
    label(ax, 0.927, 0.585, "[CLS, F_s, F_t, F_f] + branch embedding", size=6.7, color=COL["muted"])
    box(ax, (0.866, 0.515), (0.087, 0.052), "2-layer encoder\n4 heads", fc="white", ec=COL["red"], lw=0.9, r=0.010, fontsize=7.0)
    box(ax, (0.866, 0.455), (0.087, 0.042), "linear head\nP(real), P(fake)", fc="white", ec=COL["red"], lw=0.9, r=0.010, fontsize=7.0)
    arrow(ax, (0.910, 0.606), (0.910, 0.567), color=COL["red"], lw=1.0, ms=8)
    arrow(ax, (0.910, 0.515), (0.910, 0.497), color=COL["red"], lw=1.0, ms=8)

    # Branch arrows into fusion
    arrow(ax, (0.790, 0.724), (0.845, 0.625), color=COL["blue"], lw=1.5, rad=-0.10)
    arrow(ax, (0.790, 0.484), (0.845, 0.585), color=COL["teal"], lw=1.5, rad=0.05)
    arrow(ax, (0.790, 0.244), (0.845, 0.545), color=COL["orange"], lw=1.5, rad=0.15)

    # Output
    box(ax, (0.845, 0.285), (0.13, 0.075), "Prediction\nreal / fake", fc="white", ec=COL["navy"], lw=1.3, r=0.018, fontsize=9, weight="bold")
    arrow(ax, (0.910, 0.430), (0.910, 0.360), color=COL["navy"], lw=1.3, ms=10)

    # Ablation callouts
    box(ax, (0.835, 0.760), (0.145, 0.095), "Ablation fusion\nR6: stf3_new_concat\nGated weighted concat", fc="white", ec=COL["line"], lw=1.0, r=0.016, fontsize=7.4)
    arrow(ax, (0.855, 0.760), (0.872, 0.705), color=COL["line"], lw=1.0, dashed=True, ms=7)

    box(ax, (0.035, 0.145), (0.155, 0.155), "Experiment modes\n\nR3: Temporal only\nR4: Spatial + Frequency\nR5: Spatial + Temporal\nR6: Three branches + concat\nR7: Three branches + Transformer", fc="white", ec=COL["line"], lw=1.0, r=0.018, fontsize=7.0)

    # Small equations
    box(ax, (0.545, 0.806), (0.225, 0.043), "D3: second-order embedding-distance dynamics; ReStraV: 21-D trajectory geometry", fc="white", ec="#D5DCEB", lw=0.8, r=0.012, fontsize=6.7)

    # Footer
    label(ax, 0.505, 0.055, "Key change from the old STF3-Lite figure: ResNet18 / pixel difference / SmallCNN / FFT-only modules are replaced by foundation embeddings, D3-ReStraV dynamics, WaveRep-style wavelet-spectral statistics, and branch-token fusion.", size=7.2, color=COL["muted"])

    fig.savefig(out_dir / "stf3_new_architecture.svg", bbox_inches="tight")
    fig.savefig(out_dir / "stf3_new_architecture.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "stf3_new_architecture.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
