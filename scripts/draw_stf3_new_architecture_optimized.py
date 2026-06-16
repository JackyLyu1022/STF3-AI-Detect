from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.2,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


OUT_DIR = Path("outputs/figures/stf3_new_architecture_optimized")
BASE = OUT_DIR / "stf3_new_architecture_optimized"


C = {
    "ink": "#1F2A2E",
    "muted": "#637179",
    "line": "#1F2A2E",
    "soft_line": "#87949B",
    "panel": "#F8F9F9",
    "spatial": "#E8F1FA",
    "spatial_edge": "#4F82B4",
    "temporal": "#E5F3EF",
    "temporal_edge": "#438F7F",
    "frequency": "#FAEEE0",
    "frequency_edge": "#C77A28",
    "fusion": "#F8EAEA",
    "fusion_edge": "#B7534F",
    "gray": "#EEF1F2",
    "yellow": "#FFF3C9",
    "white": "#FFFFFF",
}


def rounded(ax, x, y, w, h, text="", fc="#FFFFFF", ec="#222222", lw=1.0, fs=7.0,
            weight="normal", radius=0.08, ls="-", color=None, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.014,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight=weight,
            color=color or C["ink"],
            linespacing=1.12,
            zorder=z + 1,
        )
    return patch


def label(ax, x, y, text, fs=7.0, weight="normal", color=None, ha="center", va="center", style="normal"):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fs,
        fontweight=weight,
        color=color or C["ink"],
        linespacing=1.12,
        fontstyle=style,
    )


def arrow(ax, xy1, xy2, color=None, lw=1.15, ms=8, rad=0.0, ls="-", z=10):
    ax.add_patch(
        FancyArrowPatch(
            xy1,
            xy2,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color or C["line"],
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={rad}",
            linestyle=ls,
            zorder=z,
        )
    )


def blank_clip(ax, x, y, w=0.58, h=0.45, n=5):
    rounded(ax, x - 0.16, y - 0.22, w + 0.32, n * 0.56 + 0.28, "", "#FFFFFF", "#7D8A91", 0.9, radius=0.06)
    for i in range(n):
        yy = y + (n - 1 - i) * 0.56
        ax.add_patch(Rectangle((x, yy), w, h, facecolor="#F4F6F7", edgecolor="#8A969C", linewidth=0.75))
        ax.plot([x + 0.08, x + w - 0.08], [yy + 0.08, yy + h - 0.08], color="#B7C1C6", lw=0.65)
        ax.plot([x + 0.08, x + w - 0.08], [yy + h - 0.08, yy + 0.08], color="#B7C1C6", lw=0.65)
        if i == 2:
            label(ax, x + w / 2, yy + h / 2, "...", fs=9.0, color=C["muted"])


def frame_stack_icon(ax, x, y, edge="#7D8A91"):
    for i, dx in enumerate([0.12, 0.06, 0.0]):
        ax.add_patch(Rectangle((x + dx, y - dx), 0.46, 0.34, facecolor="#F7F8F8", edgecolor=edge, linewidth=0.65))


def conv_stack(ax, x, y, color, n=4, label_text=""):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * 0.08, y + i * 0.025), 0.25, 0.44, facecolor=color, edgecolor="#56636A", linewidth=0.55))
    if label_text:
        label(ax, x + 0.42, y - 0.08, label_text, fs=5.5, color=C["muted"])


def vector_icon(ax, x, y, color, n=6):
    for i in range(n):
        h = 0.13 + (i % 3) * 0.08
        ax.add_patch(Rectangle((x + i * 0.11, y), 0.07, h, facecolor=color, edgecolor="#56636A", linewidth=0.45))


def wavelet_icon(ax, x, y):
    colors = ["#F1D7B9", "#E9C08C", "#D9A76B", "#C88843"]
    labels = ["LL", "LH", "HL", "HH"]
    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            ax.add_patch(Rectangle((x + j * 0.24, y + (1 - i) * 0.21), 0.21, 0.18, facecolor=colors[idx], edgecolor="#7C5B37", linewidth=0.5))
            label(ax, x + j * 0.24 + 0.105, y + (1 - i) * 0.21 + 0.09, labels[idx], fs=4.1, color="#5E4228")


def fft_icon(ax, x, y):
    ax.add_patch(Circle((x + 0.25, y + 0.22), 0.22, facecolor="#FFFFFF", edgecolor=C["frequency_edge"], linewidth=0.7))
    for r in [0.07, 0.14]:
        ax.add_patch(Circle((x + 0.25, y + 0.22), r, facecolor="none", edgecolor="#D9A76B", linewidth=0.55))
    ax.plot([x + 0.03, x + 0.47], [y + 0.22, y + 0.22], color="#D9A76B", lw=0.5)
    ax.plot([x + 0.25, x + 0.25], [y + 0.00, y + 0.44], color="#D9A76B", lw=0.5)


def token(ax, x, y, text, ec, fc="#FFFFFF", w=0.42):
    rounded(ax, x, y, w, 0.32, text, fc, ec, 0.9, fs=6.5, radius=0.05)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16.2, 8.0), dpi=220)
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    label(ax, 0.35, 7.70, "STF3-New: Spatiotemporal-Frequency Fusion for AI-generated Video Detection",
          fs=12.1, weight="bold", ha="left")
    label(ax, 0.35, 7.36,
          "Architecture schematic: sampled video frames are encoded by spatial, temporal, and frequency branches, then fused as branch tokens.",
          fs=7.2, color=C["muted"], ha="left")

    # Input column.
    rounded(ax, 0.35, 2.05, 1.45, 4.52, "", "#FFFFFF", "#222222", 1.1, radius=0.07)
    label(ax, 0.58, 6.37, "Input clip $V$", fs=7.4, weight="bold", ha="left")
    blank_clip(ax, 0.74, 3.02, 0.58, 0.45, n=5)
    label(ax, 1.07, 2.54, "empty frame\nplaceholders", fs=5.7, color=C["muted"])

    rounded(ax, 2.10, 4.02, 1.50, 0.88, "Uniform sampling\n$T=8$, $112\\times112$", C["gray"], "#6F7C83", 1.0, fs=7.0, radius=0.07)
    arrow(ax, (1.80, 4.43), (2.10, 4.46), color=C["line"])

    # Extraction core.
    rounded(ax, 4.03, 1.23, 7.78, 5.62, "", "#FFFFFF", "#222222", 1.15, radius=0.06)
    label(ax, 4.28, 6.62, "STF3-New Feature Extraction Core", fs=8.7, weight="bold", ha="left")
    label(ax, 4.28, 6.36, "Three parallel extractors produce aligned 256-D branch features.", fs=6.6, color=C["muted"], ha="left")

    lanes = [
        ("Spatial Feature Extractor", "R2 spatial_dino", 5.35, C["spatial"], C["spatial_edge"]),
        ("Temporal Dynamics Extractor", "R3 temporal_d3_restrav", 3.72, C["temporal"], C["temporal_edge"]),
        ("Frequency Feature Extractor", "R1 frequency_wave", 2.09, C["frequency"], C["frequency_edge"]),
    ]
    for title, exp, y, fc, ec in lanes:
        rounded(ax, 4.28, y - 0.50, 7.24, 1.12, "", fc, ec, 1.0, radius=0.07)
        label(ax, 4.48, y + 0.43, title, fs=7.4, weight="bold", ha="left")
        label(ax, 4.48, y + 0.20, exp, fs=6.0, color=C["muted"], ha="left")

    # Branch internals.
    y = 5.35
    cy = y - 0.12
    rounded(ax, 4.95, y - 0.38, 0.64, 0.44, "RGB\nframes", "#FFFFFF", C["spatial_edge"], 0.9, fs=5.4, radius=0.05)
    frame_stack_icon(ax, 5.04, y - 0.27, C["spatial_edge"])
    rounded(ax, 5.90, y - 0.42, 1.28, 0.52, "Frozen DINOv2\nViT-S/14", "#FFFFFF", C["spatial_edge"], 1.0, fs=6.0, radius=0.05)
    conv_stack(ax, 6.46, y - 0.32, "#B8D3EA", n=4)
    rounded(ax, 7.50, y - 0.40, 1.10, 0.48, "$z_t\\in\\mathbb{R}^{384}$\nembedding", "#FFFFFF", C["spatial_edge"], 0.9, fs=5.8, radius=0.05)
    rounded(ax, 8.88, y - 0.42, 1.28, 0.52, "Temporal attention\npooling", "#FFFFFF", C["spatial_edge"], 0.9, fs=5.9, radius=0.05)
    rounded(ax, 10.46, y - 0.38, 0.70, 0.44, "MLP\nadapter", "#FFFFFF", C["spatial_edge"], 0.9, fs=5.5, radius=0.05)
    rounded(ax, 11.30, y - 0.34, 0.38, 0.36, "$F_s$", "#FFFFFF", C["spatial_edge"], 0.95, fs=6.8, radius=0.05)
    for a, b in [((5.59, cy), (5.90, cy)), ((7.18, cy), (7.50, cy)),
                 ((8.60, cy), (8.88, cy)), ((10.16, cy), (10.46, cy)),
                 ((11.16, cy), (11.30, cy))]:
        arrow(ax, a, b, color=C["spatial_edge"])

    y = 3.72
    cy = y - 0.12
    rounded(ax, 4.95, y - 0.38, 0.94, 0.44, "$z_1,\\ldots,z_T$\ntrajectory", "#FFFFFF", C["temporal_edge"], 0.9, fs=5.7, radius=0.05)
    vector_icon(ax, 5.10, y - 0.30, "#C6E4DC")
    rounded(ax, 6.22, y - 0.42, 1.46, 0.52, "D3 second-order\nembedding-distance\ndynamics", "#FFFFFF", C["temporal_edge"], 0.9, fs=5.6, radius=0.05)
    label(ax, 6.95, y - 0.47, "$\\|z_t-z_{t-1}\\|$, $\\Delta^2z_t$", fs=5.3, color=C["muted"])
    rounded(ax, 8.06, y - 0.42, 1.36, 0.52, "ReStraV 21-D\ntrajectory geometry", "#FFFFFF", C["temporal_edge"], 0.9, fs=5.8, radius=0.05)
    vector_icon(ax, 8.48, y - 0.32, "#C6E4DC", n=7)
    rounded(ax, 9.78, y - 0.42, 1.20, 0.52, "D3 + ReStraV\nMLP adapter", "#FFFFFF", C["temporal_edge"], 0.9, fs=5.8, radius=0.05)
    rounded(ax, 11.30, y - 0.34, 0.38, 0.36, "$F_t$", "#FFFFFF", C["temporal_edge"], 0.95, fs=6.8, radius=0.05)
    for a, b in [((5.89, cy), (6.22, cy)), ((7.68, cy), (8.06, cy)),
                 ((9.42, cy), (9.78, cy)), ((10.98, cy), (11.30, cy))]:
        arrow(ax, a, b, color=C["temporal_edge"])

    y = 2.09
    cy = y - 0.12
    rounded(ax, 4.95, y - 0.38, 0.64, 0.44, "Raw RGB\nframes", "#FFFFFF", C["frequency_edge"], 0.9, fs=5.4, radius=0.05)
    frame_stack_icon(ax, 5.04, y - 0.27, C["frequency_edge"])
    rounded(ax, 5.90, y - 0.42, 1.36, 0.52, "WaveRep Haar\nwavelet bands", "#FFFFFF", C["frequency_edge"], 0.9, fs=5.8, radius=0.05)
    wavelet_icon(ax, 6.88, y - 0.35)
    rounded(ax, 7.56, y - 0.42, 1.08, 0.52, "Radial FFT\nspectral stats", "#FFFFFF", C["frequency_edge"], 0.9, fs=5.8, radius=0.05)
    fft_icon(ax, 8.08, y - 0.34)
    rounded(ax, 8.96, y - 0.42, 1.00, 0.52, "Band-energy\nstatistics", "#FFFFFF", C["frequency_edge"], 0.9, fs=5.8, radius=0.05)
    rounded(ax, 10.28, y - 0.42, 0.90, 0.52, "MLP\nadapter", "#FFFFFF", C["frequency_edge"], 0.9, fs=5.8, radius=0.05)
    rounded(ax, 11.30, y - 0.34, 0.38, 0.36, "$F_f$", "#FFFFFF", C["frequency_edge"], 0.95, fs=6.8, radius=0.05)
    rounded(
        ax,
        9.35,
        y + 0.22,
        2.02,
        0.24,
        "Aug: wavelet band replacement, $p=0.1$",
        "#FFF7E8",
        "#D5A66D",
        0.65,
        fs=5.25,
        radius=0.04,
        color="#7E5935",
    )
    for a, b in [((5.59, cy), (5.90, cy)), ((7.26, cy), (7.56, cy)),
                 ((8.64, cy), (8.96, cy)), ((9.96, cy), (10.28, cy)),
                 ((11.18, cy), (11.30, cy))]:
        arrow(ax, a, b, color=C["frequency_edge"])

    # Shared input split and embedding reuse.
    arrow(ax, (3.60, 4.46), (4.03, 4.46), color=C["line"])
    ax.plot([4.03, 4.20], [4.46, 4.46], color=C["soft_line"], lw=1.1)
    for yy in [5.23, 3.60, 1.93]:
        arrow(ax, (4.20, 4.46), (4.48, yy), color=C["soft_line"], lw=1.0)
    arrow(ax, (7.50, 5.04), (5.42, 3.88), color=C["temporal_edge"], lw=0.9, rad=0.18, ls="--", ms=7)
    label(ax, 6.34, 4.57, "DINOv2 trajectory reused", fs=5.8, color=C["temporal_edge"])

    # Fusion module.
    rounded(ax, 12.28, 1.60, 3.42, 5.10, "", "#FFFFFF", C["fusion_edge"], 1.35, radius=0.08)
    label(ax, 12.52, 6.44, "Joint Branch-token Transformer Module", fs=8.0, weight="bold", ha="left")
    label(ax, 12.52, 6.18, "Feature blender for R7 stf3_new", fs=6.3, color=C["muted"], ha="left")

    rounded(ax, 12.62, 5.20, 2.70, 0.78, "", "#FFFFFF", C["fusion_edge"], 1.0, radius=0.06)
    for x, t, ec in [(12.78, "CLS", C["fusion_edge"]), (13.36, "$F_s$", C["spatial_edge"]),
                     (13.94, "$F_t$", C["temporal_edge"]), (14.52, "$F_f$", C["frequency_edge"])]:
        token(ax, x, 5.58, t, ec, w=0.44)
    label(ax, 13.98, 5.34, "$[\\mathrm{CLS},F_s,F_t,F_f]$", fs=6.8, color=C["muted"])

    rounded(ax, 12.74, 4.28, 2.46, 0.60, "1x1 projection / feature blender", C["fusion"], C["fusion_edge"], 1.0, fs=6.6, radius=0.06)
    rounded(ax, 12.74, 3.34, 2.46, 0.66, "Branch-token Transformer Encoder\n2 layers, 4 heads", "#FFFFFF", C["fusion_edge"], 1.0, fs=6.6, radius=0.06)
    rounded(ax, 12.94, 2.54, 2.06, 0.52, "CLS token + Linear classifier", "#FFFFFF", C["fusion_edge"], 1.0, fs=6.5, radius=0.06)
    rounded(ax, 13.18, 1.90, 1.58, 0.42, "$P(\\mathrm{real}),\\ P(\\mathrm{fake})$", "#FFFFFF", C["fusion_edge"], 1.0, fs=6.5, radius=0.06)
    for a, b in [((13.98, 5.20), (13.98, 4.88)), ((13.98, 4.28), (13.98, 4.00)),
                 ((13.98, 3.34), (13.98, 3.06)), ((13.98, 2.54), (13.98, 2.32))]:
        arrow(ax, a, b, color=C["fusion_edge"])

    # Token routes.
    for y, ec in [(5.23, C["spatial_edge"]), (3.60, C["temporal_edge"]), (1.93, C["frequency_edge"])]:
        arrow(ax, (11.68, y), (12.62, y), color=ec, lw=1.1)
    ax.plot([12.62, 12.62], [1.93, 5.58], color="#AEB8BD", lw=0.9)
    arrow(ax, (12.62, 5.58), (12.78, 5.72), color=C["fusion_edge"], lw=0.95, ms=7)

    # Baseline and ablation mapping.
    rounded(ax, 12.28, 0.84, 3.42, 0.48, "R6 stf3_new_concat: gated weighted concat fusion baseline",
            "#FFFFFF", "#B89E9B", 0.95, fs=6.1, color=C["muted"], radius=0.06)

    rounded(ax, 4.03, 0.26, 11.67, 0.52, "", "#FFFFFF", "#7D8A91", 0.95, radius=0.05, ls="--")
    label(ax, 4.22, 0.60, "Experiment mapping", fs=6.9, weight="bold", ha="left")
    label(
        ax,
        6.82,
        0.39,
        "R4 spatial_frequency_new = Spatial + Frequency     R5 spatial_temporal_new = Spatial + Temporal",
        fs=5.9,
        color=C["muted"],
    )
    label(
        ax,
        12.18,
        0.39,
        "R6 = gated concat     R7 = branch-token Transformer",
        fs=5.9,
        color=C["muted"],
    )

    # Data-flow legend, styled like compact method figures.
    rounded(ax, 0.35, 0.31, 2.95, 1.02, "", "#FFFFFF", "#7D8A91", 0.85, radius=0.04, ls="--")
    label(ax, 0.56, 1.08, "Legend", fs=6.8, weight="bold", ha="left")
    ax.plot([0.60, 1.10], [0.86, 0.86], color=C["line"], lw=1.1)
    arrow(ax, (1.10, 0.86), (1.30, 0.86), color=C["line"], lw=1.1, ms=7)
    label(ax, 1.50, 0.86, "main data flow", fs=5.8, color=C["muted"], ha="left")
    ax.plot([0.60, 1.30], [0.62, 0.62], color=C["temporal_edge"], lw=1.0, linestyle="--")
    label(ax, 1.50, 0.62, "embedding reuse", fs=5.8, color=C["muted"], ha="left")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.06}
        if ext in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(BASE.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)

    print(f"saved: {BASE.with_suffix('.svg')}")
    print(f"saved: {BASE.with_suffix('.pdf')}")
    print(f"saved: {BASE.with_suffix('.png')}")
    print(f"saved: {BASE.with_suffix('.tiff')}")


if __name__ == "__main__":
    main()
