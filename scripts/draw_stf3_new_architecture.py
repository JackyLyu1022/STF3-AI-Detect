from pathlib import Path
from textwrap import fill

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.4,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


OUT_DIR = Path("outputs/figures/stf3_new_architecture")
BASE = OUT_DIR / "stf3_new_architecture"


COLORS = {
    "ink": "#263238",
    "muted": "#5F6B72",
    "line": "#8A959B",
    "panel": "#F7F8F8",
    "input": "#E9ECEF",
    "spatial": "#D9E7F5",
    "spatial_edge": "#5D8DB8",
    "temporal": "#D7EEE9",
    "temporal_edge": "#4E9B8D",
    "frequency": "#F6E0C8",
    "frequency_edge": "#C67C35",
    "fusion": "#F1D4D3",
    "fusion_edge": "#B45F5B",
    "neutral": "#EEF1F2",
    "white": "#FFFFFF",
}


def add_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    facecolor,
    edgecolor,
    lw=1.1,
    fontsize=7.2,
    weight="normal",
    radius=0.09,
    color=None,
    zorder=2,
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color or COLORS["ink"],
        fontweight=weight,
        linespacing=1.16,
        zorder=zorder + 1,
    )
    return box


def add_label(ax, x, y, text, fontsize=7.2, color=None, weight="normal", ha="center", va="center"):
    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        color=color or COLORS["ink"],
        fontweight=weight,
        ha=ha,
        va=va,
        linespacing=1.18,
    )


def add_arrow(ax, start, end, color=None, lw=1.15, mutation_scale=8, rad=0.0, zorder=5):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color or COLORS["line"],
        shrinkA=2,
        shrinkB=2,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def add_mini_video(ax, x, y, w, h):
    offsets = [(0.18, -0.12), (0.09, -0.06), (0.0, 0.0)]
    shades = ["#D8DEE3", "#E5EAED", "#F3F5F6"]
    for (dx, dy), shade in zip(offsets, shades):
        ax.add_patch(
            Rectangle(
                (x + dx, y + dy),
                w,
                h,
                facecolor=shade,
                edgecolor="#7F8B92",
                linewidth=0.8,
                zorder=2,
            )
        )
    ax.plot([x + 0.12, x + w - 0.12], [y + 0.18, y + h - 0.18], color="#A9B4BA", lw=0.8, zorder=3)
    ax.plot([x + 0.12, x + w - 0.12], [y + h - 0.18, y + 0.18], color="#A9B4BA", lw=0.8, zorder=3)


def add_token(ax, x, y, label, edge="#7F8B92", face="#FFFFFF", w=0.62, h=0.34):
    add_box(ax, x, y, w, h, label, face, edge, lw=0.85, fontsize=6.5, radius=0.055)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15.8, 8.3), dpi=220)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    add_label(
        ax,
        0.35,
        8.66,
        "STF3-New: Spatiotemporal-Frequency Fusion for AI-generated Video Detection",
        fontsize=12.2,
        weight="bold",
        ha="left",
    )
    add_label(
        ax,
        0.35,
        8.28,
        "Three complementary feature views are encoded as branch tokens and fused by a lightweight transformer.",
        fontsize=7.6,
        color=COLORS["muted"],
        ha="left",
    )

    # Input and uniform sampling.
    add_mini_video(ax, 0.54, 5.12, 1.18, 0.78)
    add_box(
        ax,
        0.34,
        4.28,
        1.72,
        0.58,
        "Input video $V$",
        COLORS["white"],
        "#7F8B92",
        fontsize=7.5,
        weight="bold",
        radius=0.075,
    )
    sampling = add_box(
        ax,
        2.25,
        4.36,
        1.92,
        0.82,
        "Uniform sampling\n$T=8$ frames\n$112\\times112$",
        COLORS["input"],
        "#7F8B92",
        fontsize=7.15,
        radius=0.08,
    )
    add_arrow(ax, (2.04, 4.64), (2.25, 4.76), lw=1.05)

    # Feature extraction core container.
    core = FancyBboxPatch(
        (4.48, 1.56),
        7.96,
        6.28,
        boxstyle="round,pad=0.026,rounding_size=0.12",
        linewidth=1.05,
        edgecolor="#B8C0C5",
        facecolor="#FBFBFB",
        zorder=0,
    )
    ax.add_patch(core)
    add_label(
        ax,
        4.68,
        7.55,
        "STF3-New feature extraction core",
        fontsize=8.6,
        weight="bold",
        ha="left",
    )
    add_label(
        ax,
        4.68,
        7.24,
        "Parallel branch adapters produce aligned 256-D tokens.",
        fontsize=7.05,
        color=COLORS["muted"],
        ha="left",
    )

    lanes = [
        {
            "name": "Spatial branch",
            "tag": "R2 spatial_dino",
            "y": 6.12,
            "face": COLORS["spatial"],
            "edge": COLORS["spatial_edge"],
            "input": "RGB sampled\nframes",
            "main": "Frozen DINOv2\nViT-S/14 foundation encoder",
            "features": "Per-frame embedding\n$z_t\\in\\mathbb{R}^{384}$",
            "adapter": "Temporal attention pooling\n+ MLP adapter",
            "out": "$F_s\\in\\mathbb{R}^{256}$",
        },
        {
            "name": "Temporal branch",
            "tag": "R3 temporal_d3_restrav",
            "y": 4.26,
            "face": COLORS["temporal"],
            "edge": COLORS["temporal_edge"],
            "input": "DINOv2 embedding\ntrajectory\n$z_1,\\ldots,z_T$",
            "main": "D3 second-order dynamics\nadjacent distance\n+ second difference",
            "features": "ReStraV 21-D\ntrajectory geometry",
            "adapter": "D3 + ReStraV\nMLP adapter",
            "out": "$F_t\\in\\mathbb{R}^{256}$",
        },
        {
            "name": "Frequency branch",
            "tag": "R1 frequency_wave",
            "y": 2.40,
            "face": COLORS["frequency"],
            "edge": COLORS["frequency_edge"],
            "input": "Raw RGB sampled\nframes",
            "main": "WaveRep-style Haar\nwavelet decomposition\nmulti-level band energy",
            "features": "Radial FFT\nspectral statistics",
            "adapter": "Wavelet + FFT\nMLP adapter",
            "out": "$F_f\\in\\mathbb{R}^{256}$",
            "note": "Training augmentation:\nwavelet band replacement, $p=0.1$",
        },
    ]

    for lane in lanes:
        y = lane["y"]
        # Lane background.
        ax.add_patch(
            FancyBboxPatch(
                (4.72, y - 0.50),
                7.36,
                1.28,
                boxstyle="round,pad=0.018,rounding_size=0.10",
                linewidth=0.75,
                edgecolor=lane["edge"],
                facecolor=lane["face"],
                alpha=0.32,
                zorder=0.6,
            )
        )
        add_label(ax, 4.95, y + 0.58, lane["name"], fontsize=7.7, weight="bold", ha="left")
        add_label(ax, 4.95, y + 0.36, lane["tag"], fontsize=6.3, color=COLORS["muted"], ha="left")

        lane_mid = y - 0.10
        add_box(
            ax,
            4.98,
            y - 0.40,
            1.08,
            0.56,
            lane["input"],
            COLORS["white"],
            lane["edge"],
            fontsize=6.15,
            radius=0.06,
        )
        add_box(
            ax,
            6.40,
            y - 0.43,
            1.62,
            0.64,
            lane["main"],
            COLORS["white"],
            lane["edge"],
            fontsize=6.05,
            radius=0.06,
        )
        add_box(
            ax,
            8.36,
            y - 0.43,
            1.34,
            0.64,
            lane["features"],
            COLORS["white"],
            lane["edge"],
            fontsize=6.05,
            radius=0.06,
        )
        add_box(
            ax,
            10.02,
            y - 0.43,
            1.34,
            0.64,
            lane["adapter"],
            COLORS["white"],
            lane["edge"],
            fontsize=6.05,
            radius=0.06,
        )
        add_box(
            ax,
            11.61,
            y - 0.34,
            0.42,
            0.46,
            lane["out"],
            COLORS["white"],
            lane["edge"],
            fontsize=6.15,
            radius=0.06,
        )
        add_arrow(ax, (6.06, lane_mid), (6.40, lane_mid), color=lane["edge"], lw=1.05)
        add_arrow(ax, (8.02, lane_mid), (8.36, lane_mid), color=lane["edge"], lw=1.05)
        add_arrow(ax, (9.70, lane_mid), (10.02, lane_mid), color=lane["edge"], lw=1.05)
        add_arrow(ax, (11.36, lane_mid), (11.61, lane_mid), color=lane["edge"], lw=1.05)

        if lane["name"] == "Frequency branch":
            add_label(ax, 10.69, y - 0.58, lane["note"], fontsize=6.05, color="#7B5A3B")

    # Branch routing from sampling.
    split_x = 4.42
    add_arrow(ax, (4.17, 4.77), (split_x, 4.77), lw=1.1)
    for y in [6.02, 4.16, 2.30]:
        add_arrow(ax, (split_x, 4.77), (4.98, y), lw=0.95, rad=0.08 if y > 4.77 else -0.08)

    # Indicate DINO trajectory reused by temporal branch.
    add_arrow(
        ax,
        (8.36, 5.82),
        (5.52, 4.50),
        color=COLORS["temporal_edge"],
        lw=0.9,
        mutation_scale=7,
        rad=0.18,
    )
    add_label(
        ax,
        6.63,
        5.28,
        "embedding trajectory",
        fontsize=5.95,
        color=COLORS["temporal_edge"],
        ha="center",
    )

    # Fusion module.
    fusion_panel = FancyBboxPatch(
        (12.92, 2.14),
        2.68,
        4.88,
        boxstyle="round,pad=0.026,rounding_size=0.12",
        linewidth=1.05,
        edgecolor=COLORS["fusion_edge"],
        facecolor="#FCF7F7",
        zorder=0,
    )
    ax.add_patch(fusion_panel)
    add_label(ax, 13.10, 6.74, "Feature fusion", fontsize=8.4, weight="bold", ha="left")
    add_label(ax, 13.10, 6.46, "R7 stf3_new", fontsize=6.8, color=COLORS["muted"], ha="left")

    add_box(
        ax,
        13.04,
        5.42,
        2.22,
        0.74,
        "",
        "#FFFFFF",
        COLORS["fusion_edge"],
        fontsize=6.55,
        radius=0.07,
        color=COLORS["muted"],
    )

    token_y = 5.84
    token_xs = [13.15, 13.70, 14.25, 14.80]
    for x, label, edge in zip(
        token_xs,
        ["CLS", "$F_s$", "$F_t$", "$F_f$"],
        [COLORS["fusion_edge"], COLORS["spatial_edge"], COLORS["temporal_edge"], COLORS["frequency_edge"]],
    ):
        add_token(ax, x, token_y, label, edge=edge, w=0.46 if label == "CLS" else 0.42, h=0.32)
    add_label(ax, 14.14, 5.62, "Branch-token assembly", fontsize=6.5, color=COLORS["muted"])
    add_label(ax, 14.14, 5.47, "$[\\mathrm{CLS},F_s,F_t,F_f]$", fontsize=7.0, color=COLORS["muted"])

    transformer = add_box(
        ax,
        13.18,
        4.44,
        2.02,
        0.72,
        "Branch-token\nTransformer encoder\n2 layers, 4 heads",
        COLORS["fusion"],
        COLORS["fusion_edge"],
        fontsize=6.65,
        radius=0.075,
    )
    classifier = add_box(
        ax,
        13.32,
        3.38,
        1.74,
        0.58,
        "CLS token\nLinear classifier",
        COLORS["white"],
        COLORS["fusion_edge"],
        fontsize=6.7,
        radius=0.075,
    )
    output = add_box(
        ax,
        13.39,
        2.52,
        1.58,
        0.52,
        "$P(\\mathrm{real}),\\ P(\\mathrm{fake})$",
        COLORS["white"],
        COLORS["fusion_edge"],
        fontsize=6.75,
        weight="bold",
        radius=0.075,
    )
    add_arrow(ax, (14.18, 5.42), (14.18, 5.16), color=COLORS["fusion_edge"], lw=1.05)
    add_arrow(ax, (14.18, 4.44), (14.18, 3.96), color=COLORS["fusion_edge"], lw=1.05)
    add_arrow(ax, (14.18, 3.38), (14.18, 3.04), color=COLORS["fusion_edge"], lw=1.05)

    # Route feature outputs into a left-side token assembly bus so the fusion
    # internals remain readable.
    bus_x = 13.02
    ax.plot([bus_x, bus_x], [2.30, 5.78], color="#B6C0C5", lw=0.95, zorder=2)
    for src_y, edge in [
        (6.02, COLORS["spatial_edge"]),
        (4.16, COLORS["temporal_edge"]),
        (2.30, COLORS["frequency_edge"]),
    ]:
        add_arrow(ax, (12.03, src_y), (bus_x, src_y), color=edge, lw=1.05, rad=0.0)
    add_arrow(ax, (bus_x, 5.78), (13.05, token_y + 0.15), color=COLORS["fusion_edge"], lw=0.9)

    # Baseline comparison box.
    add_box(
        ax,
        12.88,
        1.38,
        2.66,
        0.46,
        "R6 stf3_new_concat: gated weighted concat fusion baseline",
        "#FFFFFF",
        "#B8A7A6",
        fontsize=6.05,
        radius=0.065,
        color=COLORS["muted"],
    )

    # Ablation mapping.
    mapping = FancyBboxPatch(
        (4.62, 0.32),
        10.72,
        0.84,
        boxstyle="round,pad=0.020,rounding_size=0.09",
        linewidth=0.85,
        edgecolor="#C2C8CC",
        facecolor="#FAFAFA",
    )
    ax.add_patch(mapping)
    add_label(ax, 4.86, 0.87, "Experiment mapping", fontsize=7.5, weight="bold", ha="left")
    map_text = (
        "R4 spatial_frequency_new = Spatial + Frequency     "
        "R5 spatial_temporal_new = Spatial + Temporal\n"
        "R6 stf3_new_concat = Spatial + Temporal + Frequency with gated concat     "
        "R7 stf3_new = Spatial + Temporal + Frequency with branch-token Transformer"
    )
    add_label(ax, 4.86, 0.53, map_text, fontsize=6.45, color=COLORS["muted"], ha="left")

    # Small method-source cue without adding obsolete baselines.
    add_label(
        ax,
        0.36,
        0.52,
        "Method sources: foundation spatial semantics,\nembedding-distance dynamics,\ntrajectory geometry,\nand wavelet/spectral artifacts.",
        fontsize=6.6,
        color=COLORS["muted"],
        ha="left",
    )

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.08}
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
