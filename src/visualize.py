from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

from src.dataset import read_video_frames
from src.models.frequency_branch import FrequencyBranch
from src.utils import ensure_dir


def plot_predictions(pred_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(pred_csv)
    y = df["label"].astype(int).to_numpy()
    p = df["fake_prob"].astype(float).to_numpy()
    pred = (p >= 0.5).astype(int)

    cm = confusion_matrix(y, pred, labels=[0, 1])
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(4, 3))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "roc_curve.png", dpi=200)
        plt.close()

    if "generator" in df.columns:
        gen = df.groupby("generator")["correct"].mean().sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        gen.plot(kind="bar")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(out_dir / "accuracy_by_generator.png", dpi=200)
        plt.close()


def plot_history(history_json: Path, out_dir: Path) -> None:
    if not history_json.exists():
        return
    data = json.loads(history_json.read_text(encoding="utf-8"))
    hist = data.get("history", [])
    if not hist:
        return
    epochs = [h["epoch"] for h in hist]
    for key in ["loss", "acc", "auc", "f1"]:
        plt.figure(figsize=(5, 3))
        plt.plot(epochs, [h["train"].get(key, np.nan) for h in hist], marker="o", label="train")
        plt.plot(epochs, [h["val"].get(key, np.nan) for h in hist], marker="o", label="val")
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"history_{key}.png", dpi=200)
        plt.close()


def plot_frequency_examples(data_root: Path, metadata_csv: Path, out_dir: Path, image_size: int = 224) -> None:
    df = pd.read_csv(metadata_csv)
    samples = []
    for label_name in ["real", "fake"]:
        sub = df[df["label_name"] == label_name]
        if len(sub):
            samples.append((label_name, data_root / sub.iloc[0]["rel_path"]))
    if len(samples) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    for row_idx, (name, path) in enumerate(samples):
        frames = read_video_frames(path, num_frames=1, image_size=image_size, random_sample=False).unsqueeze(0)
        spec = FrequencyBranch.frames_to_spectrum(frames)[0, 0, 0].numpy()
        img = frames[0, 0].permute(1, 2, 0).numpy()
        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title(f"{name} frame")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].imshow(spec, cmap="magma")
        axes[row_idx, 1].set_title(f"{name} FFT log-mag")
        axes[row_idx, 1].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "frequency_examples.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="")
    parser.add_argument("--history", default="")
    parser.add_argument("--data-root", default="data/GenVideo-Val")
    parser.add_argument("--metadata", default="data/GenVideo-Val/metadata.csv")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    if args.predictions:
        plot_predictions(Path(args.predictions), out_dir)
    if args.history:
        plot_history(Path(args.history), out_dir)
    plot_frequency_examples(Path(args.data_root), Path(args.metadata), out_dir)
    print(f"[done] figures in {out_dir}")


if __name__ == "__main__":
    main()
