from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import read_video_frames
from src.models.frequency_branch import FrequencyBranch
from src.models.model_factory import build_model
from src.metrics import logits_to_fake_prob
from src.utils import get_device


MODEL: torch.nn.Module | None = None
DEVICE = get_device()
CKPT_ARGS: dict = {}
CHECKPOINT_PATH: str | None = None


def load_model(checkpoint: str | Path) -> tuple[torch.nn.Module, dict]:
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu")
    args = ckpt.get("args", {})
    mode = args.get("model", "spatial")
    model = build_model(
        mode=mode,
        pretrained_spatial=bool(args.get("pretrained_spatial", False)),
        foundation_backbone=args.get("foundation_backbone", "dinov2_vits14"),
        freeze_foundation=not bool(args.get("finetune_foundation", False)),
        local_files_only=bool(args.get("local_files_only", False)),
        image_size=int(args.get("image_size", 224)),
        wavelet_aug_prob=0.0,
        branch_dropout=0.0,
        d3_loss=args.get("d3_loss", "l2"),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model, args


def make_frame_grid(frames: torch.Tensor, max_frames: int = 8) -> np.ndarray:
    # frames: [T,3,H,W] in [0,1]
    frames = frames[:max_frames].detach().cpu().numpy()
    frames = np.transpose(frames, (0, 2, 3, 1))
    n = len(frames)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    h, w = frames.shape[1:3]
    canvas = np.ones((rows * h, cols * w, 3), dtype=np.float32)
    for i, img in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return (canvas * 255).clip(0, 255).astype(np.uint8)


def make_frequency_figure(frames: torch.Tensor) -> str:
    out = Path(tempfile.gettempdir()) / f"stf3_freq_{time.time_ns()}.png"
    x = frames[:1].unsqueeze(0)  # [1,1,3,H,W]
    spec = FrequencyBranch.frames_to_spectrum(x)[0, 0, 0].detach().cpu().numpy()
    frame = frames[0].permute(1, 2, 0).detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(frame)
    axes[0].set_title("Sampled frame")
    axes[0].axis("off")
    axes[1].imshow(spec, cmap="magma")
    axes[1].set_title("FFT log-magnitude")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)


def predict(video_path: str | None):
    global MODEL, CKPT_ARGS
    if video_path is None:
        return "请先上传一个视频。", None, None, None
    if MODEL is None:
        return "模型还没有加载。请用 --checkpoint 指定模型权重。", None, None, None

    num_frames = int(CKPT_ARGS.get("num_frames", 16))
    image_size = int(CKPT_ARGS.get("image_size", 224))
    # Demo 为了快，最多用 16 帧；如果 checkpoint 是 smoke 的 4 帧就尊重它。
    num_frames = max(1, min(num_frames, 16))

    try:
        frames = read_video_frames(video_path, num_frames=num_frames, image_size=image_size, random_sample=False)
        batch = frames.unsqueeze(0).to(DEVICE)
        start = time.time()
        with torch.no_grad():
            out = MODEL(batch, return_dict=True) if hasattr(MODEL, "branch_names") else MODEL(batch)
            logits = out["logits"] if isinstance(out, dict) else out
            fake_prob = float(logits_to_fake_prob(logits).item())
        elapsed = time.time() - start
        real_prob = 1.0 - fake_prob
        label = "AI-generated / 疑似 AI 生成" if fake_prob >= 0.5 else "Real / 疑似真实视频"
        result = (
            f"## 检测结果：{label}\n\n"
            f"- AI 生成概率：**{fake_prob:.4f}**\n"
            f"- 真实视频概率：**{real_prob:.4f}**\n"
            f"- 模型模式：`{CKPT_ARGS.get('model', 'unknown')}`\n"
            f"- 采样帧数：`{num_frames}`\n"
            f"- 推理耗时：`{elapsed:.3f}s`\n\n"
            f"> 注意：本 Demo 是课程项目模型，结论仅用于实验展示，不作为真实内容审核的最终依据。"
        )
        grid = make_frame_grid(frames, max_frames=min(8, num_frames))
        freq_path = make_frequency_figure(frames)
        probs = {"Real": real_prob, "AI-generated": fake_prob}
        return result, probs, grid, freq_path
    except Exception as e:
        return f"处理失败：{type(e).__name__}: {e}", None, None, None


def build_app() -> gr.Blocks:
    with gr.Blocks(title="STF³-Detect Demo") as demo:
        gr.Markdown(
            "# STF³-Detect-Lite：AI 生成视频检测 Demo\n"
            "上传一段视频，系统会均匀采样若干帧，并输出 Real / AI-generated 概率。"
        )
        with gr.Row():
            video = gr.Video(label="上传视频")
            with gr.Column():
                btn = gr.Button("开始检测", variant="primary")
                result = gr.Markdown(label="检测结果")
                probs = gr.Label(label="预测概率")
        with gr.Row():
            frames = gr.Image(label="采样帧预览", type="numpy")
            freq = gr.Image(label="频域可视化", type="filepath")
        btn.click(fn=predict, inputs=video, outputs=[result, probs, frames, freq])
        gr.Markdown(
            "## 使用说明\n"
            "1. 推荐上传 mp4/mov 等常见视频格式。\n"
            "2. 如果模型是 smoke test 权重，结果只表示流程跑通，不代表真实性能。\n"
            "3. 正式展示请使用完整训练后的 `best.pt`。"
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/smoke_spatial/best.pt")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    global MODEL, CKPT_ARGS, CHECKPOINT_PATH
    CHECKPOINT_PATH = args.checkpoint
    MODEL, CKPT_ARGS = load_model(args.checkpoint)
    print(f"[device] {DEVICE}")
    print(f"[checkpoint] {args.checkpoint}")
    print(f"[model] {CKPT_ARGS.get('model', 'unknown')}")
    app = build_app()
    app.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
