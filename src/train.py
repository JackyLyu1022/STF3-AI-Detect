from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import GenVideoValDataset
from src.metrics import compute_binary_metrics, logits_to_fake_prob
from src.models.model_factory import ALL_MODEL_MODES, build_model
from src.utils import ensure_dir, get_device, save_json, set_seed


def collate_fn(batch):
    frames = torch.stack([b["frames"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    meta = [{k: v for k, v in b.items() if k not in {"frames", "label"}} for b in batch]
    return {"frames": frames, "labels": labels, "meta": meta}


class FakeWeightedCrossEntropy(nn.Module):
    def __init__(self, fake_weight: float = 1.0) -> None:
        super().__init__()
        self.fake_weight = float(fake_weight)
        self.base = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses = self.base(logits, labels)
        weights = torch.ones_like(losses)
        weights = torch.where(labels == 1, weights * self.fake_weight, weights)
        return (losses * weights).mean()


def make_train_criterion(args, train_ds: GenVideoValDataset, device: torch.device) -> nn.Module:
    fake_weight = float(args.fake_loss_weight)
    if args.pos_weight == "auto":
        labels = [int(row["label"]) for row in train_ds.rows]
        num_real = sum(1 for y in labels if y == 0)
        num_fake = sum(1 for y in labels if y == 1)
        if num_fake > 0:
            fake_weight *= num_real / num_fake
    if fake_weight <= 0:
        raise ValueError(f"fake loss weight must be positive, got {fake_weight}")
    if abs(fake_weight - 1.0) < 1e-12:
        return nn.CrossEntropyLoss()
    print(f"[loss] class_weight real=1.0000 fake={fake_weight:.4f}")
    return FakeWeightedCrossEntropy(fake_weight=fake_weight).to(device)


@torch.no_grad()
def run_eval(model, loader, device, amp=False):
    model.eval()
    labels_all, probs_all = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    for batch in tqdm(loader, desc="eval", leave=False):
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast(autocast_device, enabled=amp and device.type == "cuda"):
            out = model(frames, return_dict=True) if hasattr(model, "branch_names") else model(frames)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = criterion(logits, labels)
        probs = logits_to_fake_prob(logits).detach().cpu().numpy().tolist()
        labels_all.extend(labels.cpu().numpy().tolist())
        probs_all.extend(probs)
        total_loss += loss.item() * labels.numel()
    metrics = compute_binary_metrics(labels_all, probs_all).to_dict()
    metrics["loss"] = total_loss / max(1, len(labels_all))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/GenVideo-Val")
    parser.add_argument("--train-csv", default="data/GenVideo-Val/splits/random_train.csv")
    parser.add_argument("--val-csv", default="data/GenVideo-Val/splits/random_val.csv")
    parser.add_argument("--model", default="spatial", choices=ALL_MODEL_MODES)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pretrained-spatial", action="store_true")
    parser.add_argument("--foundation-backbone", default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14", "clip-vit-base-patch16", "xclip-base-patch16", "hf-dinov2-base"])
    parser.add_argument("--finetune-foundation", action="store_true", help="By default STF3-New freezes DINO/CLIP/XCLIP encoders.")
    parser.add_argument("--local-files-only", action="store_true", help="Use only local HuggingFace weights for CLIP/XCLIP/DINO HF backbones.")
    parser.add_argument("--wavelet-aug-prob", type=float, default=0.1)
    parser.add_argument("--wavelet-aug-mode", default="batch", choices=["batch", "bank"], help="Use donor bank to enable WaveRep augmentation with batch size 1.")
    parser.add_argument("--branch-dropout", type=float, default=0.1)
    parser.add_argument("--d3-loss", default="l2", choices=["l2", "cos"])
    parser.add_argument("--aux-loss-weight", type=float, default=0.2)
    parser.add_argument("--pos-weight", default="none", choices=["none", "auto"], help="Set fake class weight from train real/fake ratio.")
    parser.add_argument("--fake-loss-weight", type=float, default=1.0, help="Manual multiplier for fake class loss.")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--out-dir", default="runs/debug")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = get_device()
    print(f"[device] {device}")
    print(f"[model] {args.model}")

    train_ds = GenVideoValDataset(args.train_csv, args.data_root, args.num_frames, args.image_size, train=True, max_samples=args.max_train_samples or None)
    val_ds = GenVideoValDataset(args.val_csv, args.data_root, args.num_frames, args.image_size, train=False, max_samples=args.max_val_samples or None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate_fn)

    model = build_model(
        mode=args.model,
        pretrained_spatial=args.pretrained_spatial,
        foundation_backbone=args.foundation_backbone,
        freeze_foundation=not args.finetune_foundation,
        local_files_only=args.local_files_only,
        image_size=args.image_size,
        wavelet_aug_prob=args.wavelet_aug_prob,
        wavelet_aug_mode=args.wavelet_aug_mode,
        branch_dropout=args.branch_dropout,
        d3_loss=args.d3_loss,
    ).to(device)
    criterion = make_train_criterion(args, train_ds, device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[params] trainable={sum(p.numel() for p in trainable_params):,} total={sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    best_auc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        labels_all, probs_all = [], []
        start = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            frames = batch["frames"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(autocast_device, enabled=args.amp and device.type == "cuda"):
                out = model(frames, return_dict=True) if hasattr(model, "branch_names") else model(frames)
                if isinstance(out, dict):
                    logits = out["logits"]
                    loss = criterion(logits, labels)
                    aux = out.get("aux_logits", {})
                    if aux and args.aux_loss_weight > 0:
                        aux_loss = sum(criterion(v, labels) for v in aux.values()) / len(aux)
                        loss = loss + args.aux_loss_weight * aux_loss
                else:
                    logits = out
                    loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            probs = logits_to_fake_prob(logits).detach().cpu().numpy().tolist()
            labels_all.extend(labels.cpu().numpy().tolist())
            probs_all.extend(probs)
            running_loss += loss.item() * labels.numel()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_metrics = compute_binary_metrics(labels_all, probs_all).to_dict()
        train_metrics["loss"] = running_loss / max(1, len(labels_all))
        val_metrics = run_eval(model, val_loader, device, amp=args.amp)
        epoch_info = {
            "epoch": epoch,
            "seconds": time.time() - start,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_info)
        save_json({"args": vars(args), "history": history}, out_dir / "history.json")
        print(f"[epoch {epoch}] train acc={train_metrics['acc']:.4f} auc={train_metrics['auc']:.4f} loss={train_metrics['loss']:.4f} | val acc={val_metrics['acc']:.4f} auc={val_metrics['auc']:.4f} loss={val_metrics['loss']:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
        scheduler.step()

        auc = val_metrics.get("auc", float("nan"))
        score = auc if auc == auc else val_metrics.get("acc", 0.0)
        if score > best_auc:
            best_auc = score
            ckpt = {
                "model_state": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"[save] {out_dir / 'best.pt'}")

    torch.save({"model_state": model.state_dict(), "args": vars(args), "history": history}, out_dir / "last.pt")
    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
