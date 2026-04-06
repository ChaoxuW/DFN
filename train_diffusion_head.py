"""Train DiffusionHead on top of a frozen FisherNet (or SpFisherNet).

Pipeline:
  1. Load trained FisherNet / SpFisherNet (frozen).
  2. Extract 5-scale Fisher Vectors for the full training set and cache to disk
     (only done once — subsequent runs reuse the cache).
  3. Train DenoisingMLP with timestep-conditioned BCE loss + K-fold reuse.

Usage:
  # With standard FisherNet:
  python train_diffusion_head.py --fishernet_ckpt checkpoints/fishernet.pth

  # With SpFisherNet:
  python train_diffusion_head.py --fishernet_ckpt checkpoints/sp_fishernet.pth \\
      --model_type sp_fisher
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import VOCDataset, VOC_CLASSES
from data.voc import get_svm_transforms_all_scales
from models.heads.diffusion_head import DiffusionHead
from utils.logger import get_logger
from utils.seed import set_seed

VOC_ROOT     = './datasets/VOC'
TRAIN_SPLITS = ['train2007', 'train2012', 'val2007', 'val2012']


def parse_args():
    p = argparse.ArgumentParser(description="Train DiffusionHead")
    p.add_argument("--fishernet_ckpt",  type=str, required=True,
                   help="Path to trained FisherNet or SpFisherNet checkpoint")
    p.add_argument("--model_type",      type=str, default="fisher",
                   choices=["fisher", "sp_fisher"])
    p.add_argument("--num_gaussians",   type=int,   default=32)
    p.add_argument("--save_path",       type=str, default="checkpoints/diffusion_head.pth")
    p.add_argument("--fv_cache",        type=str, default="checkpoints/train_fvs.npz",
                   help="Path to cache extracted Fisher Vectors (reused across runs)")
    p.add_argument("--log_file",        type=str, default="logs/train_diffusion.log")
    p.add_argument("--T",               type=int,   default=1000,
                   help="Total diffusion timesteps")
    p.add_argument("--max_iters",       type=int,   default=20000)
    p.add_argument("--batch_size",      type=int,   default=64)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--hidden_dim",         type=int,   default=256,
                   help="Transformer token dimension (emb_dim)")
    p.add_argument("--t_emb_dim",          type=int,   default=128)
    p.add_argument("--num_heads",          type=int,   default=8)
    p.add_argument("--num_layers",         type=int,   default=4)
    p.add_argument("--num_context_tokens", type=int,   default=16,
                   help="Number of Fisher KV context tokens for cross-attention")
    p.add_argument("--dropout",            type=float, default=0.1)
    p.add_argument("--K_fold",          type=int,   default=4,
                   help="Multi-timestep reuse factor (ADD paper K=4)")
    p.add_argument("--num_sample_steps",type=int,   default=20,
                   help="Denoising steps during evaluation")
    p.add_argument("--log_interval",    type=int,   default=200)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


# ── Feature extraction & caching ──────────────────────────────────────────────

def extract_and_cache_fvs(diff_head, splits, transforms, cache_path, logger):
    """Extract FVs for all splits and save as .npz (only done once)."""
    if os.path.exists(cache_path):
        logger.info(f"Loading cached FVs from {cache_path} ...")
        data = np.load(cache_path)
        return data["fvs"], data["labels"]

    logger.info("Extracting Fisher Vectors for training set (will be cached) ...")
    fvs, labels = [], []
    for split in splits:
        raw_set = VOCDataset(root=VOC_ROOT, split=split, transform=None)
        logger.info(f"  [{split}] {len(raw_set)} images")
        for idx, (img, label) in enumerate(raw_set):
            fv = diff_head.extract_fv(img, transforms)      # (fv_dim,)
            fvs.append(fv.numpy())
            labels.append(label.numpy())
            if (idx + 1) % 200 == 0:
                logger.info(f"    {idx + 1}/{len(raw_set)}")

    fvs_np    = np.stack(fvs,    axis=0).astype(np.float32)  # (N, fv_dim)
    labels_np = np.stack(labels, axis=0).astype(np.float32)  # (N, 20)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, fvs=fvs_np, labels=labels_np)
    logger.info(f"Cached {fvs_np.shape[0]} FVs to {cache_path}")
    return fvs_np, labels_np


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    logger = get_logger("train_diffusion", args.log_file)
    logger.info(f"Args: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load frozen FisherNet
    if args.model_type == "fisher":
        from models.fisher.fisher_net import FisherNet
        fisher_net = FisherNet(num_gaussians=args.num_gaussians)
        fv_dim = fisher_net.fisher.out_dim          # 2 * K * D = 16384
    else:
        from models.fisher.sp_fisher_net import SpFisherNet
        fisher_net = SpFisherNet(num_gaussians=args.num_gaussians)
        fv_dim = fisher_net.fisher.out_dim          # K * D + K = 8224

    state = torch.load(args.fishernet_ckpt, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    fisher_net.load_state_dict(state, strict=True)
    fisher_net.to(device)
    fisher_net.eval()
    logger.info(f"Loaded {args.model_type} FisherNet from {args.fishernet_ckpt}  (fv_dim={fv_dim})")

    # Build DiffusionHead (fisher_net already on device)
    diff_head = DiffusionHead(
        fisher_net=fisher_net,
        fv_dim=fv_dim,
        num_classes=len(VOC_CLASSES),
        T=args.T,
        hidden_dim=args.hidden_dim,
        t_emb_dim=args.t_emb_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_context_tokens=args.num_context_tokens,
        dropout=args.dropout,
    ).to(device)

    # Extract / load cached Fisher Vectors  ──────────────────────────────────
    transforms = get_svm_transforms_all_scales()
    fvs_np, labels_np = extract_and_cache_fvs(
        diff_head, TRAIN_SPLITS, transforms, args.fv_cache, logger
    )
    logger.info(f"Training FV shape: {fvs_np.shape}")

    # Build a simple TensorDataset from cached FVs
    fvs_t    = torch.from_numpy(fvs_np)
    labels_t = torch.from_numpy(labels_np)
    dataset  = TensorDataset(fvs_t, labels_t)
    loader   = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # Only train the DenoisingMLP — FisherNet is frozen
    optimizer = torch.optim.AdamW(
        diff_head.denoiser.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iters, eta_min=1e-6
    )

    # Training loop ─────────────────────────────────────────────────────────
    diff_head.train()
    diff_head.net_frozen.eval()   # keep feature extractor in eval at all times

    data_iter  = iter(loader)
    total_loss = 0.0

    for iteration in range(1, args.max_iters + 1):
        try:
            c, y0 = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            c, y0 = next(data_iter)

        c  = c.to(device)
        y0 = y0.to(device)

        optimizer.zero_grad()
        loss = diff_head(c, y0, K_fold=args.K_fold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diff_head.denoiser.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if iteration % args.log_interval == 0:
            avg = total_loss / args.log_interval
            lr  = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Iter [{iteration}/{args.max_iters}]  loss: {avg:.4f}  lr: {lr:.2e}"
            )
            total_loss = 0.0

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(diff_head.denoiser.state_dict(), args.save_path)
    logger.info(f"Saved DenoisingTransformer to {args.save_path}")


if __name__ == "__main__":
    main()
