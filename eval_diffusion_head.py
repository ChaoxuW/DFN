"""Evaluate DiffusionHead on VOC test set.

Pipeline:
  1. Load frozen FisherNet + trained DenoisingMLP.
  2. Extract 5-scale Fisher Vectors for train and test sets (with caching).
  3. Run iterative diffusion sampling to get per-class scores.
  4. Compute per-class AP and mAP.

Usage:
  python eval_diffusion_head.py \\
      --fishernet_ckpt checkpoints/fishernet.pth \\
      --diffusion_ckpt checkpoints/diffusion_head.pth

  # SpFisherNet variant:
  python eval_diffusion_head.py \\
      --fishernet_ckpt checkpoints/sp_fishernet.pth \\
      --diffusion_ckpt checkpoints/diffusion_head.pth \\
      --model_type sp_fisher
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from data.dataset import VOCDataset, VOC_CLASSES
from data.voc import get_svm_transforms_all_scales
from models.heads.diffusion_head import DiffusionHead
from utils.logger import get_logger
from utils.metrics import compute_mAP
from utils.seed import set_seed

VOC_ROOT     = './datasets/VOC'
TRAIN_SPLITS = ['train2007', 'train2012', 'val2007', 'val2012']
TEST_SPLITS  = ['test2007']


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DiffusionHead on VOC")
    p.add_argument("--fishernet_ckpt",   type=str, required=True)
    p.add_argument("--diffusion_ckpt",   type=str, required=True)
    p.add_argument("--model_type",       type=str, default="fisher",
                   choices=["fisher", "sp_fisher"])
    p.add_argument("--num_gaussians",    type=int,   default=32)
    p.add_argument("--T",                type=int,   default=1000)
    p.add_argument("--hidden_dim",         type=int,   default=256,
                   help="Transformer token dimension (emb_dim)")
    p.add_argument("--t_emb_dim",          type=int,   default=128)
    p.add_argument("--num_heads",          type=int,   default=8)
    p.add_argument("--num_layers",         type=int,   default=4)
    p.add_argument("--num_context_tokens", type=int,   default=16,
                   help="Number of Fisher KV context tokens for cross-attention")
    p.add_argument("--dropout",            type=float, default=0.1)
    p.add_argument("--num_sample_steps", type=int,   default=20,
                   help="Denoising steps during inference")
    p.add_argument("--batch_size",       type=int,   default=256,
                   help="Batch size for diffusion inference (FVs are precomputed)")
    p.add_argument("--fv_cache_test",    type=str, default="checkpoints/test_fvs.npz")
    p.add_argument("--log_file",         type=str, default="logs/eval_diffusion.log")
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


# ── Feature extraction & caching ──────────────────────────────────────────────

def extract_and_cache_fvs(diff_head, splits, transforms, cache_path, logger):
    if os.path.exists(cache_path):
        logger.info(f"Loading cached FVs from {cache_path} ...")
        data = np.load(cache_path)
        return data["fvs"], data["labels"]

    logger.info(f"Extracting FVs for splits {splits} ...")
    fvs, labels = [], []
    for split in splits:
        raw_set = VOCDataset(root=VOC_ROOT, split=split, transform=None)
        logger.info(f"  [{split}] {len(raw_set)} images")
        for idx, (img, label) in enumerate(raw_set):
            fv = diff_head.extract_fv(img, transforms)
            fvs.append(fv.numpy())
            labels.append(label.numpy())
            if (idx + 1) % 200 == 0:
                logger.info(f"    {idx + 1}/{len(raw_set)}")

    fvs_np    = np.stack(fvs,    axis=0).astype(np.float32)
    labels_np = np.stack(labels, axis=0).astype(np.float32)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, fvs=fvs_np, labels=labels_np)
    logger.info(f"Cached {fvs_np.shape[0]} FVs to {cache_path}")
    return fvs_np, labels_np


# ── Batched diffusion inference ────────────────────────────────────────────────

@torch.no_grad()
def predict_all_scores(diff_head, fvs_np, batch_size, num_steps, device, logger):
    """Run predict_scores in batches, return (N, 20) numpy array."""
    N = fvs_np.shape[0]
    all_scores = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        c = torch.from_numpy(fvs_np[start:end]).to(device)
        scores = diff_head.predict_scores(c, num_steps=num_steps)  # (B, 20)
        all_scores.append(scores.cpu().numpy())
        if (end // batch_size) % 10 == 0:
            logger.info(f"  Inference: {end}/{N}")
    return np.concatenate(all_scores, axis=0)  # (N, 20)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    logger = get_logger("eval_diffusion", args.log_file)
    logger.info(f"Args: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load frozen FisherNet
    if args.model_type == "fisher":
        from models.fisher.fisher_net import FisherNet
        fisher_net = FisherNet(num_gaussians=args.num_gaussians)
        fv_dim = fisher_net.fisher.out_dim
    else:
        from models.fisher.sp_fisher_net import SpFisherNet
        fisher_net = SpFisherNet(num_gaussians=args.num_gaussians)
        fv_dim = fisher_net.fisher.out_dim

    state = torch.load(args.fishernet_ckpt, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    fisher_net.load_state_dict(state, strict=True)
    fisher_net.to(device)
    fisher_net.eval()

    # Build DiffusionHead and load trained denoiser weights
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

    denoiser_state = torch.load(args.diffusion_ckpt, map_location="cpu")
    diff_head.denoiser.load_state_dict(denoiser_state)
    diff_head.eval()
    logger.info(f"Loaded DenoisingTransformer from {args.diffusion_ckpt}")

    transforms = get_svm_transforms_all_scales()

    # Extract / load test FVs
    logger.info("Extracting test FVs ...")
    test_fvs, test_labels = extract_and_cache_fvs(
        diff_head, TEST_SPLITS, transforms, args.fv_cache_test, logger
    )
    logger.info(f"Test FV shape: {test_fvs.shape}")

    # Run diffusion inference
    logger.info(f"Running diffusion inference ({args.num_sample_steps} steps) ...")
    scores = predict_all_scores(
        diff_head, test_fvs, args.batch_size, args.num_sample_steps, device, logger
    )  # (N_test, 20)

    # Evaluate mAP
    ap_dict, mAP = compute_mAP(scores, test_labels.astype(int), class_names=VOC_CLASSES)

    logger.info("\n" + "=" * 50)
    logger.info("Per-class AP:")
    for cls, ap in ap_dict.items():
        logger.info(f"  {cls:<15s}  AP = {ap * 100:.1f}%")
    logger.info("-" * 50)
    logger.info(f"  mAP = {mAP * 100:.1f}%")
    logger.info("=" * 50)

    print("\nPer-class AP:")
    for cls, ap in ap_dict.items():
        print(f"  {cls:<15s}  {ap * 100:.1f}%")
    print(f"\nmAP: {mAP * 100:.2f}%")


if __name__ == "__main__":
    main()
