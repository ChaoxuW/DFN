"""Test multi-label classification with SVM head.

Pipeline:
  1. Load trained FisherNet (fc_out bypassed).
  2. For each test image extract Fisher Vectors at 5 scales, average them,
     then apply power- and L2-normalization.
  3. Fit 20 one-vs-all LinearSVMs on training set Fisher Vectors.
  4. Evaluate on test set: report per-class AP and mAP.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from data.dataset import VOCDataset, VOC_CLASSES
from data.voc import get_svm_transforms_all_scales

# Hardcoded VOC root and splits
VOC_ROOT     = './datasets/VOC'
TRAIN_SPLITS = ['train2007', 'train2012', 'val2007', 'val2012']
TEST_SPLITS  = ['test2007']
from models.fisher.fisher_net import FisherNet
from models.heads.svm_head import SVMHead
from utils.logger import get_logger
from utils.metrics import compute_mAP
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="SVM test for FisherNet")
    p.add_argument("--fishernet_ckpt", type=str, required=True,
                   help="Path to trained FisherNet checkpoint")
    p.add_argument("--log_file",       type=str, default="logs/svm_test.log")
    p.add_argument("--num_gaussians",   type=int,   default=32)
    p.add_argument("--svm_C",           type=float, default=1.0)
    p.add_argument("--fv_cache_train",  type=str, default="checkpoints/train_fvs.npz",
                   help="Cache path for train FVs – reused on subsequent runs")
    p.add_argument("--fv_cache_test",   type=str, default="checkpoints/test_fvs.npz",
                   help="Cache path for test FVs")
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature extraction helper (raw PIL images needed for multi-scale transform)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_and_cache_fvs(svm_head, splits, transforms, cache_path, logger):
    """Extract normalized multi-scale Fisher Vectors with .npz caching.

    On first run extracts all FVs and saves to cache_path.
    On subsequent runs loads directly from cache_path (fast).
    """
    if os.path.exists(cache_path):
        logger.info(f"Loading cached FVs from {cache_path} ...")
        data = np.load(cache_path)
        return data["fvs"], data["labels"]

    fvs, labels = [], []
    for split in splits:
        raw_set = VOCDataset(root=VOC_ROOT, split=split, transform=None)
        logger.info(f"  [{split}] {len(raw_set)} images")
        for idx, (img, label) in enumerate(raw_set):
            fv = svm_head.extract_fv(img, transforms)   # (D,)
            fvs.append(fv.cpu().numpy())
            labels.append(label.numpy())
            if (idx + 1) % 200 == 0:
                logger.info(f"    {idx + 1}/{len(raw_set)}")

    fvs_np    = np.stack(fvs,    axis=0).astype(np.float32)
    labels_np = np.stack(labels, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, fvs=fvs_np, labels=labels_np)
    logger.info(f"Cached {fvs_np.shape[0]} FVs to {cache_path}")
    return fvs_np, labels_np


def main():
    args = parse_args()
    set_seed(args.seed)
    logger = get_logger("svm_test", args.log_file)
    logger.info(f"Args: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load FisherNet
    model = FisherNet(num_gaussians=args.num_gaussians)
    state = torch.load(args.fishernet_ckpt, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # Build SVMHead (wraps model, bypasses fc_out)
    svm_head = SVMHead(fisher_net=model, num_classes=len(VOC_CLASSES), C=args.svm_C)

    # Multi-scale transforms (no flip)
    transforms = get_svm_transforms_all_scales()

    # ── Extract training features and fit SVMs ────────────────────────────────
    logger.info(f"Extracting train Fisher Vectors from splits: {TRAIN_SPLITS}")
    train_fvs, train_labels = extract_and_cache_fvs(
        svm_head, TRAIN_SPLITS, transforms, args.fv_cache_train, logger
    )
    logger.info(f"Train FV shape: {train_fvs.shape}")
    logger.info("Fitting 20 one-vs-all SVMs ...")
    svm_head.fit(train_fvs, train_labels)
    logger.info("SVM training done.")

    # ── Extract test features ─────────────────────────────────────────────────
    logger.info(f"Extracting test Fisher Vectors from splits: {TEST_SPLITS}")
    test_fvs, test_labels = extract_and_cache_fvs(
        svm_head, TEST_SPLITS, transforms, args.fv_cache_test, logger
    )
    logger.info(f"Test FV shape: {test_fvs.shape}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    scores = svm_head.decision_function(test_fvs)   # (N, 20)
    ap_dict, mAP = compute_mAP(scores, test_labels, class_names=VOC_CLASSES)

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