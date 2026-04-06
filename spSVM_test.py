"""Test multi-label classification with SVM head on Spherical FisherNet."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from data.dataset import VOCDataset, VOC_CLASSES
from data.voc import get_svm_transforms_all_scales

VOC_ROOT     = './datasets/VOC'
TRAIN_SPLITS = ['train2007', 'train2012', 'val2007', 'val2012']
TEST_SPLITS  = ['test2007']

from models.fisher.sp_fisher_net import SpFisherNet
from models.heads.svm_head import SVMHead
from utils.logger import get_logger
from utils.metrics import compute_mAP
from utils.seed import set_seed

def parse_args():
    p = argparse.ArgumentParser(description="SVM test for Spherical FisherNet")
    p.add_argument("--fishernet_ckpt",  type=str, required=True)
    p.add_argument("--log_file",        type=str, default="logs/spsvm_test.log")
    p.add_argument("--num_gaussians",   type=int,   default=32)
    p.add_argument("--svm_C",           type=float, default=1.0)
    p.add_argument("--fv_cache_train",  type=str, default="checkpoints/sp_train_fvs.npz",
                   help="Cache path for train FVs – reused on subsequent runs")
    p.add_argument("--fv_cache_test",   type=str, default="checkpoints/sp_test_fvs.npz",
                   help="Cache path for test FVs")
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()

@torch.no_grad()
def extract_and_cache_fvs(svm_head, splits, transforms, cache_path, logger):
    if os.path.exists(cache_path):
        logger.info(f"Loading cached FVs from {cache_path} ...")
        data = np.load(cache_path)
        return data["fvs"], data["labels"]

    fvs, labels = [], []
    for split in splits:
        raw_set = VOCDataset(root=VOC_ROOT, split=split, transform=None)
        logger.info(f"  [{split}] {len(raw_set)} images")
        for idx, (img, label) in enumerate(raw_set):
            fv = svm_head.extract_fv(img, transforms)
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
    logger = get_logger("spsvm_test", args.log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpFisherNet(num_gaussians=args.num_gaussians)
    state = torch.load(args.fishernet_ckpt, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    svm_head = SVMHead(fisher_net=model, num_classes=len(VOC_CLASSES), C=args.svm_C)
    transforms = get_svm_transforms_all_scales()

    logger.info(f"Extracting train FVs from splits: {TRAIN_SPLITS}")
    train_fvs, train_labels = extract_and_cache_fvs(
        svm_head, TRAIN_SPLITS, transforms, args.fv_cache_train, logger
    )
    logger.info(f"Train FV shape: {train_fvs.shape}")
    svm_head.fit(train_fvs, train_labels)

    logger.info(f"Extracting test FVs from splits: {TEST_SPLITS}")
    test_fvs, test_labels = extract_and_cache_fvs(
        svm_head, TEST_SPLITS, transforms, args.fv_cache_test, logger
    )
    logger.info(f"Test FV shape: {test_fvs.shape}")
    scores = svm_head.decision_function(test_fvs)
    
    ap_dict, mAP = compute_mAP(scores, test_labels, class_names=VOC_CLASSES)

    for cls, ap in ap_dict.items():
        print(f"  {cls:<15s}  {ap * 100:.1f}%")
    print(f"\nmAP: {mAP * 100:.2f}%")

if __name__ == "__main__":
    main()