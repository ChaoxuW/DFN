"""Train Spherical FisherNet on VOC."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, ConcatDataset

from data.dataset import VOCDataset
from data.voc import get_fishernet_transform
from losses.msce_loss import MultiLabelSigmoidLoss
from models.fisher.sp_fisher_net import SpFisherNet
from models.fisher.sp_gmm import SpGMM
from utils.logger import get_logger
from utils.seed import set_seed

VOC_ROOT      = './datasets/VOC'
TRAIN_SPLITS  = ['train2007', 'train2012', 'val2007', 'val2012']

def variable_size_collate(batch):
    images = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch], dim=0)
    return images, labels

def parse_args():
    p = argparse.ArgumentParser(description="Train Spherical FisherNet")
    p.add_argument("--backbone_ckpt",    type=str, required=True)
    p.add_argument("--save_path",        type=str, default="checkpoints/sp_fishernet.pth")
    p.add_argument("--log_file",         type=str, default="logs/train_sp_fisher.log")
    p.add_argument("--batch_size",       type=int,   default=2)
    p.add_argument("--max_iters",        type=int,   default=40000)
    p.add_argument("--lr_fisher",        type=float, default=0.1)
    p.add_argument("--lr_fc_out",        type=float, default=0.001)
    p.add_argument("--lr_other",         type=float, default=0.0001)
    p.add_argument("--momentum",         type=float, default=0.9)
    p.add_argument("--weight_decay",     type=float, default=5e-4)
    p.add_argument("--num_gaussians",    type=int,   default=32)
    p.add_argument("--gmm_samples",      type=int,   default=5000)
    p.add_argument("--gmm_iters",        type=int,   default=50)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--log_interval",     type=int,   default=100)
    p.add_argument("--lr_decay_step",    type=int,   default=5000)
    p.add_argument("--lr_decay_gamma",   type=float, default=0.5)
    return p.parse_args()

@torch.no_grad()
def collect_patch_features(model, loader, max_samples, device, logger):
    model.eval()
    collected = []
    n = 0
    for images, _ in loader:
        for img in images:
            feat = model.features(img.unsqueeze(0).to(device)).squeeze(0)
            patches = model.spp(feat)
            emb = model.patch_embed(patches)
            collected.append(emb.cpu())
            n += emb.shape[0]
            if n >= max_samples:
                break
        if n >= max_samples:
            break
    all_feats = torch.cat(collected, dim=0)[:max_samples]
    return all_feats

def main():
    args = parse_args()
    set_seed(args.seed)
    logger = get_logger("train_sp_fisher", args.log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_fishernet_transform(train=True)
    train_set = ConcatDataset([VOCDataset(root=VOC_ROOT, split=s, transform=transform) for s in TRAIN_SPLITS])
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=variable_size_collate,
    )

    model = SpFisherNet(num_gaussians=args.num_gaussians)
    model.load_backbone_weights(args.backbone_ckpt)
    model.to(device)

    logger.info("Fitting Spherical GMM ...")
    patch_feats = collect_patch_features(model, train_loader, args.gmm_samples, device, logger)
    sp_gmm = SpGMM(num_gaussians=args.num_gaussians, num_iters=args.gmm_iters)
    sp_gmm.fit(patch_feats.to(device))
    model.init_from_sp_gmm(sp_gmm)
    
    model.init_gaussian_weights(mean=0.0, std=0.01)

    fisher_params   = list(model.fisher.parameters())
    fc_out_params   = list(model.fc_out.parameters())
    fisher_ids      = {id(p) for p in fisher_params}
    fc_out_ids      = {id(p) for p in fc_out_params}
    other_params    = [p for p in model.parameters() if id(p) not in fisher_ids and id(p) not in fc_out_ids]

    optimizer = torch.optim.SGD([
        {"params": other_params,   "lr": args.lr_other},
        {"params": fisher_params,  "lr": args.lr_fisher},
        {"params": fc_out_params,  "lr": args.lr_fc_out},
    ], momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    criterion = MultiLabelSigmoidLoss()

    model.train()
    data_iter  = iter(train_loader)
    total_loss = 0.0

    for iteration in range(1, args.max_iters + 1):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)

        labels = labels.to(device)
        optimizer.zero_grad()
        scores = model(images)
        loss   = criterion(scores, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if iteration % args.log_interval == 0:
            avg = total_loss / args.log_interval
            logger.info(f"Iter [{iteration}/{args.max_iters}] loss: {avg:.4f}")
            total_loss = 0.0

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":
    main()