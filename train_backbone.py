"""Train VGG16 backbone on VOC for multi-label classification.

Training config (from paper):
  - 9000 iterations, mini-batch size 32
  - Conv/other layers  lr = 0.001
  - Classifier layers  lr = 0.01
  - Divide lr by 10 after every 3000 iterations
  - Optimizer: SGD momentum=0.9, weight_decay=5e-4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, ConcatDataset

from data.dataset import VOCDataset
from data.voc import get_vgg16_transform

# Hardcoded VOC root and splits
VOC_ROOT      = './datasets/VOC'
TRAIN_SPLITS  = ['train2007', 'train2012', 'val2007', 'val2012']
from losses.msce_loss import MultiLabelSigmoidLoss
from models.backbone.vgg import VGG16
from utils.logger import get_logger
from utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train VGG16 backbone")
    p.add_argument("--save_path",           type=str, default="checkpoints/backbone.pth")
    p.add_argument("--log_file",            type=str, default="logs/train_backbone.log")
    p.add_argument("--batch_size",          type=int,   default=32)
    p.add_argument("--max_iters",           type=int,   default=9000)
    p.add_argument("--lr_classifier",       type=float, default=0.01)
    p.add_argument("--lr_base",             type=float, default=0.001)
    p.add_argument("--momentum",            type=float, default=0.9)
    p.add_argument("--weight_decay",        type=float, default=5e-4)
    p.add_argument("--lr_decay_step",       type=int,   default=3000)
    p.add_argument("--lr_decay_gamma",      type=float, default=0.1)
    p.add_argument("--num_workers",         type=int,   default=4)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--pretrained_imagenet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    logger = get_logger("train_backbone", args.log_file)
    logger.info(f"Args: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    transform = get_vgg16_transform(train=True)
    train_set = ConcatDataset([
        VOCDataset(root=VOC_ROOT, split=s, transform=transform)
        for s in TRAIN_SPLITS
    ])
    logger.info(f"Training samples: {len(train_set)} across splits {TRAIN_SPLITS}")
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    model = VGG16()
    if args.pretrained_imagenet:
        logger.info("Loading ImageNet pretrained weights ...")
        model.load_pretrained_imagenet()
    model.to(device)

    classifier_params = list(model.classifier.parameters())
    cls_ids           = {id(p) for p in classifier_params}
    base_params       = [p for p in model.parameters() if id(p) not in cls_ids]

    optimizer = torch.optim.SGD(
        [
            {"params": base_params,       "lr": args.lr_base},
            {"params": classifier_params, "lr": args.lr_classifier},
        ],
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma,
    )
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

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if iteration % 100 == 0:
            avg = total_loss / 100
            lr0 = optimizer.param_groups[0]["lr"]
            lr1 = optimizer.param_groups[1]["lr"]
            logger.info(
                f"Iter [{iteration}/{args.max_iters}]  loss: {avg:.4f}  "
                f"lr_base: {lr0:.2e}  lr_cls: {lr1:.2e}"
            )
            total_loss = 0.0

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    logger.info(f"Saved backbone to {args.save_path}")


if __name__ == "__main__":
    main()
