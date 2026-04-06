import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]
NUM_CLASSES = len(VOC_CLASSES)   # 20


class VOCDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.images_dir = Path(root) / 'images' / split
        self.labels_dir = Path(root) / 'labels' / split
        self.transform = transform
        self.samples = []
        for label_path in sorted(self.labels_dir.glob('*.txt')):
            img_path = self.images_dir / (label_path.stem + '.jpg')
            if img_path.exists():
                self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
                if 0 <= class_id < NUM_CLASSES:
                    label[class_id] = 1.0

        return image, label
