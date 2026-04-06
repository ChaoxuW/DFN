import random

import torchvision.transforms as T
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

FIVE_SCALES = [480, 576, 688, 864, 1200]


class ResizeLongestSide:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return img.resize((new_w, new_h), Image.BILINEAR)


class RandomResizeFiveScales:
    def __call__(self, img):
        target = random.choice(FIVE_SCALES)
        return ResizeLongestSide(target)(img)


def get_vgg16_transform(train=True):
    norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if train:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        norm,
    ])


def get_fishernet_transform(train=True):
    norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if train:
        return T.Compose([
            RandomResizeFiveScales(),
            T.ToTensor(),
            norm,
            T.RandomHorizontalFlip(),
        ])
    return T.Compose([
        ResizeLongestSide(688),
        T.ToTensor(),
        norm,
    ])


# ── SVM multi-scale transforms (no flip) ──────────────────────────────────────

def _get_svm_transform(scale: int) -> T.Compose:
    """Base factory: resize longest side to `scale`, ToTensor, normalize."""
    norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return T.Compose([
        ResizeLongestSide(scale),
        T.ToTensor(),
        norm,
    ])


def get_svm_transform_scale_480() -> T.Compose:
    return _get_svm_transform(480)


def get_svm_transform_scale_576() -> T.Compose:
    return _get_svm_transform(576)


def get_svm_transform_scale_688() -> T.Compose:
    return _get_svm_transform(688)


def get_svm_transform_scale_864() -> T.Compose:
    return _get_svm_transform(864)


def get_svm_transform_scale_1200() -> T.Compose:
    return _get_svm_transform(1200)


def get_svm_transforms_all_scales() -> list:
    """Return a list of all five SVM transforms in ascending scale order."""
    return [_get_svm_transform(s) for s in FIVE_SCALES]
