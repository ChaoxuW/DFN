import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        # Naming matches torchvision VGG16 so pretrained weights can be loaded
        # via load_state_dict(..., strict=False)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),    # 0
            nn.ReLU(inplace=True),                          # 1
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # 2
            nn.ReLU(inplace=True),                          # 3
            nn.MaxPool2d(kernel_size=2, stride=2),          # 4
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 5
            nn.ReLU(inplace=True),                          # 6
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # 7
            nn.ReLU(inplace=True),                          # 8
            nn.MaxPool2d(kernel_size=2, stride=2),          # 9
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 10
            nn.ReLU(inplace=True),                          # 11
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 12
            nn.ReLU(inplace=True),                          # 13
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 14
            nn.ReLU(inplace=True),                          # 15
            nn.MaxPool2d(kernel_size=2, stride=2),          # 16
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # 17
            nn.ReLU(inplace=True),                          # 18
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 19
            nn.ReLU(inplace=True),                          # 20
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 21
            nn.ReLU(inplace=True),                          # 22
            nn.MaxPool2d(kernel_size=2, stride=2),          # 23
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 24
            nn.ReLU(inplace=True),                          # 25
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 26
            nn.ReLU(inplace=True),                          # 27
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 28
            nn.ReLU(inplace=True),                          # 29
            nn.MaxPool2d(kernel_size=2, stride=2),          # 30
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # 0  fc1 — matches pretrained key
            nn.ReLU(inplace=True),          # 1
            nn.Dropout(),                   # 2
            nn.Linear(4096, 4096),          # 3  fc2 — matches pretrained key
            nn.ReLU(inplace=True),          # 4
            nn.Dropout(),                   # 5
            nn.Linear(4096, 256),           # 6  replaces original fc3
            nn.ReLU(inplace=True),          # 7
            nn.Linear(256, 20),             # 8  final output: 20 classes
        )

    def load_pretrained_imagenet(self) -> "VGG16":
        """Load matching weights from the official torchvision VGG16
        pretrained on ImageNet.  Layers whose shapes differ (e.g.
        classifier.6 which is 256-out instead of 1000-out) are skipped.
        """
        from torchvision.models import vgg16, VGG16_Weights
        pretrained_state = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).state_dict()
        own_state = self.state_dict()
        # Keep only keys that exist in current model AND have matching shapes
        filtered = {
            k: v for k, v in pretrained_state.items()
            if k in own_state and own_state[k].shape == v.shape
        }
        own_state.update(filtered)
        self.load_state_dict(own_state, strict=True)
        return self

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
