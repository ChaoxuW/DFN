import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.vgg import VGG16
from models.fisher.fisher_layer import FisherLayer
from models.utils import SPPNet


class FisherNet(nn.Module):
    """Full FisherNet pipeline.

    Architecture (per image):
        1. VGG16 conv blocks 1-5, last MaxPool removed  → (512, H', W')
        2. SPP                                           → (N_patches, 25088)
        3. VGG16 classifier[0:8]  (fc1→fc2→fc3→ReLU)   → (N_patches, 256)
        4. FisherLayer                                   → (2·K·D,)
        5. fc_out                                        → (20,)

    Args:
        num_gaussians : K, number of Fisher/GMM components (default 32).
        num_classes   : output dimension (default 20 for VOC).
    """

    def __init__(self, num_gaussians: int = 32, num_classes: int = 20):
        super(FisherNet, self).__init__()

        vgg = VGG16()

        # ── Conv backbone: features[0..29], drop the last MaxPool (index 30) ──
        self.features = nn.Sequential(*list(vgg.features.children())[:-1])

        # ── SPP: single-image (512, H', W') → (N_patches, 25088) ─────────────
        self.spp = SPPNet()

        # ── Patch embedding: classifier indices 0-7 → (N_patches, 256) ───────
        # 0: Linear(25088→4096)  1: ReLU  2: Dropout
        # 3: Linear(4096→4096)   4: ReLU  5: Dropout
        # 6: Linear(4096→256)    7: ReLU
        self.patch_embed = nn.Sequential(*list(vgg.classifier.children())[:8])

        # ── Fisher layer: (N_patches, 256) → (2·K·D,) ────────────────────────
        self.fisher = FisherLayer(in_dim=256, num_gaussians=num_gaussians)

        # ── Final FC: (2·K·D,) → (num_classes,) ──────────────────────────────
        self.fc_out = nn.Linear(self.fisher.out_dim, num_classes)

    # ------------------------------------------------------------------
    # Weight initialisation helpers
    # ------------------------------------------------------------------

    def load_backbone_weights(self, path: str) -> "FisherNet":
        """Load fine-tuned backbone weights into self.features and
        self.patch_embed.

        The checkpoint may be either a full FisherNet state dict or a raw
        VGG16 state dict — both are handled.

        Args:
            path: path to the .pth checkpoint file.
        """
        state = torch.load(path, map_location="cpu")
        # unwrap common checkpoint wrappers
        if "state_dict" in state:
            state = state["state_dict"]
        if "model" in state:
            state = state["model"]

        # ── features ──────────────────────────────────────────────────
        feat_state = {
            k[len("features."):]: v
            for k, v in state.items()
            if k.startswith("features.")
        }
        if feat_state:
            self.features.load_state_dict(feat_state, strict=False)

        # ── patch_embed (classifier fc layers) ────────────────────────
        embed_state = {}
        for k, v in state.items():
            if k.startswith("classifier."):
                embed_state[k[len("classifier."):]] = v
            elif k.startswith("patch_embed."):
                embed_state[k[len("patch_embed."):]] = v
        if embed_state:
            self.patch_embed.load_state_dict(embed_state, strict=False)

        return self

    def init_from_gmm(self, gmm) -> "FisherNet":
        """Initialise FisherLayer w and b from a fitted GMM.

        Args:
            gmm: a fitted GMM instance with get_fisher_init() method.
        """
        w, b = gmm.get_fisher_init()
        with torch.no_grad():
            self.fisher.w.copy_(w)
            self.fisher.b.copy_(b)
        return self

    def init_gaussian_weights(
        self, mean: float = 0.0, std: float = 0.01
    ) -> "FisherNet":
        """Initialise only fc_out with Gaussian(mean, std).
        patch_embed keeps the backbone pretrained weights unchanged.
        Bias is set to 0.
        """
        nn.init.normal_(self.fc_out.weight, mean=mean, std=std)
        if self.fc_out.bias is not None:
            nn.init.constant_(self.fc_out.bias, 0.0)
        return self

    # ------------------------------------------------------------------

    def _forward_single(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Process one image.

        Args:
            x         : (3, H, W) single image tensor (no batch dim).
            normalize : if True (default) apply power + L2 norm before return.
                        Pass False when the caller will average multiple scales
                        before normalizing (e.g. SVMHead.extract_fv).
        Returns:
            phi: (2·K·D,) Fisher vector.
        """
        device = next(self.parameters()).device
        # features expects a batch dim
        feat = self.features(x.unsqueeze(0).to(device)).squeeze(0)   # (512, H', W')
        patches = self.spp(feat)                           # (N_patches, 25088)
        patch_feats = self.patch_embed(patches)            # (N_patches, 256)
        phi = self.fisher(patch_feats)                     # (2·K·D,)
        if normalize:
            # +1e-12 inside sqrt to avoid Inf gradient at exactly-zero entries
            phi = phi.sign() * (phi.abs() + 1e-12).sqrt()
            phi = F.normalize(phi, p=2, dim=0, eps=1e-12)
        return phi

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) batch tensor  OR  list of (3, H, W) tensors
               (list form is used when images have different spatial sizes).
        Returns:
            scores: (B, num_classes) raw class scores.
        """
        if isinstance(x, (list, tuple)):
            # Variable-size images: must process individually (training path)
            phis = [self._forward_single(img, normalize=False) for img in x]
        else:
            # Same-size batch: run VGG features as one batch for GPU efficiency,
            # then loop over each image for SPP / Fisher (which are per-image).
            device = next(self.parameters()).device
            feats = self.features(x.to(device))            # (B, 512, H', W')
            phis = []
            for i in range(feats.shape[0]):
                patches = self.spp(feats[i])               # (N_patches, 25088)
                patch_feats = self.patch_embed(patches)    # (N_patches, 256)
                phi = self.fisher(patch_feats)             # (2·K·D,)
                # No normalization here: paper trains on raw Fisher vectors
                phis.append(phi)

        phi_batch = torch.stack(phis, dim=0)               # (B, 2·K·D)
        scores = self.fc_out(phi_batch)                    # (B, num_classes)
        return scores

