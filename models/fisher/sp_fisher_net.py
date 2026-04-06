#这里写球面上的fisher_net代码
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.vgg import VGG16
from models.fisher.sp_fisher_layer import SpFisherLayer
from models.utils import SPPNet


class SpFisherNet(nn.Module):
    """Full Spherical FisherNet pipeline.

    Args:
        num_gaussians : K, number of vMF components (default 32).
        num_classes   : output dimension (default 20 for VOC).
    """

    def __init__(self, num_gaussians: int = 32, num_classes: int = 20):
        super(SpFisherNet, self).__init__()

        vgg = VGG16()

        self.features = nn.Sequential(*list(vgg.features.children())[:-1])
        self.spp = SPPNet()
        self.patch_embed = nn.Sequential(*list(vgg.classifier.children())[:8])

        # ── Spherical Fisher layer: (N_patches, 256) → (K·D + K,) ────────────
        self.fisher = SpFisherLayer(in_dim=256, num_gaussians=num_gaussians)

        # ── Final FC: (K·D + K,) → (num_classes,) ────────────────────────────
        self.fc_out = nn.Linear(self.fisher.out_dim, num_classes)

    def load_backbone_weights(self, path: str) -> "SpFisherNet":
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        if "model" in state:
            state = state["model"]

        feat_state = {k[len("features."):]: v for k, v in state.items() if k.startswith("features.")}
        if feat_state:
            self.features.load_state_dict(feat_state, strict=False)

        embed_state = {}
        for k, v in state.items():
            if k.startswith("classifier."):
                embed_state[k[len("classifier."):]] = v
            elif k.startswith("patch_embed."):
                embed_state[k[len("patch_embed."):]] = v
        if embed_state:
            self.patch_embed.load_state_dict(embed_state, strict=False)

        return self

    def init_from_sp_gmm(self, sp_gmm) -> "SpFisherNet":
        """Initialise SpFisherLayer mu and kappa from a fitted SpGMM."""
        mu, kappa = sp_gmm.get_fisher_init()
        with torch.no_grad():
            self.fisher.mu.copy_(mu)
            self.fisher.kappa.copy_(kappa)
        return self

    def init_gaussian_weights(self, mean: float = 0.0, std: float = 0.01) -> "SpFisherNet":
        nn.init.normal_(self.fc_out.weight, mean=mean, std=std)
        if self.fc_out.bias is not None:
            nn.init.constant_(self.fc_out.bias, 0.0)
        return self

    def _forward_single(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        device = next(self.parameters()).device
        feat = self.features(x.unsqueeze(0).to(device)).squeeze(0)
        patches = self.spp(feat)
        patch_feats = self.patch_embed(patches)
        phi = self.fisher(patch_feats)
        if normalize:
            phi = phi.sign() * (phi.abs() + 1e-12).sqrt()
            phi = F.normalize(phi, p=2, dim=0, eps=1e-12)
        return phi

    def forward(self, x) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            phis = [self._forward_single(img, normalize=False) for img in x]
        else:
            device = next(self.parameters()).device
            feats = self.features(x.to(device))
            phis = []
            for i in range(feats.shape[0]):
                patches = self.spp(feats[i])
                patch_feats = self.patch_embed(patches)
                phi = self.fisher(patch_feats)
                phis.append(phi)

        phi_batch = torch.stack(phis, dim=0)
        scores = self.fc_out(phi_batch)
        return scores