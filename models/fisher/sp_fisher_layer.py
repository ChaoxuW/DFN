#这里写球面上的FV代码
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpFisherLayer(nn.Module):
    """Spherical Fisher Layer based on von Mises-Fisher (vMF) distribution.

    For each of the K vMF components the layer maintains two parameters:
        mu_k    (shape D) — component mean direction, L2 normalized in forward
        kappa_k (shape 1) — concentration parameter

    Forward computation:
        c_ijk   = x_ij^T mu_k                               cosine similarity
        γ_j(k)  = softmax_k(kappa_k * c_ijk)                soft-assignment weights

        Tangent (first-order) gradient:
            v_ijk = x_ij - c_ijk * mu_k
            G_mu  = γ_j(k) * √kappa_k * (v_ijk / √(1 - c_ijk²))

        Concentration (second-order) gradient:
            G_kappa = γ_j(k) * (1/√2) * [kappa_k * (1 - c_ijk) - 1]

        φ(x_ij) = concat(G_mu, G_kappa)  flattened  [N, K·D + K]
        φ(X_i)  = mean_j φ(x_ij)                    mean-pooling over patches

    Args:
        in_dim        : dimension D of each input patch (default 256).
        num_gaussians : number of vMF components K (default 32).
    """

    def __init__(self, in_dim: int = 256, num_gaussians: int = 32, eps: float = 1e-4):
        super(SpFisherLayer, self).__init__()
        self.D = in_dim
        self.K = num_gaussians
        self.eps = eps

        # mu: component directions (shape K × D)
        self.mu = nn.Parameter(torch.randn(num_gaussians, in_dim))
        # kappa: concentration parameters (shape K)
        self.kappa = nn.Parameter(torch.ones(num_gaussians))

        self._init_weights()

    def _init_weights(self):
        # Initialise with random directions on hypersphere
        nn.init.normal_(self.mu, mean=0.0, std=1.0)
        # Initialize concentration to a reasonable positive scalar
        nn.init.constant_(self.kappa, 10.0)

    @property
    def out_dim(self) -> int:
        """Output dimension: K · D (for mu) + K (for kappa)."""
        return self.K * self.D + self.K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (N, D) patches for a single image.

        Returns:
            phi : (K·D + K,) Spherical Fisher vector for the image.
        """
        N, D = x.shape
        
        # Ensure inputs and cluster centers are L2 normalized (Spherical geometry)
        x_norm = F.normalize(x, p=2, dim=-1, eps=self.eps)
        mu_norm = F.normalize(self.mu, p=2, dim=-1, eps=self.eps)
        kappa_pos = self.kappa.clamp(min=self.eps)

        # ── Step 1: Cosine similarity c_ijk ──────────────────────────────────
        c = torch.matmul(x_norm, mu_norm.T)                     # (N, K)

        # ── Step 2: Soft-assignment γ_j(k) ───────────────────────────────────
        kappa_view = kappa_pos.view(1, self.K)                  # (1, K)
        gamma = F.softmax(kappa_view * c, dim=-1)               # (N, K)

        # ── Step 3: First-order term (Tangent Projection) ────────────────────
        x_exp = x_norm.unsqueeze(1)                             # (N, 1, D)
        c_exp = c.unsqueeze(-1)                                 # (N, K, 1)
        mu_exp = mu_norm.unsqueeze(0)                           # (1, K, D)

        v = x_exp - c_exp * mu_exp                              # (N, K, D) residual
        denominator = torch.sqrt(1.0 - c_exp**2 + self.eps)     # (N, K, 1) sin(theta)

        gamma_exp = gamma.unsqueeze(-1)                         # (N, K, 1)
        kappa_exp = kappa_pos.view(1, self.K, 1)                # (1, K, 1)

        # G_mu shape: (N, K, D)
        G_mu = gamma_exp * torch.sqrt(kappa_exp) * (v / denominator)
        G_mu_flat = G_mu.reshape(N, self.K * self.D)            # (N, K·D)

        # ── Step 4: Second-order term (Concentration Deviation) ──────────────
        # G_kappa shape: (N, K)
        G_kappa = gamma * (1.0 / math.sqrt(2)) * (kappa_view * (1.0 - c) - 1.0)

        # ── Step 5: Concat and Pooling ───────────────────────────────────────
        phi_patches = torch.cat([G_mu_flat, G_kappa], dim=-1)   # (N, K·D + K)
        phi = phi_patches.mean(dim=0)                           # (K·D + K,)

        return phi