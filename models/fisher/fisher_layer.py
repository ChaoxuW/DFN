import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FisherLayer(nn.Module):
    """Fisher Layer as described in Figure 3.

    For each of the K Gaussian components the layer maintains two parameter
    vectors:
        w_k  (shape D) — scale, analogous to 1/σ_k
        b_k  (shape D) — bias,  analogous to −μ_k

    Forward computation (notation follows the paper):
        y_ijk_1 = w_k ⊙ (x_ij + b_k)               element-wise product
        y_ijk_2 = y_ijk_1²                            square
        y_ijk_3 = Σ_d y_ijk_2d  = y_ijk_1ᵀ y_ijk_1  sum over dim D → scalar
        y_ijk_4 = −½ y_ijk_3                          scale(−½)
        γ_j(k)  = softmax_k(y_ijk_4)                 soft-assignment weights

        y_ijk_5 = y_ijk_2 − 1                         bias(−1)
        y_ijk_6 = y_ijk_5 / √2                        scale(1/√2)

        σ-part  = γ_j(k) · y_ijk_6     [N, K, D]     upper Product in figure
        μ-part  = γ_j(k) · y_ijk_1     [N, K, D]     lower Product in figure

        φ(x_ij) = concat(σ-part, μ-part)  flattened  [N, 2·K·D]
        φ(X_i)  = mean_j φ(x_ij)                     mean-pooling over patches

    Args:
        in_dim        : dimension D of each input patch (default 256).
        num_gaussians : number of Gaussian components K (default 32).
    """

    def __init__(self, in_dim: int = 256, num_gaussians: int = 32):
        super(FisherLayer, self).__init__()
        self.D = in_dim
        self.K = num_gaussians

        # w: scale parameters, initialised to 1  (shape K × D)
        self.w = nn.Parameter(torch.ones(num_gaussians, in_dim))
        # b: bias parameters,  initialised to 0  (shape K × D)
        self.b = nn.Parameter(torch.zeros(num_gaussians, in_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.w, mean=1.0, std=0.01)
        nn.init.normal_(self.b, mean=0.0, std=0.01)

    @property
    def out_dim(self) -> int:
        """Output dimension: 2 · K · D."""
        return 2 * self.K * self.D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (N, D) patches for a single image, where N is the number of
                patches and D == self.in_dim.

        Returns:
            phi : (2·K·D,) Fisher vector for the image.
        """
        N, D = x.shape
        K = self.K

        # ── Step 1: element-wise product with bias ───────────────────────────
        # x: (N, D) → (N, 1, D) broadcast over K
        x_exp = x.unsqueeze(1)                          # (N, 1, D)
        y1 = self.w * (x_exp + self.b)                 # (N, K, D)

        # ── Step 2: square ───────────────────────────────────────────────────
        y2 = y1 ** 2                                    # (N, K, D)

        # ── Step 3: sum over feature dimension ──────────────────────────────
        y3 = y2.sum(dim=-1)                             # (N, K)

        # ── Step 4: scale(−½) ───────────────────────────────────────────────
        y4 = -0.5 * y3                                  # (N, K)

        # ── Step 5: soft-assignment weights via softmax ──────────────────────
        gamma = F.softmax(y4, dim=-1)                   # (N, K)

        # ── Step 6: bias(−1) ────────────────────────────────────────────────
        y5 = y2 - 1                                     # (N, K, D)

        # ── Step 7: scale(1/√2) ─────────────────────────────────────────────
        y6 = y5 / math.sqrt(2)                          # (N, K, D)

        # ── Step 8 & 9: weighted products (concat in figure) ────────────────
        g = gamma.unsqueeze(-1)                         # (N, K, 1)
        sigma_part = (g * y6).reshape(N, K * D)        # (N, K·D)  σ-gradient
        mu_part    = (g * y1).reshape(N, K * D)        # (N, K·D)  μ-gradient

        # ── Step 10: concat σ and μ parts ───────────────────────────────────
        phi_patches = torch.cat([sigma_part, mu_part], dim=-1)  # (N, 2·K·D)

        # ── Step 11: mean-pooling over patches ──────────────────────────────
        phi = phi_patches.mean(dim=0)                   # (2·K·D,)

        return phi
