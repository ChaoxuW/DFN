import torch
import torch.nn.functional as F


class SpGMM:
    """K-component Spherical GMM (vMF mixture) fitted with Spherical K-Means.

    Used to estimate per-component mean directions μ_k and concentration
    parameters κ_k from L2-normalized feature vectors, which then initialise
    the SpFisherLayer.

    Args:
        num_gaussians : number of mixture components K.
        num_iters     : number of K-Means iterations.
        eps           : small constant for numerical stability.
    """

    def __init__(self, num_gaussians: int = 32, num_iters: int = 100, eps: float = 1e-6):
        self.K = num_gaussians
        self.num_iters = num_iters
        self.eps = eps

        # Parameters set after fit()
        self.mu: torch.Tensor | None = None       # (K, D) component means (L2 normalized)
        self.kappa: torch.Tensor | None = None    # (K,)   concentration parameters

    def fit(self, X: torch.Tensor) -> "SpGMM":
        """Fit the Spherical K-Means to data X to approximate vMF.

        Args:
            X: (N, D) tensor of feature vectors. Will be L2 normalized internally.

        Returns:
            self (for chaining).
        """
        device = X.device
        N, D = X.shape

        # Ensure input features are on the unit hypersphere
        X = F.normalize(X, p=2, dim=-1, eps=self.eps)

        # ── K-means++ style initialization using cosine distance ─────────────
        mu = self._spherical_kmeans_plusplus_init(X)

        # ── Spherical K-Means iterations ─────────────────────────────────────
        for _ in range(self.num_iters):
            # E-step: Assign to closest center (max cosine similarity)
            C = torch.matmul(X, mu.T)              # (N, K)
            assign = torch.argmax(C, dim=-1)       # (N,)

            # M-step: Update centers
            new_mu = torch.zeros_like(mu)
            for k in range(self.K):
                mask = (assign == k)
                if mask.sum() > 0:
                    new_mu[k] = X[mask].sum(dim=0)
                else:
                    # Re-initialize empty cluster with a random point
                    idx = torch.randint(N, (1,), device=device).item()
                    new_mu[k] = X[idx]

            # Project back to the unit hypersphere
            mu = F.normalize(new_mu, p=2, dim=-1, eps=self.eps)

        # ── Estimate concentration parameter κ ───────────────────────────────
        C = torch.matmul(X, mu.T)
        assign = torch.argmax(C, dim=-1)
        kappa = torch.zeros(self.K, device=device)

        for k in range(self.K):
            mask = (assign == k)
            if mask.sum() > 0:
                # Mean resultant length
                R_bar = C[mask, k].mean().item()
                # Bound R_bar to avoid numerical explosion
                R_bar = min(max(R_bar, self.eps), 1.0 - self.eps)
                # Approximation formula for kappa
                kappa[k] = (R_bar * D - R_bar**3) / (1.0 - R_bar**2)
            else:
                kappa[k] = 1.0

        self.mu = mu
        self.kappa = kappa
        return self

    def get_fisher_init(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, kappa) for initialising SpFisherLayer parameters.

        Returns:
            mu: (K, D) tensor
            kappa: (K,) tensor
        """
        if self.mu is None or self.kappa is None:
            raise RuntimeError("Call fit() before get_fisher_init().")
        return self.mu, self.kappa

    def _spherical_kmeans_plusplus_init(self, X: torch.Tensor) -> torch.Tensor:
        """K-means++ initialization adapted for spherical space."""
        N, D = X.shape
        device = X.device
        chosen_idx = torch.randint(N, (1,), device=device).item()
        centers = [X[chosen_idx]]

        for _ in range(1, self.K):
            stack = torch.stack(centers, dim=0)               # (c, D)
            # Cosine distance: 1 - cosine_similarity
            sim = torch.matmul(X, stack.T)                    # (N, c)
            dists = 1.0 - sim.max(dim=1).values               # (N,)
            probs = dists.clamp(min=0.0) ** 2
            probs = probs / (probs.sum() + 1e-8)
            idx = torch.multinomial(probs, 1).item()
            centers.append(X[idx])

        return torch.stack(centers, dim=0)                    # (K, D)