import torch


class GMM:
    """K-component diagonal Gaussian Mixture Model fitted with EM.

    Used to estimate per-component means μ_k and standard deviations σ_k from
    a collection of feature vectors, which then initialise the FisherLayer:
        w_k = 1 / σ_k
        b_k = −μ_k

    Args:
        num_gaussians : number of mixture components K.
        num_iters     : number of EM iterations.
        eps           : small constant added to variance for numerical stability.
    """

    def __init__(self, num_gaussians: int = 32, num_iters: int = 100, eps: float = 1e-6):
        self.K = num_gaussians
        self.num_iters = num_iters
        self.eps = eps

        # Parameters set after fit()
        self.pi: torch.Tensor | None = None    # (K,)   mixing coefficients
        self.mu: torch.Tensor | None = None    # (K, D) component means
        self.sigma: torch.Tensor | None = None # (K, D) component std-devs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: torch.Tensor) -> "GMM":
        """Fit the GMM to data X via EM.

        Args:
            X: (N, D) tensor of feature vectors.

        Returns:
            self (for chaining).
        """
        N, D = X.shape
        K = self.K
        device = X.device

        # ── Initialise with k-means++ style seeding ───────────────────────
        mu = self._kmeans_plusplus_init(X)          # (K, D)
        sigma = torch.ones(K, D, device=device)     # (K, D)
        pi = torch.full((K,), 1.0 / K, device=device)

        # ── EM iterations ──────────────────────────────────────────────────
        for _ in range(self.num_iters):
            # E-step: compute responsibilities  r[n,k]
            log_r = self._log_responsibilities(X, pi, mu, sigma)  # (N, K)
            r = torch.softmax(log_r, dim=1)                        # (N, K)

            # M-step
            r_sum = r.sum(dim=0).clamp(min=self.eps)               # (K,)

            pi = r_sum / N                                          # (K,)

            mu = (r.T @ X) / r_sum.unsqueeze(1)                    # (K, D)

            diff = X.unsqueeze(1) - mu.unsqueeze(0)                # (N, K, D)
            var = (r.unsqueeze(2) * diff ** 2).sum(dim=0) / r_sum.unsqueeze(1)
            sigma = (var + self.eps).sqrt()                         # (K, D)

        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        return self

    def get_fisher_init(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (w, b) for initialising FisherLayer parameters.

            w_k = 1 / σ_k   shape (K, D)
            b_k = −μ_k      shape (K, D)

        Returns:
            w: (K, D) tensor
            b: (K, D) tensor
        """
        if self.mu is None or self.sigma is None:
            raise RuntimeError("Call fit() before get_fisher_init().")
        w = 1.0 / self.sigma.clamp(min=self.eps)
        b = -self.mu
        return w, b

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _kmeans_plusplus_init(self, X: torch.Tensor) -> torch.Tensor:
        """K-means++ initialisation for component means."""
        N, D = X.shape
        device = X.device
        chosen_idx = torch.randint(N, (1,), device=device).item()
        centers = [X[chosen_idx]]

        for _ in range(1, self.K):
            stack = torch.stack(centers, dim=0)             # (c, D)
            dists = torch.cdist(X, stack).min(dim=1).values # (N,)
            probs = (dists ** 2)
            probs = probs / (probs.sum() + 1e-8)
            idx = torch.multinomial(probs, 1).item()
            centers.append(X[idx])

        return torch.stack(centers, dim=0)                  # (K, D)

    def _log_responsibilities(
        self,
        X: torch.Tensor,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log(π_k) + log N(x; μ_k, σ_k²) for each (n, k) pair.

        Diagonal covariance is assumed, so the log-likelihood factorises
        across dimensions.

        Returns:
            (N, K) log unnormalised responsibilities.
        """
        D = X.shape[1]
        # X: (N, D) → (N, 1, D);  mu/sigma: (K, D) → (1, K, D)
        x_exp = X.unsqueeze(1)
        mu_exp = mu.unsqueeze(0)
        sigma_exp = sigma.unsqueeze(0).clamp(min=self.eps)

        log_norm = (
            -0.5 * D * torch.log(torch.tensor(2 * 3.141592653589793, device=X.device))
            - sigma_exp.log().sum(dim=-1)                          # (1, K)
            - 0.5 * ((x_exp - mu_exp) / sigma_exp).pow(2).sum(dim=-1)  # (N, K)
        )
        return pi.log().unsqueeze(0) + log_norm                    # (N, K)