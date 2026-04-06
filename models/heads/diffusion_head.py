"""Discrete Diffusion Head for multi-label VOC classification.

Follows the ADD (Authentic Discrete Diffusion) framework:
  - Forward : q(y_t | y_0) = N(sqrt(ᾱ_t) y_0,  (1 - ᾱ_t) I)
  - Training: L = E_t[ ᾱ_t · BCE(σ(f_θ(y_t, t, c)),  y_0) ]
  - Inference: argmax-and-renoise loop (argmax + re-noise → repeat)

The frozen FisherNet (or SpFisherNet) provides the conditioning vector c via the
same 5-scale extraction pipeline as SVMHead.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sinusoidal timestep embedding ─────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) long or float → (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None]            # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


# ── Adaptive LayerNorm ────────────────────────────────────────────────────────

class AdaLayerNorm(nn.Module):
    """LayerNorm whose scale and shift are conditioned on a timestep vector.

    Uses zero-centred initialisation so the module behaves like a plain
    LayerNorm at the start of training (scale=1, shift=0).
    """

    def __init__(self, emb_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * emb_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x   : (B, seq, emb_dim)
        # cond: (B, cond_dim)
        ss = self.proj(F.silu(cond))                       # (B, 2*emb_dim)
        scale, shift = ss.chunk(2, dim=-1)                  # (B, emb_dim)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ── Transformer block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Self-Attention + Cross-Attention + FFN, all gated by AdaLN.

    Self-Attention lets label tokens model co-occurrence and semantic topology.
    Cross-Attention lets label tokens query the Fisher context (K, V) for
    fine-grained feature-label binding.
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        cond_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention
        self.sa_norm = AdaLayerNorm(emb_dim, cond_dim)
        self.sa_attn = nn.MultiheadAttention(emb_dim, num_heads,
                                             dropout=dropout, batch_first=True)
        self.sa_drop = nn.Dropout(dropout)

        # Cross-attention: label tokens (Q) ← Fisher context (K, V)
        self.ca_norm_q = AdaLayerNorm(emb_dim, cond_dim)
        self.ca_norm_k = nn.LayerNorm(emb_dim)
        self.ca_attn   = nn.MultiheadAttention(emb_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        self.ca_drop   = nn.Dropout(dropout)

        # FFN
        self.ffn_norm = AdaLayerNorm(emb_dim, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        x       : (B, 20, emb_dim)   label tokens
        context : (B, S,  emb_dim)   Fisher context tokens (K, V)
        t_cond  : (B, cond_dim)      timestep conditioning for AdaLN
        """
        # Self-attention (pre-norm + residual)
        h = self.sa_norm(x, t_cond)
        h, _ = self.sa_attn(h, h, h)
        x = x + self.sa_drop(h)

        # Cross-attention (pre-norm + residual)
        q = self.ca_norm_q(x, t_cond)
        k = self.ca_norm_k(context)
        h, _ = self.ca_attn(q, k, k)
        x = x + self.ca_drop(h)

        # FFN (pre-norm + residual)
        h = self.ffn_norm(x, t_cond)
        x = x + self.ffn(h)

        return x


# ── Transformer denoiser ──────────────────────────────────────────────────────

class DenoisingTransformer(nn.Module):
    """Transformer denoiser for the ADD framework.

    Treats the 20 label dimensions as independent tokens:
      - Self-Attention  : label tokens attend to one another to model
                          co-occurrence patterns and semantic topology.
      - Cross-Attention : label tokens (Q) attend to Fisher context tokens
                          (K, V), enabling fine-grained feature-label binding.
      - AdaLN           : timestep embedding modulates every LayerNorm via
                          learned scale / shift (DiT-style).

    The conditioning Fisher vector is converted to ``num_context_tokens``
    context tokens via a two-stage MLP:
        fv_dim  →  2·emb_dim  →  num_context_tokens · emb_dim
    This keeps parameter count ≈ 10 M regardless of fv_dim.

    Args:
        fv_dim             : Fisher vector dim (FisherNet: 16384, SpFisher: 8224).
        num_classes        : number of label classes (20 for VOC).
        emb_dim            : transformer token dimension.
        num_heads          : number of attention heads (must divide emb_dim).
        num_layers         : number of TransformerBlock layers.
        t_emb_dim          : sinusoidal timestep embedding dimension.
        num_context_tokens : number of Fisher context tokens (KV sequence length).
        dropout            : dropout rate used throughout.
    """

    def __init__(
        self,
        fv_dim: int,
        num_classes: int = 20,
        emb_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        t_emb_dim: int = 128,
        num_context_tokens: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.num_classes        = num_classes
        self.num_context_tokens = num_context_tokens

        # Timestep conditioning: sinusoidal → MLP → emb_dim
        self.time_emb  = SinusoidalTimeEmbedding(t_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(t_emb_dim, 4 * t_emb_dim),
            nn.SiLU(),
            nn.Linear(4 * t_emb_dim, emb_dim),
        )

        # Label tokenisation: scalar y_t_i  →  emb_dim  +  positional embedding
        self.label_value_proj = nn.Linear(1, emb_dim)
        self.label_pos_emb    = nn.Embedding(num_classes, emb_dim)

        # Fisher vector → context tokens (two-stage projection)
        self.fv_proj = nn.Sequential(
            nn.Linear(fv_dim, emb_dim * 2),
            nn.GELU(),
            nn.Linear(emb_dim * 2, num_context_tokens * emb_dim),
        )
        self.context_pos_emb = nn.Embedding(num_context_tokens, emb_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, cond_dim=emb_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Per-label output head
        self.out_norm = nn.LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, 1)

    def forward(
        self,
        y_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            y_t : (B, num_classes) noisy label vector
            t   : (B,) long        timestep indices
            c   : (B, fv_dim)      conditioning Fisher vector
        Returns:
            logits : (B, num_classes)
        """
        B, device = y_t.shape[0], y_t.device

        # Timestep conditioning vector
        t_cond = self.time_proj(self.time_emb(t))          # (B, emb_dim)

        # Label tokens: per-label scalar → emb_dim + positional + timestep
        pos_ids = torch.arange(self.num_classes, device=device)
        x = (self.label_value_proj(y_t.unsqueeze(-1))      # (B, 20, emb_dim)
             + self.label_pos_emb(pos_ids))                 # broadcast (20, emb_dim)
        x = x + t_cond.unsqueeze(1)                         # (B, 20, emb_dim)

        # Fisher context tokens: two-stage projection + positional embedding
        ctx_ids = torch.arange(self.num_context_tokens, device=device)
        ctx = self.fv_proj(c).view(B, self.num_context_tokens, -1)  # (B, S, emb_dim)
        ctx = ctx + self.context_pos_emb(ctx_ids)                    # (B, S, emb_dim)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, ctx, t_cond)

        # Per-label logits
        x = self.out_norm(x)                                # (B, 20, emb_dim)
        return self.out_head(x).squeeze(-1)                 # (B, 20)


# ── DiffusionHead ─────────────────────────────────────────────────────────────

class DiffusionHead(nn.Module):
    """Discrete diffusion head on top of a frozen FisherNet.

    The frozen FisherNet acts purely as a feature extractor.
    Only DenoisingTransformer parameters are trained.

    Args:
        fisher_net         : trained FisherNet or SpFisherNet (will be frozen).
        fv_dim             : Fisher vector dimension (e.g. 16384 or 8224).
        num_classes        : number of classes (20 for VOC).
        T                  : total diffusion timesteps.
        hidden_dim         : transformer token dimension (``emb_dim``).
        t_emb_dim          : sinusoidal timestep embedding dimension.
        num_heads          : number of attention heads.
        num_layers         : number of TransformerBlock layers.
        num_context_tokens : Fisher vector is split into this many KV tokens.
        dropout            : dropout rate throughout the transformer.
        beta_start         : start value of linear noise schedule.
        beta_end           : end value of linear noise schedule.
    """

    def __init__(
        self,
        fisher_net,
        fv_dim: int,
        num_classes: int = 20,
        T: int = 1000,
        hidden_dim: int = 256,
        t_emb_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        num_context_tokens: int = 16,
        dropout: float = 0.1,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()

        # Freeze the feature extractor
        self.net_frozen = fisher_net
        self.net_frozen.eval()
        for p in self.net_frozen.parameters():
            p.requires_grad_(False)

        self.num_classes = num_classes
        self.T = T

        # Trainable denoiser (Transformer-based)
        self.denoiser = DenoisingTransformer(
            fv_dim=fv_dim,
            num_classes=num_classes,
            emb_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            t_emb_dim=t_emb_dim,
            num_context_tokens=num_context_tokens,
            dropout=dropout,
        )

        # Linear noise schedule
        betas     = torch.linspace(beta_start, beta_end, T)   # (T,)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)               # (T,) ᾱ_t

        self.register_buffer('betas',     betas)
        self.register_buffer('alpha_bar', alpha_bar)

    # ------------------------------------------------------------------
    # Feature extraction  (identical to SVMHead pipeline)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_fv(self, img, transforms: list) -> torch.Tensor:
        """Extract 5-scale averaged, power- and L2-normalized Fisher vector.

        Args:
            img        : PIL.Image
            transforms : list of 5 transforms (from get_svm_transforms_all_scales)
        Returns:
            fv : (fv_dim,) CPU tensor
        """
        self.net_frozen.eval()
        device = next(self.net_frozen.parameters()).device
        fvs = []
        for t in transforms:
            x = t(img).to(device)
            fv = self.net_frozen._forward_single(x, normalize=False)
            fvs.append(fv)
        avg_fv = torch.stack(fvs, dim=0).mean(dim=0)
        avg_fv = avg_fv.sign() * avg_fv.abs().sqrt()           # power norm
        avg_fv = F.normalize(avg_fv, p=2, dim=0, eps=1e-12)    # L2 norm
        return avg_fv.cpu()

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(self, y0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to y0 at timestep t.
        y0 : (B, C)  float binary labels {0, 1}
        t  : (B,)    long timestep indices
        Returns noisy y_t : (B, C)
        """
        ab = self.alpha_bar[t].view(-1, 1)   # (B, 1)  ᾱ_t
        noise = torch.randn_like(y0)
        return ab.sqrt() * y0 + (1.0 - ab).sqrt() * noise

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(
        self,
        c: torch.Tensor,
        y0: torch.Tensor,
        K_fold: int = 4,
    ) -> torch.Tensor:
        """Timestep-conditioned BCE loss with K-fold multi-timestep reuse.

        K-fold reuse (ADD paper §3.3): each conditioning vector c is paired
        with K different random timesteps, giving K× more gradient signal
        without re-extracting features.

        Args:
            c      : (B, fv_dim)  conditioning Fisher vectors
            y0     : (B, 20)      binary ground-truth labels
            K_fold : int          timesteps sampled per conditioning vector
        Returns:
            scalar loss
        """
        B, device = y0.shape[0], y0.device

        # Expand each sample K_fold times
        c_exp  = c.unsqueeze(1).expand(B, K_fold, -1).reshape(B * K_fold, -1)   # (B·K, fv_dim)
        y0_exp = y0.unsqueeze(1).expand(B, K_fold, -1).reshape(B * K_fold, -1)  # (B·K, 20)

        # Sample K_fold distinct timesteps per sample
        t = torch.randint(0, self.T, (B * K_fold,), device=device)              # (B·K,)

        # Forward process: add noise
        y_t = self.q_sample(y0_exp, t)                                           # (B·K, 20)

        # Predict logits
        logits = self.denoiser(y_t, t, c_exp)                                    # (B·K, 20)

        # Timestep-conditioned weighted BCE  (ᾱ_t as coefficient, ADD paper eq.)
        ab = self.alpha_bar[t].view(-1, 1)                                       # (B·K, 1)
        loss = F.binary_cross_entropy_with_logits(logits, y0_exp, reduction='none')  # (B·K, 20)
        return (ab * loss).mean()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor,
        num_steps: int = 20,
    ) -> torch.Tensor:
        """Iterative argmax-and-renoise sampling (ADD inference loop).

        Args:
            c         : (B, fv_dim) conditioning Fisher vectors
            num_steps : number of denoising steps
        Returns:
            (B, 20) binary predictions {0, 1}
        """
        B, device = c.shape[0], c.device
        y_t = torch.randn(B, self.num_classes, device=device)

        # Evenly spaced timesteps: T-1 → 0
        step_idx = torch.linspace(
            self.T - 1, 0, num_steps, device=device
        ).long()

        for i, t_val in enumerate(step_idx):
            t = torch.full((B,), int(t_val.item()), dtype=torch.long, device=device)
            logits  = self.denoiser(y_t, t, c)                  # (B, 20)
            y0_hat  = (torch.sigmoid(logits) > 0.5).float()     # (B, 20) discrete

            if i < num_steps - 1:
                # Re-noise with smaller coefficient at next (lower) timestep
                ab_prev = self.alpha_bar[step_idx[i + 1]].clamp(min=0.0)
                noise   = torch.randn_like(y0_hat)
                y_t = ab_prev.sqrt() * y0_hat + (1.0 - ab_prev).sqrt() * noise
            else:
                y_t = y0_hat

        return y_t  # (B, 20)

    @torch.no_grad()
    def predict_scores(
        self,
        c: torch.Tensor,
        num_steps: int = 20,
    ) -> torch.Tensor:
        """Run sampling and return sigmoid probabilities at the final step.

        Useful for mAP evaluation (ranking needs continuous scores).

        Returns:
            (B, 20) float scores in (0, 1)
        """
        B, device = c.shape[0], c.device
        y_t = torch.randn(B, self.num_classes, device=device)
        step_idx = torch.linspace(
            self.T - 1, 0, num_steps, device=device
        ).long()

        logits = None
        for i, t_val in enumerate(step_idx):
            t = torch.full((B,), int(t_val.item()), dtype=torch.long, device=device)
            logits = self.denoiser(y_t, t, c)
            y0_hat = (torch.sigmoid(logits) > 0.5).float()

            if i < num_steps - 1:
                ab_prev = self.alpha_bar[step_idx[i + 1]].clamp(min=0.0)
                noise   = torch.randn_like(y0_hat)
                y_t = ab_prev.sqrt() * y0_hat + (1.0 - ab_prev).sqrt() * noise

        return torch.sigmoid(logits)  # (B, 20)

