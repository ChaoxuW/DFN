import torch
import torch.nn as nn
import torch.nn.functional as F


# SPP window sizes in FEATURE MAP space.
# These correspond to image-space patches of [64,96,128,160,192,224,256] px
# divided by the VGG stride of 16 (4 MaxPool layers, each stride-2).
_SPP_SCALES = [4, 6, 8, 10, 12, 14, 16]
# sub-window grid inside each scale window: 7x7 max-pool -> output (7,7)
_POOL_OUTPUT = 7
# sliding stride in feature map space (image stride 32 / VGG stride 16 = 2)
_STRIDE = 2


def _extract_patches_at_scale(feat: torch.Tensor, scale: int):
    """Extract all sliding windows of size `scale x scale` from `feat`,
    apply adaptive max-pool to each window to get a (C,7,7) tensor,
    then flatten to 25088.

    Vectorised with Tensor.unfold — no Python loops over patches.

    Args:
        feat:  (C, H, W) feature map (single image, no batch dim)
        scale: window size in feature-map pixels

    Returns:
        Tensor of shape (num_patches, C*7*7)  or None if the feature map is
        smaller than the requested window size.
    """
    C, H, W = feat.shape

    if H < scale or W < scale:
        return None

    # unfold extracts all sliding windows in one go
    # (C, H, W) → (C, n_rows, scale) → (C, n_rows, n_cols, scale, scale)
    patches = feat.unfold(1, scale, _STRIDE).unfold(2, scale, _STRIDE)
    n_rows, n_cols = patches.shape[1], patches.shape[2]
    num_patches = n_rows * n_cols

    if num_patches == 0:
        return None

    # Reshape to (num_patches, C, scale, scale) for batch pooling
    patches = patches.contiguous().view(C, num_patches, scale, scale)
    patches = patches.permute(1, 0, 2, 3)  # (num_patches, C, scale, scale)

    # Single batched adaptive_max_pool2d call instead of per-patch loop
    pooled = F.adaptive_max_pool2d(patches, (_POOL_OUTPUT, _POOL_OUTPUT))
    # (num_patches, C, 7, 7)

    return pooled.reshape(num_patches, -1)  # (num_patches, C*7*7)


def spp(feat: torch.Tensor):
    """Spatial Pyramid Pooling over 7 predefined scales.

    Args:
        feat: (C, H, W) feature map for a single image.
              Expected to come from VGG16 features, so C=512.

    Returns:
        List of tensors (one per scale that fits the input), each of shape
        (num_patches, C*7*7).  Scales that are larger than the input are
        silently skipped.
    """
    return [
        r for r in
        (_extract_patches_at_scale(feat, s) for s in _SPP_SCALES)
        if r is not None
    ]


class SPPNet(nn.Module):
    """Wraps the SPP function as an nn.Module.

    Input : (C, H, W) feature map for a single image (no batch dimension).
    Output: (total_patches, C*7*7) tensor — all patches across all 7 scales
            concatenated along dim 0.

    Usage in a larger pipeline (loop over images in a batch):
        spp_net = SPPNet()
        patches = spp_net(feat)   # feat: (C, H, W)
    """

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (C, H, W) single-image feature map.
        Returns:
            (total_patches, C*7*7) tensor.
        """
        scale_tensors = spp(feat)
        if not scale_tensors:
            raise ValueError(
                f"Feature map ({feat.shape}) is smaller than the smallest "
                f"SPP scale ({_SPP_SCALES[0]}). Check input resolution."
            )
        return torch.cat(scale_tensors, dim=0)   # (total_patches, C*7*7)

