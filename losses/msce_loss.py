import torch
import torch.nn as nn


class MultiLabelSigmoidLoss(nn.Module):
    """Multi-label sigmoid cross-entropy loss.

    For each sample i and class c:
        Loss(s_i, y_i) = -sum_{c=1}^{C} [ y_ic * log(sigma(s_ic))
                                         + (1 - y_ic) * log(1 - sigma(s_ic)) ]
    """

    def __init__(self, reduction: str = "mean"):
        super(MultiLabelSigmoidLoss, self).__init__()
        # BCEWithLogitsLoss applies sigmoid internally and is numerically stable
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (N, C) raw scores s_ic (before sigmoid)
            targets: (N, C) binary labels y_ic in {0, 1}
        Returns:
            scalar loss
        """
        return self.bce(logits, targets.float())