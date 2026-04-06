import numpy as np
import torch
from sklearn.svm import LinearSVC


class SVMHead:
    """One-vs-all linear SVM classifier on top of FisherNet features.

    Pipeline (train & test):
        1. For each image, apply 5 scale transforms and extract a Fisher vector
           per scale using the trained FisherNet (fc_out bypassed).
        2. Average the 5 Fisher vectors  →  multi-scale Fisher vector.
        3. Power-normalization : x ← sign(x) √|x|
        4. L2-normalization    : x ← x / ‖x‖₂
        5. Train / apply 20 independent one-vs-all LinearSVCs (C=1).
        6. At test time output a (20,) binary prediction per image.

    Args:
        fisher_net  : A trained FisherNet instance.  The last fc_out layer is
                      bypassed; only _forward_single() is used.
        num_classes : Number of classes (default 20 for VOC).
        C           : SVM regularisation parameter C_svm (default 1.0).
    """

    def __init__(self, fisher_net, num_classes: int = 20, C: float = 1.0):
        self.net = fisher_net
        self.net.eval()
        self.num_classes = num_classes
        # 20 independent one-vs-all LinearSVMs
        # dual=False: primal formulation, faster when n_samples >= n_features
        # (here D=16384, N~16551 so primal is preferred by sklearn docs)
        self.svms = [LinearSVC(C=C, dual=False, max_iter=10000) for _ in range(num_classes)]
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def power_normalize(x: torch.Tensor) -> torch.Tensor:
        """x ← sign(x) · √(|x| + ε)  (ε avoids Inf gradient at zero)"""
        return x.sign() * (x.abs() + 1e-12).sqrt()

    @staticmethod
    def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """x ← x / ‖x‖₂"""
        return x / (x.norm(p=2) + eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply power-normalization then L2-normalization."""
        x = self.power_normalize(x)
        x = self.l2_normalize(x)
        return x

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_fv(self, img, transforms: list) -> torch.Tensor:
        """Extract multi-scale Fisher vector for a single PIL image.

        Applies each of the 5 scale transforms, extracts a raw Fisher vector
        per scale via FisherNet._forward_single(), averages them, then applies
        power + L2 normalization.

        Args:
            img        : PIL.Image instance.
            transforms : list of 5 torchvision transforms, e.g. from
                         data.voc.get_svm_transforms_all_scales().

        Returns:
            fv : (2·K·D,) normalized Fisher vector (CPU torch.Tensor).
        """
        fvs = []
        for t in transforms:
            x = t(img)                                         # (3, H_s, W_s)
            # normalize=False: get raw Fisher vectors before any normalization,
            # so multi-scale averaging is done in the original feature space.
            fv = self.net._forward_single(x, normalize=False)  # (2·K·D,)
            fvs.append(fv)
        avg_fv = torch.stack(fvs, dim=0).mean(dim=0)          # (2·K·D,)
        # Apply power + L2 normalization exactly once on the averaged vector
        return self.normalize(avg_fv)

    # ------------------------------------------------------------------
    # SVM training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "SVMHead":
        """Train 20 one-vs-all LinearSVMs.

        Args:
            X : (N, D) float array of normalized Fisher vectors.
            Y : (N, 20) int array of binary labels {0, 1}.

        Returns:
            self
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.cpu().numpy()

        for c in range(self.num_classes):
            self.svms[c].fit(X, Y[:, c])
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels for each sample.

        Args:
            X : (N, D) float array of normalized Fisher vectors.

        Returns:
            preds : (N, 20) int array of predictions in {0, 1}.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        preds = np.stack(
            [self.svms[c].predict(X) for c in range(self.num_classes)],
            axis=1,
        )
        return preds.astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw SVM decision scores (useful for mAP evaluation).

        Args:
            X : (N, D) float array of normalized Fisher vectors.

        Returns:
            scores : (N, 20) float array of decision values.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before decision_function().")
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        return np.stack(
            [self.svms[c].decision_function(X) for c in range(self.num_classes)],
            axis=1,
        )