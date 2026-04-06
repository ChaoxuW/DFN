import numpy as np


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute average precision for a single class.

    Args:
        scores : (N,) float array of decision scores or probabilities.
        labels : (N,) int array of ground-truth labels in {0, 1}.

    Returns:
        AP as a float in [0, 1].
    """
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    n_pos = labels.sum()
    if n_pos == 0:
        return 0.0
    precision = tp / np.arange(1, len(labels) + 1)
    recall_delta = labels_sorted / n_pos
    return float((precision * recall_delta).sum())


def compute_mAP(
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list[str] | None = None,
) -> tuple[dict[str, float], float]:
    """Compute per-class AP and mAP.

    Args:
        scores      : (N, C) decision scores.
        labels      : (N, C) binary ground-truth.
        class_names : optional list of C class name strings.

    Returns:
        ap_dict : dict mapping class name (or index str) to AP value.
        mAP     : float, mean of all per-class APs.
    """
    C = scores.shape[1]
    if class_names is None:
        class_names = [str(c) for c in range(C)]
    ap_dict = {}
    for c, name in enumerate(class_names):
        ap_dict[name] = average_precision(scores[:, c], labels[:, c])
    mAP = float(np.mean(list(ap_dict.values())))
    return ap_dict, mAP