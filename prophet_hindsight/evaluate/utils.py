import numpy as np
from typing import Literal


def _bin_stats(probs: list[float], labels: list[float], weights: list[float], n_bins: int, strategy: Literal["uniform", "quantile"]) \
    -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function to calculate the bin statistics for the calibration metric.

    :param probs: List of probabilities.
    :param labels: List of labels.
    :param weights: List of weights.
    :param n_bins: Number of bins.
    :param strategy: Strategy to use for discretization.

    :returns: A tuple of bin centers, confidence, accuracy, and counts.
    """
    probs, labels, weights = np.asarray(probs, dtype=float), np.asarray(labels, dtype=float), np.asarray(weights, dtype=float)
    assert probs.shape == labels.shape and probs.shape == weights.shape and probs.ndim == 1, "probs, labels, and weights must have the same shape"

    # we check that the weights need to sum to equal the length of the input arrays
    assert np.isclose(weights.sum(), len(probs)), f"weights need to sum to equal the length of the input arrays, but got {weights.sum()} != {len(probs)}"
    
    n = len(probs)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:  # strategy == "quantile"
        # unique quantiles so we don't create empty bins when duplicates occur
        q = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(probs, q))
        # ensure at least 2 edges (if all probs identical)
        if edges.size < 2:
            edges = np.array([probs.min(), probs.max() + 1e-12])

    # assign to bins; `right=False` so the left edge is inclusive, right edge exclusive
    bin_ids = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, len(edges) - 2)

    counts, conf, acc = [], [], []
    bin_left, bin_right = edges[:-1], edges[1:]

    for b in range(len(edges)-1):
        idx = (bin_ids == b)
        if not np.any(idx):
            counts.append(0)
            conf.append(np.nan)
            acc.append(np.nan)
        else:
            counts.append(int(weights[idx].sum()))
            conf.append((probs[idx] * weights[idx]).sum() / weights[idx].sum())
            acc.append((labels[idx] * weights[idx]).sum() / weights[idx].sum())

    counts, conf, acc = np.asarray(counts), np.asarray(conf), np.asarray(acc)

    # Bin centers & widths for plotting
    bin_centers = 0.5 * (bin_left + bin_right)
    bin_widths = bin_right - bin_left

    return bin_centers, bin_widths, conf, acc, counts


def _calculate_ece(conf: np.ndarray, acc: np.ndarray, counts: np.ndarray, total_samples: int) -> float:
    """
    Calculate the Expected Calibration Error (ECE) from bin statistics.
    
    :param conf: Confidence (average predicted probability) for each bin.
    :param acc: Accuracy (fraction of correct predictions) for each bin.
    :param counts: Number of samples in each bin.
    :param total_samples: Total number of samples.
    
    :returns: The ECE score.
    """
    valid_mask = counts > 0
    ece_score = np.sum(np.abs(conf[valid_mask] - acc[valid_mask]) * (counts[valid_mask] / total_samples))
    return ece_score