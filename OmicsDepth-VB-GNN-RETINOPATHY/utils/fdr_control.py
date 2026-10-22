"""False discovery rate control.

* bayesian_fdr_edges: used for causal edges (SI, Section 1.2 and Algorithm 1,
  Step 5). The variational posterior yields edge probabilities, not p-values, so
  1 - pi_ij is treated as a local FDR and edges are selected by the Bayesian FDR
  rule of Newton et al. (2004).
* structural_benjamini_hochberg: classical BH step-up procedure on p-values,
  used for the donor-level differential tests of main-text Table 2.
"""
import numpy as np


def bayesian_fdr_edges(pi_matrix: np.ndarray, q: float = 0.05, exclude_diagonal: bool = True) -> np.ndarray:
    """Select the largest set S of edges, taken in decreasing order of pi_ij,
    whose mean posterior probability of absence (1 - pi_ij) is <= q.

    Returns a boolean adjacency mask with the same shape as pi_matrix.
    """
    pi = np.asarray(pi_matrix, dtype=float)
    mask_valid = np.ones_like(pi, dtype=bool)
    if exclude_diagonal and pi.ndim == 2 and pi.shape[0] == pi.shape[1]:
        np.fill_diagonal(mask_valid, False)
    flat_idx = np.flatnonzero(mask_valid)
    lfdr = 1.0 - pi.ravel()[flat_idx]
    order = np.argsort(lfdr, kind="stable")
    running_mean = np.cumsum(lfdr[order]) / np.arange(1, len(order) + 1)
    ok = np.flatnonzero(running_mean <= q)
    selected = np.zeros(pi.size, dtype=bool)
    if ok.size:
        selected[flat_idx[order[: ok[-1] + 1]]] = True
    return selected.reshape(pi.shape)


def structural_benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up procedure on a 1-D array of p-values.

    Returns a boolean mask of rejected hypotheses (same order as the input).
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below_threshold = sorted_p_values <= thresholds
    if not np.any(below_threshold):
        return np.zeros(n, dtype=bool)
    max_idx = np.max(np.where(below_threshold)[0])
    significant_mask = np.zeros(n, dtype=bool)
    significant_mask[sorted_indices[: max_idx + 1]] = True
    return significant_mask
