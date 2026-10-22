"""Causal graph evaluation (main text, Section 3): AUROC, PR-AUC, FDR and SHD."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def structural_hamming_distance(true_adj: np.ndarray, pred_adj: np.ndarray) -> int:
    """Directed SHD: missing + extra edges, with a reversed edge counted once."""
    T = np.asarray(true_adj) > 0
    P = np.asarray(pred_adj) > 0
    np.fill_diagonal(T, False)
    np.fill_diagonal(P, False)
    diff = T ^ P
    # an edge i->j in truth predicted as j->i shows up twice in diff; count it once
    reversed_pairs = T & ~P & P.T & ~T.T
    return int(diff.sum() - reversed_pairs.sum())


def evaluate_causal_graph(true_adj: np.ndarray, pred_prob: np.ndarray, pred_mask: np.ndarray = None,
                          verbose: bool = True) -> dict:
    """
    Args:
        true_adj: [d, d] {0,1} ground-truth adjacency.
        pred_prob: [d, d] posterior edge probabilities (scores).
        pred_mask: [d, d] boolean selected edges (e.g. after Bayesian FDR); optional.
    Self-loops are excluded from all metrics.
    """
    true_adj = np.asarray(true_adj)
    off = ~np.eye(true_adj.shape[0], dtype=bool)
    y_true = (true_adj[off] > 0).astype(int)
    y_score = np.asarray(pred_prob)[off]
    res = dict(auroc=roc_auc_score(y_true, y_score), pr_auc=average_precision_score(y_true, y_score))
    if pred_mask is not None:
        m = np.asarray(pred_mask, dtype=bool)
        sel = m[off]
        n_sel = int(sel.sum())
        res["n_selected"] = n_sel
        res["fdr"] = float(((sel == 1) & (y_true == 0)).sum() / n_sel) if n_sel else 0.0
        res["shd"] = structural_hamming_distance(true_adj, m)
    if verbose:
        print("Evaluation Metrics -> " + " | ".join(
            f"{k.upper()}: {v:.4f}" if isinstance(v, float) else f"{k.upper()}: {v}" for k, v in res.items()))
    return res
