"""Synthetic cyclic non-linear SEM with a batch confounder (main text, Section 3).

Data are drawn from the equilibrium of
    x = tanh(x W) + batch_effect + noise
by fixed-point iteration; the true graph may contain feedback loops.
"""
import numpy as np


def simulate_cyclic_sem(n: int = 500, d: int = 20, k_mean: float = 2.0, n_cycles: int = 2,
                        noise: float = 0.1, dropout: float = 0.0, batch_strength: float = 0.5,
                        seed: int = 0, max_iter: int = 500):
    """Returns (X [n x d], true_adj [d x d] {0,1}, covariates [n x 1] batch label)."""
    rng = np.random.default_rng(seed)
    p = min(k_mean / max(d - 1, 1), 1.0)
    # random DAG (upper triangular in a random order) ...
    A = np.triu(rng.random((d, d)) < p / 2, k=1).astype(float)
    # ... plus explicit feedback loops (back edges)
    for _ in range(n_cycles):
        i, j = sorted(rng.choice(d, size=2, replace=False))
        A[i, j] = 1.0
        A[j, i] = 1.0
    perm = rng.permutation(d)
    A = A[np.ix_(perm, perm)]
    # weights scaled so that the map is a contraction (unique equilibrium)
    W = A * rng.uniform(0.5, 1.0, size=(d, d)) * rng.choice([-1, 1], size=(d, d))
    radius = np.max(np.abs(np.linalg.eigvals(W))) if A.any() else 0.0
    if radius >= 0.9:
        W *= 0.9 / radius
    batch = rng.integers(0, 2, size=(n, 1)).astype(float)
    shift = batch_strength * rng.normal(size=(1, d))
    eps = noise * rng.normal(size=(n, d))
    X = np.zeros((n, d))
    for _ in range(max_iter):
        X_new = np.tanh(X @ W) + batch * shift + eps
        if np.max(np.abs(X_new - X)) < 1e-8:
            X = X_new
            break
        X = X_new
    if dropout > 0:
        X = X * (rng.random(X.shape) >= dropout)
    return X, A, batch
