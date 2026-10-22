"""Smoke tests. Run from the repository root:  python -m tests.test_smoke  (or: pytest tests)

NumPy-only components are always tested; PyTorch components are tested when torch is installed.
"""
import importlib.util

import numpy as np

from data.preprocess import pseudobulk_aggregate, residualize, shrinkage_covariance, standardize_omics_data
from data.synthetic import simulate_cyclic_sem
from scripts.evaluate import evaluate_causal_graph, structural_hamming_distance
from utils.fdr_control import bayesian_fdr_edges, structural_benjamini_hochberg

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_synthetic_has_cycle():
    X, A, batch = simulate_cyclic_sem(n=200, d=10, n_cycles=1, seed=0)
    assert X.shape == (200, 10) and A.shape == (10, 10) and batch.shape == (200, 1)
    assert np.isfinite(X).all()
    assert ((A * A.T) > 0).any(), "expected at least one 2-cycle"


def test_bayesian_fdr():
    pi = np.array([[0.0, 0.99, 0.2], [0.97, 0.0, 0.5], [0.1, 0.9, 0.0]])
    S = bayesian_fdr_edges(pi, q=0.05)
    # lfdr sorted: 0.01, 0.03, 0.10 (running mean 0.047 <= 0.05), next 0.5 -> stop
    assert S[0, 1] and S[1, 0] and S[2, 1] and not S[1, 2] and S.sum() == 3
    sel = 1 - pi[S]
    assert sel.mean() <= 0.05 and not S.diagonal().any()


def test_bh():
    p = np.array([0.001, 0.2, 0.01, 0.04, 0.9])
    assert structural_benjamini_hochberg(p, 0.05).tolist() == [True, False, True, False, False]


def test_shd_and_eval():
    T = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    P = np.array([[0, 0, 0], [1, 0, 1], [1, 0, 0]])     # one reversed, one extra
    assert structural_hamming_distance(T, P) == 2
    r = evaluate_causal_graph(T, P.astype(float), P.astype(bool), verbose=False)
    assert 0 <= r["pr_auc"] <= 1 and r["shd"] == 2 and abs(r["fdr"] - 2 / 3) < 1e-9


def test_preprocess():
    rng = np.random.default_rng(0)
    counts = rng.poisson(5, size=(60, 4))
    donors = np.repeat(np.arange(6), 10)
    cts = np.tile(["RGC", "Muller"], 30)
    M, order, labels = pseudobulk_aggregate(counts, donors, cts)
    assert M.shape == (6, 8) and len(labels) == 8
    X = rng.normal(size=(50, 5))
    C = rng.normal(size=(50, 2))
    R = residualize(X + C @ rng.normal(size=(2, 5)), C)
    assert np.abs(np.corrcoef(np.column_stack([R, C]), rowvar=False)[:5, 5:]).max() < 1e-8
    S = shrinkage_covariance(standardize_omics_data(X))
    assert np.linalg.eigvalsh(S).min() > 0


def test_decoder_shapes_numpy_mirror():
    """Mirror of VBGNN.decode in NumPy (same einsum strings) to check shapes/masking."""
    n, d, m = 7, 5, 3
    rng = np.random.default_rng(0)
    x, z, W1 = rng.normal(size=(n, d)), rng.random((d, d)), rng.normal(size=(d, d, m))
    z[2, :] = 0.0                                        # node 2 has no children
    h = np.maximum(np.einsum("ni,ij,jim->njm", x, z, W1), 0)
    out = np.einsum("njm,jm->nj", h, rng.normal(size=(d, m)))
    assert out.shape == (n, d)
    x2 = x.copy(); x2[:, 2] += 100.0                    # changing node 2 must not affect anything
    h2 = np.maximum(np.einsum("ni,ij,jim->njm", x2, z, W1), 0)
    assert np.allclose(h, h2)


def test_torch_training_runs():
    if not HAS_TORCH:
        return "skipped (torch not installed)"
    from scripts.train import TrainConfig, noisy_prior, train_vbgnn
    from scripts.simulate_cf import do_intervention
    X, A, batch = simulate_cyclic_sem(n=150, d=8, seed=0)
    out = train_vbgnn(standardize_omics_data(X), noisy_prior(A), covariates=batch,
                      cfg=TrainConfig(epochs=30), verbose=False)
    Pi = out["Pi"]
    eps = 1e-3
    off = ~np.eye(8, dtype=bool)
    assert (Pi[off] >= eps - 1e-7).all() and (Pi[off] <= 1 - eps + 1e-7).all()
    x_cf, conv, _ = do_intervention(out["model"], X, [0], 0.0)
    assert x_cf.shape == X.shape and np.allclose(x_cf[:, 0], 0.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            result = fn()
            print(f"SKIP {name}: {result}" if isinstance(result, str) else f"PASS {name}")
