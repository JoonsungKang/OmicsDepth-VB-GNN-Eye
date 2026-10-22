"""Counterfactual in silico intervention (main text, Section 5; SI, Section 2.4).

Pearl's three-step counterfactual on the fitted structural equations:
  1. abduction  - recover each sample's noise  eps = x - f(x_Pa);
  2. action     - set do(x_k = value) for the intervened features (e.g. succinate);
  3. prediction - propagate through the learned graph with the same noise.
Because feedback loops are allowed, prediction solves the equilibrium
x = f(x_Pa) + eps by fixed-point iteration; non-convergence is reported.
"""
import numpy as np
import torch


@torch.no_grad()
def do_intervention(model, x_factual: np.ndarray, intervene_idx, intervene_values,
                    edge_mask: np.ndarray = None, max_iter: int = 500, tol: float = 1e-6):
    """
    Args:
        model: fitted VBGNN.
        x_factual: [n, d] preprocessed observations (e.g. disease donors).
        intervene_idx: list of feature indices to intervene on.
        intervene_values: scalar or array broadcastable to [n, len(intervene_idx)].
        edge_mask: [d, d] boolean graph to use (e.g. FDR-selected edges);
            if None, the posterior edge probabilities Pi are used as weights.
    Returns:
        (x_counterfactual [n, d], converged: bool, n_iter: int)
    """
    dev = next(model.parameters()).device
    x = torch.as_tensor(x_factual, dtype=torch.float32, device=dev)
    if edge_mask is None:
        z = model.edge_probs()
    else:
        z = torch.as_tensor(edge_mask, dtype=torch.float32, device=dev) * model.offdiag
    idx = torch.as_tensor(list(intervene_idx), dtype=torch.long, device=dev)
    vals = torch.as_tensor(intervene_values, dtype=torch.float32, device=dev).expand(x.shape[0], len(idx))

    recon, _ = model.decode(x, z)
    eps = x - recon                                   # abduction
    x_cf = x.clone()
    x_cf[:, idx] = vals                               # action
    converged = False
    it = 0
    for it in range(1, max_iter + 1):                 # prediction (fixed point)
        nxt, _ = model.decode(x_cf, z)
        nxt = nxt + eps
        nxt[:, idx] = vals
        delta = (nxt - x_cf).abs().max().item()
        x_cf = nxt
        if delta < tol:
            converged = True
            break
    return x_cf.cpu().numpy(), converged, it


def restoration_percent(x_factual: np.ndarray, x_cf: np.ndarray, target_idx, control_mean: np.ndarray = None) -> float:
    """Restoration of target features (e.g. RGC survival genes).

    If control_mean is given: percentage of the disease-control gap closed,
        100 * mean_k (x_cf - x_fact) / (control - x_fact), averaged over samples.
    Otherwise: mean change of the targets in units of the factual standard deviation (x100).
    """
    t = list(target_idx)
    xf, xc = np.asarray(x_factual)[:, t], np.asarray(x_cf)[:, t]
    if control_mean is not None:
        gap = np.asarray(control_mean)[t][None, :] - xf
        ok = np.abs(gap) > 1e-8
        return float(100.0 * np.mean(((xc - xf)[ok]) / gap[ok]))
    sd = xf.std(axis=0) + 1e-8
    return float(100.0 * np.mean((xc - xf) / sd))


def run_counterfactual_demo():
    """Demo: train on synthetic data, normalise one upstream node, measure downstream restoration."""
    from scripts.train import run_training

    out = run_training()
    model, S = out["model"], out["edges"]
    from data.synthetic import simulate_cyclic_sem
    from data.preprocess import residualize, standardize_omics_data
    X, A_true, batch = simulate_cyclic_sem(n=500, d=15, k_mean=2.0, n_cycles=2, seed=1)
    X = standardize_omics_data(residualize(X, batch))
    src = int(np.argmax(S.sum(axis=1)))               # node with most selected children
    children = list(np.flatnonzero(S[src]))
    if not children:
        print("No selected children for the intervened node; nothing to evaluate.")
        return
    x_cf, conv, it = do_intervention(model, X, [src], 0.0, edge_mask=S)
    print(f"do(x_{src} = 0): converged={conv} after {it} iterations; "
          f"mean standardized change of children {children}: {restoration_percent(X, x_cf, children):.1f}%")


if __name__ == "__main__":
    run_counterfactual_demo()
