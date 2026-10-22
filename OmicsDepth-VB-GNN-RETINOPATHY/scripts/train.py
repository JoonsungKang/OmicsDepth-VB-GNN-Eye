"""Training loop implementing SI Algorithm 1 (VB-GNN for dependency-aware causal discovery).

Per iteration:
  1. sample a relaxed graph from q_phi(Z) (Gumbel-Softmax);
  2. EGS structural loss  lambda * D * beta * Tr(Pi^T Sigma Pi);
  3. likelihood through the graph-masked structural-equation decoder;
  4. minimax step: encoder minimises  -ELBO + L_struct + L_adv,
     discriminator maximises L_adv;
  5. projection of Pi onto M_eps = [eps, 1 - eps]^{d x d}.
After convergence: Bayesian FDR control on the posterior edge probabilities.

Run from the repository root:  python -m scripts.train
"""
from dataclasses import dataclass

import numpy as np
import torch

from models.discriminator import AdversarialDiscriminator, adversarial_loss
from models.vbgnn import VBGNN
from utils.fdr_control import bayesian_fdr_edges
from utils.metrics import EGSRegularizer


@dataclass
class TrainConfig:
    hidden_dim: int = 16
    epochs: int = 1000
    lr: float = 5e-3            # alpha
    lr_disc: float = 1e-3
    lambda_reg: float = 0.01    # lambda
    beta: float = 1.0           # dependency multiplier
    gamma: float = 1.0          # soft acyclicity weight
    tau: float = 0.5            # Gumbel temperature
    clip_eps: float = 1e-3      # clipping level of M_eps
    delta: float = 1e-3         # discriminator output clipping
    adv_weight: float = 1.0
    fdr_q: float = 0.05
    rho_high: float = 0.7
    rho_low: float = 0.05
    egs_normalized: bool = False
    seed: int = 0
    log_every: int = 100


def train_vbgnn(X: np.ndarray, prior_adj: np.ndarray, covariates: np.ndarray = None,
                cov_matrix: np.ndarray = None, cfg: TrainConfig = None, device: str = None,
                verbose: bool = True):
    """Fit VB-GNN.

    Args:
        X: [n, d] preprocessed (residualised, standardised) features.
        prior_adj: [d, d] {0,1} prior adjacency (edge i -> j at [i, j]).
        covariates: [n, k] nuisance covariates (batch, age, sex, PMI ...). If None,
            the adversarial term is skipped.
        cov_matrix: [d, d] positive-definite feature covariance Sigma. If None, the
            empirical covariance of X with a small ridge is used.

    Returns:
        dict with the fitted model, posterior edge probabilities Pi, the
        FDR-controlled edge mask S, and the loss history.
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n, d = X.shape
    if cov_matrix is None:
        cov_matrix = np.cov(X, rowvar=False) + 1e-3 * np.eye(d)

    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Sigma = torch.as_tensor(cov_matrix, dtype=torch.float32, device=device)
    A = torch.as_tensor(prior_adj, dtype=torch.float32, device=device)

    model = VBGNN(d, cfg.hidden_dim, A, clip_eps=cfg.clip_eps,
                  rho_high=cfg.rho_high, rho_low=cfg.rho_low).to(device)
    egs = EGSRegularizer(beta=cfg.beta, lambda_reg=cfg.lambda_reg, normalized=cfg.egs_normalized)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    use_adv = covariates is not None
    if use_adv:
        C = torch.as_tensor(covariates, dtype=torch.float32, device=device)
        if C.ndim == 1:
            C = C[:, None]
        disc = AdversarialDiscriminator(d, C.shape[1], delta=cfg.delta).to(device)
        opt_d = torch.optim.Adam(disc.parameters(), lr=cfg.lr_disc)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        # ---- Steps 1-3: sample graph, structural loss, likelihood
        recon, _, z_causal = model(Xt, tau=cfg.tau)
        nll = model.neg_log_likelihood(Xt, recon)
        kl = model.kl_divergence(gamma=cfg.gamma) / n          # per-sample ELBO scaling
        l_struct = egs(model.edge_probs(), Sigma) / n
        neg_elbo = nll + kl
        loss = neg_elbo + l_struct
        l_adv = torch.tensor(0.0, device=device)
        if use_adv:
            l_adv = adversarial_loss(disc, z_causal, C)
            loss = loss + cfg.adv_weight * l_adv

        # ---- Step 4: encoder (phi, Pi) descent
        opt.zero_grad()
        loss.backward()
        opt.step()
        # ---- projection onto M_eps
        model.project_()

        # ---- Step 4 (cont.): discriminator ascent on L_adv
        if use_adv:
            with torch.no_grad():
                _, _, z_c = model(Xt, tau=cfg.tau)
            l_d = -adversarial_loss(disc, z_c, C)
            opt_d.zero_grad()
            l_d.backward()
            opt_d.step()

        history.append(dict(epoch=epoch + 1, total=loss.item(), nll=nll.item(), kl=kl.item(),
                            struct=l_struct.item(), adv=l_adv.item()))
        if verbose and (epoch + 1) % cfg.log_every == 0:
            h = history[-1]
            print(f"Epoch [{h['epoch']}/{cfg.epochs}] total {h['total']:.4f} | nll {h['nll']:.4f} "
                  f"| kl {h['kl']:.4f} | struct {h['struct']:.4f} | adv {h['adv']:.4f}")

    # ---- Step 5: Bayesian FDR control after convergence
    Pi = model.edge_probs().detach().cpu().numpy()
    S = bayesian_fdr_edges(Pi, q=cfg.fdr_q)
    return dict(model=model, Pi=Pi, edges=S, history=history)


def noisy_prior(true_adj: np.ndarray, recall: float = 0.7, false_rate: float = 0.02, seed: int = 0) -> np.ndarray:
    """Simulated curated prior: keeps a fraction of true edges and adds false ones."""
    rng = np.random.default_rng(seed)
    keep = (true_adj > 0) & (rng.random(true_adj.shape) < recall)
    false = (true_adj == 0) & (rng.random(true_adj.shape) < false_rate)
    prior = (keep | false).astype(float)
    np.fill_diagonal(prior, 0.0)
    return prior


def run_training():
    """Demo on synthetic cyclic data with a batch confounder."""
    from data.preprocess import residualize, standardize_omics_data
    from data.synthetic import simulate_cyclic_sem
    from scripts.evaluate import evaluate_causal_graph

    X, A_true, batch = simulate_cyclic_sem(n=500, d=15, k_mean=2.0, n_cycles=2, seed=1)
    X = standardize_omics_data(residualize(X, batch))
    prior = noisy_prior(A_true, seed=1)
    out = train_vbgnn(X, prior, covariates=batch, cfg=TrainConfig(epochs=800, log_every=200))
    evaluate_causal_graph(A_true, out["Pi"], out["edges"])
    return out


if __name__ == "__main__":
    run_training()
