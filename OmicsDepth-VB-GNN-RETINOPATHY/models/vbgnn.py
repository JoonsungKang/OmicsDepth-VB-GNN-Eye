"""VB-GNN model (Supplementary Information, Sections 1.1 and 1.3, Algorithm 1).

Variational posterior over a directed graph (cycles permitted):
    q_phi(Z) = prod_{i,j} Bernoulli(pi_ij)
relaxed with Gumbel-Softmax, and a graph-masked non-linear structural-equation
decoder
    x_j = f_{theta_j}(x_{Pa_j}) + eps_j,   eps_j ~ N(0, sigma_j^2).

The edge probabilities pi_ij are optimised directly and projected onto the
clipped domain M_eps = [eps, 1 - eps]^{d x d} after every update (Algorithm 1),
which is the setting in which the gradient bounds of Proposition 1 hold.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import acyclicity_penalty


class VBGNN(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        prior_adj: torch.Tensor,
        clip_eps: float = 1e-3,
        rho_high: float = 0.7,
        rho_low: float = 0.05,
    ):
        """
        Args:
            num_features: number of molecular features d.
            hidden_dim: hidden units per node in the structural-equation decoder.
            prior_adj: {0,1}^{d x d} prior adjacency from curated databases
                (e.g. KEGG); entry (i, j) = 1 means a supported edge i -> j.
            clip_eps: clipping level eps of the domain M_eps (SI, Proposition 1).
            rho_high / rho_low: Bernoulli prior probabilities for supported /
                unsupported edges (SI, Eq. prior). Must lie in (0, 1).
        """
        super().__init__()
        if not (0.0 < clip_eps < 0.5):
            raise ValueError("clip_eps must lie in (0, 1/2)")
        if not (0.0 < rho_low < rho_high < 1.0):
            raise ValueError("require 0 < rho_low < rho_high < 1")
        d = num_features
        self.num_features = d
        self.clip_eps = clip_eps

        prior_adj = prior_adj.float()
        # Algorithm 1: pi_ij <- Sigmoid([A_prior]_ij), projected onto M_eps
        self.pi = nn.Parameter(torch.sigmoid(prior_adj).clamp(clip_eps, 1.0 - clip_eps))
        rho = torch.where(prior_adj > 0, torch.full_like(prior_adj, rho_high),
                          torch.full_like(prior_adj, rho_low))
        self.register_buffer("rho", rho)
        self.register_buffer("offdiag", 1.0 - torch.eye(d))  # no self-loops

        # Decoder: W1[j, i, :] maps parent i of node j into node j's hidden layer
        self.W1 = nn.Parameter(torch.randn(d, d, hidden_dim) / math.sqrt(d))
        self.b1 = nn.Parameter(torch.zeros(d, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(d, hidden_dim) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(d))
        self.log_sigma = nn.Parameter(torch.zeros(d))  # noise s.d. sigma_j

    # ------------------------------------------------------------------ graph
    def edge_probs(self) -> torch.Tensor:
        """Posterior edge probabilities Pi (diagonal masked to 0)."""
        return self.pi * self.offdiag

    @torch.no_grad()
    def project_(self) -> None:
        """Projection onto M_eps (Algorithm 1)."""
        self.pi.clamp_(self.clip_eps, 1.0 - self.clip_eps)

    def sample_gumbel_softmax(self, tau: float = 0.5) -> torch.Tensor:
        """Binary Concrete relaxation (SI, Eq. Gumbel):
        Z_ij = Sigmoid((ln pi - ln(1 - pi) + g1 - g2) / tau)."""
        pi = self.pi
        u1 = torch.rand_like(pi).clamp(1e-10, 1.0 - 1e-10)
        u2 = torch.rand_like(pi).clamp(1e-10, 1.0 - 1e-10)
        g1 = -torch.log(-torch.log(u1))
        g2 = -torch.log(-torch.log(u2))
        logits = torch.log(pi) - torch.log1p(-pi)
        return torch.sigmoid((logits + g1 - g2) / tau) * self.offdiag

    # ---------------------------------------------------------------- decoder
    def decode(self, x: torch.Tensor, z: torch.Tensor):
        """Structural equations for all nodes in parallel.

        h[n, j, :] = ReLU( sum_i x[n, i] * z[i, j] * W1[j, i, :] + b1[j] )
        out[n, j]  = h[n, j, :] . w2[j] + b2[j]
        """
        h = F.relu(torch.einsum("ni,ij,jim->njm", x, z, self.W1) + self.b1)
        out = torch.einsum("njm,jm->nj", h, self.w2) + self.b2
        return out, h

    def forward(self, x: torch.Tensor, tau: float = 0.5):
        """Returns (reconstruction, relaxed adjacency, z_causal).

        z_causal is the per-sample representation obtained by propagating x
        through the sampled graph (mean over hidden units, shape n x d); it is
        the causal latent passed to the adversarial discriminator and depends
        on Pi through the relaxation.
        """
        z = self.sample_gumbel_softmax(tau)
        recon, h = self.decode(x, z)
        return recon, z, h.mean(dim=-1)

    # ----------------------------------------------------------------- losses
    def neg_log_likelihood(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """-ln p(X | Z; theta) for Gaussian noise, averaged over samples."""
        log_var = 2.0 * self.log_sigma
        nll = 0.5 * ((x - recon) ** 2 * torch.exp(-log_var) + log_var + math.log(2 * math.pi))
        return nll.sum(dim=1).mean()

    def kl_divergence(self, gamma: float = 1.0) -> torch.Tensor:
        """KL(q_phi(Z) || p(Z)) for the Bernoulli prior with soft acyclicity
        penalty gamma * h(Pi) (SI, Eq. prior; plug-in evaluation of E_q[h])."""
        pi, rho = self.pi, self.rho
        kl = pi * (torch.log(pi) - torch.log(rho)) + (1.0 - pi) * (torch.log1p(-pi) - torch.log1p(-rho))
        return (kl * self.offdiag).sum() + gamma * acyclicity_penalty(self.edge_probs())
