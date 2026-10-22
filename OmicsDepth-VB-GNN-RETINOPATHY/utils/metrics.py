"""Structural regularisers (SI, Section 1.2) and the soft acyclicity penalty (SI, Eq. prior)."""
import torch
import torch.nn as nn


class EGSRegularizer(nn.Module):
    """Entropy-based Gini-Simpson structural loss

        L_struct = lambda * D * beta * Tr(Pi^T Sigma Pi),   D = 1 - sum_ij pi_ij^2.

    Because the pi_ij are independent Bernoulli parameters (not a probability
    vector), D lies in [1 - d^2, 1] rather than [0, 1] (SI, Section 1.2).

    Args:
        normalized: if True, use D = 1 - mean_ij pi_ij^2, which lies in [0, 1].
            This is NOT the formulation analysed in the SI and is provided only
            for sensitivity analyses; the default (False) matches the manuscript.
    """

    def __init__(self, beta: float = 1.0, lambda_reg: float = 0.1, normalized: bool = False):
        super().__init__()
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.normalized = normalized

    @staticmethod
    def diversity(pi_matrix: torch.Tensor, normalized: bool = False) -> torch.Tensor:
        sq = pi_matrix ** 2
        return 1.0 - (sq.mean() if normalized else sq.sum())

    def forward(self, pi_matrix: torch.Tensor, covariance_matrix: torch.Tensor) -> torch.Tensor:
        diversity = self.diversity(pi_matrix, self.normalized)
        trace_term = torch.trace(pi_matrix.t() @ covariance_matrix @ pi_matrix)
        return self.lambda_reg * diversity * self.beta * trace_term


class ShannonRegularizer(nn.Module):
    """Baseline lambda * H_Sh(Pi) used in the SI comparison (Proposition 1, Step 1)."""

    def __init__(self, lambda_reg: float = 0.1):
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, pi_matrix: torch.Tensor) -> torch.Tensor:
        p = pi_matrix
        h = -(p * torch.log(p) + (1.0 - p) * torch.log1p(-p))
        return self.lambda_reg * h.sum()


def acyclicity_penalty(pi_matrix: torch.Tensor) -> torch.Tensor:
    """h(Pi) = Tr(exp(Pi o Pi)) - d; equals 0 iff the weighted graph is acyclic."""
    d = pi_matrix.shape[0]
    return torch.trace(torch.matrix_exp(pi_matrix * pi_matrix)) - d
