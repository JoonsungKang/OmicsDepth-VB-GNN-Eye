"""Adversarial discriminator for nuisance decoupling (SI, Section 1.1).

D_psi(z_causal, z_nuisance) -> [delta, 1 - delta] estimates the probability that a
pair comes from the joint posterior Q rather than from the factorised target T
(product of marginals, sampled by shuffling z_nuisance within the batch).
Layers are spectrally normalised (Lipschitz constraint, assumption A3) and the
output is clipped to [delta, 1 - delta] so that ln D and ln(1 - D) stay finite.
"""
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class AdversarialDiscriminator(nn.Module):
    def __init__(self, causal_dim: int, nuisance_dim: int, hidden_dim: int = 64, delta: float = 1e-3):
        super().__init__()
        if not (0.0 < delta < 0.5):
            raise ValueError("delta must lie in (0, 1/2)")
        self.delta = delta
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(causal_dim + nuisance_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, 1)),
            nn.Sigmoid(),
        )

    def forward(self, z_causal: torch.Tensor, z_nuisance: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([z_causal, z_nuisance], dim=-1)).squeeze(-1)
        return out.clamp(self.delta, 1.0 - self.delta)


def adversarial_loss(disc: AdversarialDiscriminator, z_causal: torch.Tensor, z_nuisance: torch.Tensor) -> torch.Tensor:
    """L_adv = E_Q[ln D(z_c, z_n)] + E_T[ln(1 - D(z_c, z_n'))], with z_n' shuffled.

    The discriminator maximises L_adv; the VB-GNN encoder minimises it.
    """
    perm = torch.randperm(z_nuisance.shape[0], device=z_nuisance.device)
    d_joint = disc(z_causal, z_nuisance)
    d_prod = disc(z_causal, z_nuisance[perm])
    return torch.log(d_joint).mean() + torch.log(1.0 - d_prod).mean()
