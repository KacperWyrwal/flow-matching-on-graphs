"""
Loss functions for flow matching on graphs.

Combines rate KL loss from meta_fm/train.py and config_fm/loss.py.
"""

import torch


def rate_kl_divergence(target, predicted, eps=1e-8):
    """
    Generalized KL divergence between rate vectors (Bregman divergence with
    phi(x) = x log x - x). This is the principled loss for matching CTMC path
    measures -- the discrete analogue of the L2 velocity loss in Euclidean flow
    matching.

    D(r || r_theta) = sum_{b!=a} [r * log(r / r_theta) - r + r_theta]

    Non-negative, zero iff r == r_theta.

    Args:
        target:    (*, N, N) target rate matrices (off-diagonal entries used)
        predicted: (*, N, N) predicted rate matrices
        eps:       floor for numerical stability
    Returns:
        (*,) per-sample loss summed over off-diagonal entries
    """
    N = target.shape[-1]
    off_diag = ~torch.eye(N, dtype=torch.bool, device=target.device)
    r       = target[..., off_diag].clamp(min=eps)
    r_theta = predicted[..., off_diag].clamp(min=eps)
    return (r * torch.log(r / r_theta) - r + r_theta).sum(dim=-1)


def mse_loss(target, predicted):
    """MSE between off-diagonal rate matrix entries (previous default loss)."""
    N = target.shape[-1]
    off_diag = ~torch.eye(N, dtype=torch.bool, device=target.device)
    return ((target[..., off_diag] - predicted[..., off_diag]) ** 2).sum(dim=-1)


def rate_kl_loss(pred_rates, target_rates, mask):
    """Rate KL divergence loss.

    D(r || r_theta) = sum [r log(r/r_theta) - r + r_theta]

    Summed over valid transitions (mask > 0), normalized by mask count.
    """
    eps = 1e-10
    active = (target_rates > eps) & (mask > 0)

    loss_active = (target_rates[active]
                   * torch.log(target_rates[active]
                               / (pred_rates[active] + eps))
                   - target_rates[active]
                   + pred_rates[active])

    # Penalize nonzero predictions where target is 0
    inactive = (~active) & (mask > 0)
    loss_inactive = pred_rates[inactive]

    n_valid = mask.sum().clamp(min=1.0)
    return (loss_active.sum() + loss_inactive.sum()) / n_valid
