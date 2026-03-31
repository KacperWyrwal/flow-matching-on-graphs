"""
Distribution-level sampling: posterior sampling via FiLM model.

From meta_fm/sample.py — the distribution-level sampling function.
"""

import numpy as np
import torch

from otfm.core.utils import get_device


def sample_posterior_film(
    model,
    mu_starts: np.ndarray,
    node_context: np.ndarray,
    global_cond: np.ndarray,
    edge_index,
    n_steps: int = 100,
    device=None,
) -> np.ndarray:
    """
    Run K posterior trajectories in parallel (batched) for FiLM model.

    Args:
        mu_starts:    (K, N) initial distributions
        node_context: (N, node_context_dim)
        global_cond:  (global_dim,)
        edge_index:   (2, E)

    Returns: (K, N) final distributions at t=0.999
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    K, N = mu_starts.shape
    times = np.linspace(0.0, 0.999, n_steps)
    dt = times[1] - times[0] if n_steps > 1 else 0.0

    # Broadcast context to batch dimension
    nctx = torch.tensor(node_context, dtype=torch.float32, device=device)   # (N, C)
    gctx = torch.tensor(global_cond,  dtype=torch.float32, device=device)   # (D,)
    nctx_b = nctx.unsqueeze(0).expand(K, -1, -1)                            # (K, N, C)
    gctx_b = gctx.unsqueeze(0).expand(K, -1)                                # (K, D)
    ei = edge_index.to(device)

    mu = torch.tensor(mu_starts, dtype=torch.float32, device=device)        # (K, N)

    with torch.no_grad():
        for k, t in enumerate(times[:-1]):
            t_b = torch.full((K, 1), t, dtype=torch.float32, device=device)
            R_pred = model.forward_batch(mu, t_b, nctx_b, gctx_b, ei)       # (K, N, N)
            R_scaled = R_pred / (1.0 - t)
            dp = torch.bmm(mu.unsqueeze(1), R_scaled).squeeze(1)            # (K, N)
            mu = mu + dt * dp
            mu = torch.clamp(mu, min=0.0)
            s = mu.sum(dim=1, keepdim=True).clamp(min=1e-15)
            mu = mu / s

    return mu.cpu().numpy()
