"""
Sampling trajectories from trained rate matrix predictors (graph-level).

From meta_fm/sample.py — the graph-level sampling functions.
"""

import numpy as np
import torch

from otfm.core.utils import get_device


def sample_trajectory_flexible(
    model,
    mu_start: np.ndarray,
    context: np.ndarray,
    edge_index,
    n_steps: int = 200,
    device=None,
) -> tuple:
    """
    Integrate the conditional flow for FlexibleConditionalGNNRateMatrixPredictor.

    Like sample_trajectory_conditional but uses model.forward_single and
    accepts edge_index for the specific graph topology.

    Args:
        mu_start:   (N,) initial distribution
        context:    (N, context_dim) per-node conditioning, fixed throughout
        edge_index: (2, E) LongTensor for this graph

    Returns: (times, trajectory)
        times:      np.ndarray (n_steps,)
        trajectory: np.ndarray (n_steps, N)
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    N = len(mu_start)
    times = np.linspace(0.0, 0.999, n_steps)
    dt = times[1] - times[0] if n_steps > 1 else 0.0

    ctx_tensor = torch.tensor(context, dtype=torch.float32, device=device)  # (N, ctx_dim)
    edge_index_dev = edge_index.to(device)

    trajectory = np.zeros((n_steps, N))
    mu = mu_start.copy().astype(float)

    with torch.no_grad():
        for k, t in enumerate(times):
            trajectory[k] = mu

            if k < n_steps - 1:
                mu_tensor = torch.tensor(mu, dtype=torch.float32, device=device)  # (N,)
                t_tensor = torch.tensor([t], dtype=torch.float32, device=device)  # (1,)

                R_pred = model.forward_single(mu_tensor, t_tensor, ctx_tensor, edge_index_dev)
                # Model predicts u_tilde = (1-t)*R; recover R
                R_np = R_pred.cpu().numpy() / (1.0 - t)

                dp = mu @ R_np
                mu = mu + dt * dp
                mu = np.clip(mu, 0.0, None)
                s = mu.sum()
                if s > 1e-15:
                    mu /= s

    return times, trajectory


def sample_trajectory_film(
    model,
    mu_start: np.ndarray,
    node_context: np.ndarray,
    global_cond: np.ndarray,
    edge_index,
    n_steps: int = 200,
    device=None,
) -> tuple:
    """
    Integrate flow with FiLM conditioning via FiLMConditionalGNNRateMatrixPredictor.

    node_context and global_cond are fixed throughout the trajectory.

    Args:
        mu_start:     (N,) initial distribution
        node_context: (N, node_context_dim) per-node sensor features, fixed
        global_cond:  (global_dim,) raw sensor vector + tau_diff, fixed
        edge_index:   (2, E) LongTensor for this graph

    Returns: (times, trajectory)
        times:      np.ndarray (n_steps,)
        trajectory: np.ndarray (n_steps, N)
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    N = len(mu_start)
    times = np.linspace(0.0, 0.999, n_steps)
    dt = times[1] - times[0] if n_steps > 1 else 0.0

    nctx_tensor = torch.tensor(node_context, dtype=torch.float32, device=device)
    gctx_tensor = torch.tensor(global_cond, dtype=torch.float32, device=device)
    edge_index_dev = edge_index.to(device)

    trajectory = np.zeros((n_steps, N))
    mu = mu_start.copy().astype(float)

    with torch.no_grad():
        for k, t in enumerate(times):
            trajectory[k] = mu

            if k < n_steps - 1:
                mu_tensor = torch.tensor(mu, dtype=torch.float32, device=device)
                t_tensor = torch.tensor([t], dtype=torch.float32, device=device)

                R_pred = model.forward_single(
                    mu_tensor, t_tensor, nctx_tensor, gctx_tensor, edge_index_dev)
                R_np = R_pred.cpu().numpy() / (1.0 - t)

                dp = mu @ R_np
                mu = mu + dt * dp
                mu = np.clip(mu, 0.0, None)
                s = mu.sum()
                if s > 1e-15:
                    mu /= s

    return times, trajectory


def sample_trajectory(
    model,
    mu_start: np.ndarray,
    n_steps: int = 200,
    device=None,
) -> tuple:
    """
    Euler method integration:
        mu_{k+1} = mu_k + dt * mu_k @ R_theta(mu_k, t_k)

    Clip to non-negative and renormalize after each step.

    Returns: (times, trajectory)
        times: np.ndarray (n_steps,)
        trajectory: np.ndarray (n_steps, N)
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    N = len(mu_start)
    times = np.linspace(0.0, 0.999, n_steps)
    dt = times[1] - times[0] if n_steps > 1 else 0.0

    trajectory = np.zeros((n_steps, N))
    mu = mu_start.copy().astype(float)

    with torch.no_grad():
        for k, t in enumerate(times):
            trajectory[k] = mu

            if k < n_steps - 1:
                mu_tensor = torch.tensor(mu, dtype=torch.float32, device=device).unsqueeze(0)
                t_tensor = torch.tensor([[t]], dtype=torch.float32, device=device)

                R_pred = model(mu_tensor, t_tensor)
                R_np = R_pred.squeeze(0).cpu().numpy() / (1.0 - t)

                dp = mu @ R_np
                mu = mu + dt * dp

                mu = np.clip(mu, 0.0, None)
                s = mu.sum()
                if s > 1e-15:
                    mu /= s

    return times, trajectory


def sample_trajectory_conditional(
    model,
    mu_start: np.ndarray,
    context: np.ndarray,
    n_steps: int = 200,
    device=None,
) -> tuple:
    """
    Integrate the conditional flow forward from t=0 to t~1.

    Args:
        mu_start: (N,) initial distribution
        context:  (N, context_dim) per-node conditioning, fixed throughout

    Returns: (times, trajectory)
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    N = len(mu_start)
    times = np.linspace(0.0, 0.999, n_steps)
    dt = times[1] - times[0] if n_steps > 1 else 0.0

    ctx_tensor = torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0)

    trajectory = np.zeros((n_steps, N))
    mu = mu_start.copy().astype(float)

    with torch.no_grad():
        for k, t in enumerate(times):
            trajectory[k] = mu

            if k < n_steps - 1:
                mu_tensor = torch.tensor(mu, dtype=torch.float32, device=device).unsqueeze(0)
                t_tensor = torch.tensor([[t]], dtype=torch.float32, device=device)

                R_pred = model(mu_tensor, t_tensor, ctx_tensor)
                R_np = R_pred.squeeze(0).cpu().numpy() / (1.0 - t)

                dp = mu @ R_np
                mu = mu + dt * dp
                mu = np.clip(mu, 0.0, None)
                s = mu.sum()
                if s > 1e-15:
                    mu /= s

    return times, trajectory


def sample_trajectory_guided(
    model,
    mu_start: np.ndarray,
    context: np.ndarray,
    guidance_weight: float = 0.0,
    n_steps: int = 200,
    device=None,
) -> tuple:
    """
    Integrate with classifier-free guidance.

    Args:
        mu_start:        (N,) initial distribution
        context:         (N, context_dim) per-node conditioning, fixed
        guidance_weight: w >= 0; 0 = pure conditional

    Returns: (times, trajectory)
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    N = len(mu_start)
    times = np.linspace(0.0, 0.999, n_steps)
    dt = times[1] - times[0] if n_steps > 1 else 0.0

    ctx_tensor = torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
    zero_ctx = torch.zeros_like(ctx_tensor)

    trajectory = np.zeros((n_steps, N))
    mu = mu_start.copy().astype(float)

    with torch.no_grad():
        for k, t in enumerate(times):
            trajectory[k] = mu

            if k < n_steps - 1:
                mu_tensor = torch.tensor(mu, dtype=torch.float32, device=device).unsqueeze(0)
                t_tensor = torch.tensor([[t]], dtype=torch.float32, device=device)

                u_tilde_cond = model(mu_tensor, t_tensor, ctx_tensor)

                if guidance_weight == 0.0:
                    u_tilde_use = u_tilde_cond
                else:
                    u_tilde_uncond = model(mu_tensor, t_tensor, zero_ctx)
                    u_tilde_use = ((1 + guidance_weight) * u_tilde_cond
                                   - guidance_weight * u_tilde_uncond)
                    diag_mask = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
                    off = u_tilde_use.clone()
                    off[diag_mask.expand_as(off)] = 0.0
                    off = torch.clamp(off, min=0.0)
                    diag_vals = -off.sum(dim=-1)
                    u_tilde_use = off.clone()
                    u_tilde_use[:, torch.arange(N), torch.arange(N)] = diag_vals.squeeze(0)

                R_np = u_tilde_use.squeeze(0).cpu().numpy() / (1.0 - t)
                dp = mu @ R_np
                mu = mu + dt * dp
                mu = np.clip(mu, 0.0, None)
                s = mu.sum()
                if s > 1e-15:
                    mu /= s

    return times, trajectory


def backward_trajectory(
    model,
    mu_end: np.ndarray,
    n_steps: int = 200,
    device=None,
) -> tuple:
    """
    Integrate the learned flow backward from t=1 to t=0.

    Returns:
        times: np.ndarray (n_steps,) from ~1 to ~0
        trajectory: np.ndarray (n_steps, N)
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    N = len(mu_end)
    times = np.linspace(0.999, 0.0, n_steps)
    dt = (0.999 / n_steps)

    trajectory = np.zeros((n_steps, N))
    mu = mu_end.copy().astype(float)

    with torch.no_grad():
        for k, t in enumerate(times):
            trajectory[k] = mu

            if k < n_steps - 1:
                mu_tensor = torch.tensor(mu, dtype=torch.float32, device=device).unsqueeze(0)
                t_tensor = torch.tensor([[t]], dtype=torch.float32, device=device)

                R_pred = model(mu_tensor, t_tensor)
                R_np = R_pred.squeeze(0).cpu().numpy() / (1.0 - t)

                dp = mu @ R_np
                mu = mu - dt * dp

                mu = np.clip(mu, 0.0, None)
                s = mu.sum()
                if s > 1e-15:
                    mu /= s

    return times, trajectory
