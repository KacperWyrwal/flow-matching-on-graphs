"""
Training loop for FlexibleConditionalGNNRateMatrixPredictor.

From meta_fm/train.py::train_flexible_conditional.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from otfm.core.utils import EMA, get_device
from otfm.core.loss import rate_kl_divergence


def _per_sample_loss(R_pred, R_target, off_diag_mask, loss_type):
    """Compute per-sample loss (N*(N-1) normalised) for a batch."""
    N = R_pred.shape[-1]
    if loss_type == 'rate_kl':
        # rate_kl_divergence works on full matrices; mask via off_diag_mask
        mask = off_diag_mask.unsqueeze(0).expand_as(R_pred)
        r       = R_target.clamp(min=1e-8)    # target
        r_theta = R_pred.clamp(min=1e-8)      # predicted
        per_entry = r * torch.log(r / r_theta) - r + r_theta
        return (per_entry * mask.float()).sum(dim=(-2, -1)) / (N * (N - 1))
    else:  # mse
        mask = off_diag_mask.unsqueeze(0).expand_as(R_pred)
        return ((R_pred - R_target) ** 2 * mask.float()).sum(dim=(-2, -1)) / (N * (N - 1))


def train_flexible_conditional(
    model,
    dataset,
    n_epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    device=None,
    loss_weighting: str = 'uniform',
    loss_type: str = 'rate_kl',
    ema_decay: float = 0.999,
) -> dict:
    """
    Training loop for FlexibleConditionalGNNRateMatrixPredictor.

    Dataset must return (mu, tau, context, R_target, edge_index, n_nodes)
    per sample with variable N. Samples are grouped by graph topology within
    each mini-batch so that forward_batch can be used (one GNN pass per
    topology group rather than one per sample).

    loss_weighting: 'uniform', 'original', or 'linear' -- same as train().
    Returns: dict with 'losses' and 'ema' keys.
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.train()

    def _collate_variable(batch):
        return batch  # list of variable-size tuples

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False, collate_fn=_collate_variable)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4, foreach=True)
    ema = EMA(model, decay=ema_decay)

    # Cache off-diagonal masks by N to avoid recreating each step
    _off_diag_masks: dict = {}

    def _get_mask(N):
        if N not in _off_diag_masks:
            _off_diag_masks[N] = ~torch.eye(N, dtype=torch.bool, device=device)
        return _off_diag_masks[N]

    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            # Group samples by graph topology using edge_index identity
            groups: dict = {}
            for sample in batch:
                if len(sample) == 7:
                    mu, tau, context, R_target, edge_index, edge_feat, n_nodes = sample
                else:
                    mu, tau, context, R_target, edge_index, n_nodes = sample
                    edge_feat = None
                key = edge_index.data_ptr()
                if key not in groups:
                    groups[key] = []
                groups[key].append(sample)

            group_losses = []

            for group in groups.values():
                mu_b = torch.stack([s[0] for s in group]).to(device)      # (G, N)
                tau_b = torch.stack([s[1] for s in group]).to(device)     # (G, 1)
                ctx_b = torch.stack([s[2] for s in group]).to(device)     # (G, N, ctx)
                R_b = torch.stack([s[3] for s in group]).to(device)       # (G, N, N)
                ei = group[0][4].to(device)
                has_ef = len(group[0]) == 7
                ef = group[0][5].to(device) if has_ef else None
                N = int(group[0][-1])

                if ef is not None and hasattr(model, 'edge_dim') and model.edge_dim > 0:
                    R_pred = model.forward_batch(mu_b, tau_b, ctx_b, ei, ef)
                else:
                    R_pred = model.forward_batch(mu_b, tau_b, ctx_b, ei)  # (G, N, N)

                off_diag_mask = _get_mask(N)
                per_sample = _per_sample_loss(R_pred, R_b, off_diag_mask, loss_type)  # (G,)

                tau_vals = tau_b.squeeze(-1)                               # (G,)
                if loss_weighting == 'original':
                    weights = 1.0 / (1.0 - tau_vals).clamp(min=0.001) ** 2
                elif loss_weighting == 'linear':
                    weights = 1.0 / (1.0 - tau_vals).clamp(min=0.001)
                else:
                    weights = torch.ones_like(tau_vals)

                group_losses.append((weights * per_sample))               # (G,)

            batch_loss = torch.cat(group_losses).mean()

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            epoch_loss += batch_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.6f}")

    ema.apply(model)
    return {'losses': losses, 'ema': ema}


def _weighted_loss(R_pred, R_target, tau_batch, off_diag_mask,
                   loss_weighting, loss_type='rate_kl'):
    """
    Per-sample loss on off-diagonal entries, optionally weighted by time.

    loss_type:
        'rate_kl'  -- generalized KL between rate vectors (principled default).
        'mse'      -- mean squared error (fallback).

    loss_weighting:
        'original' -- weight by 1/(1-t)^2.
        'uniform'  -- no weighting.
        'linear'   -- weight by 1/(1-t).
    """
    per_sample_loss = _per_sample_loss(R_pred, R_target, off_diag_mask, loss_type)
    N = R_pred.shape[-1]

    tau = tau_batch.squeeze(-1)  # (batch,)
    if loss_weighting == 'original':
        weights = 1.0 / (1.0 - tau).clamp(min=0.001) ** 2
    elif loss_weighting == 'linear':
        weights = 1.0 / (1.0 - tau).clamp(min=0.001)
    elif loss_weighting == 'uniform':
        weights = torch.ones_like(tau)
    else:
        raise ValueError(f"Unknown loss_weighting: {loss_weighting!r}")

    return (weights * per_sample_loss).mean()


def train(
    model,
    dataset,
    n_epochs: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    device=None,
    loss_weighting: str = 'uniform',
    loss_type: str = 'rate_kl',
    ema_decay: float = 0.999,
) -> dict:
    """
    Train the rate matrix predictor.

    loss_weighting:
        'original' -- weight by 1/(1-t)^2, equivalent to MSE on raw rate
                     matrices (default).
        'uniform'  -- equal weight across all times.
        'linear'   -- weight by 1/(1-t), a middle ground.

    Optimizer: Adam. Logs every 50 epochs.
    Returns: dict with 'losses' and 'ema' keys.
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4, foreach=True)
    ema = EMA(model, decay=ema_decay)

    N = model.n_nodes
    off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=device)

    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for mu_batch, tau_batch, R_target_batch in dataloader:
            mu_batch = mu_batch.to(device)
            tau_batch = tau_batch.to(device)
            R_target_batch = R_target_batch.to(device)

            R_pred = model(mu_batch, tau_batch)  # (batch, N, N)

            loss = _weighted_loss(R_pred, R_target_batch, tau_batch,
                                  off_diag_mask, loss_weighting, loss_type)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.6f}")

    ema.apply(model)
    return {'losses': losses, 'ema': ema}


def train_conditional(
    model,
    dataset,
    n_epochs: int = 500,
    batch_size: int = 256,
    lr: float = 1e-3,
    device=None,
    context_drop_prob: float = 0.0,
    loss_weighting: str = 'uniform',
    loss_type: str = 'rate_kl',
    ema_decay: float = 0.999,
) -> dict:
    """
    Training loop for ConditionalGNNRateMatrixPredictor.

    Dataset must return (mu, tau, context, R_target) per sample.
    Model forward: model(mu, tau, context) -> (batch, N, N).

    If context_drop_prob > 0, randomly zeros out the full context for that
    fraction of samples each batch (enables classifier-free guidance at inference).

    loss_weighting: same options as train() -- 'original' (default), 'uniform',
    or 'linear'.

    Returns: dict with 'losses' and 'ema' keys.
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4, foreach=True)
    ema = EMA(model, decay=ema_decay)

    N = model.n_nodes
    off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=device)

    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for mu_batch, tau_batch, context_batch, R_target_batch in dataloader:
            mu_batch = mu_batch.to(device)
            tau_batch = tau_batch.to(device)
            context_batch = context_batch.to(device)
            R_target_batch = R_target_batch.to(device)

            if context_drop_prob > 0:
                drop_mask = (torch.rand(mu_batch.shape[0], 1, 1, device=device)
                             < context_drop_prob)
                context_batch = context_batch * (~drop_mask).float()

            R_pred = model(mu_batch, tau_batch, context_batch)

            loss = _weighted_loss(R_pred, R_target_batch, tau_batch,
                                  off_diag_mask, loss_weighting, loss_type)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{n_epochs} | Loss: {avg_loss:.6f}")

    ema.apply(model)
    return {'losses': losses, 'ema': ema}
