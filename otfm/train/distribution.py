"""
Training loop for FiLMConditionalGNNRateMatrixPredictor.

From meta_fm/train.py::train_film_conditional.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from otfm.core.utils import EMA, get_device


def _per_sample_loss(R_pred, R_target, off_diag_mask, loss_type):
    """Compute per-sample loss (N*(N-1) normalised) for a batch."""
    N = R_pred.shape[-1]
    if loss_type == 'rate_kl':
        mask = off_diag_mask.unsqueeze(0).expand_as(R_pred)
        r       = R_target.clamp(min=1e-8)
        r_theta = R_pred.clamp(min=1e-8)
        per_entry = r * torch.log(r / r_theta) - r + r_theta
        return (per_entry * mask.float()).sum(dim=(-2, -1)) / (N * (N - 1))
    else:  # mse
        mask = off_diag_mask.unsqueeze(0).expand_as(R_pred)
        return ((R_pred - R_target) ** 2 * mask.float()).sum(dim=(-2, -1)) / (N * (N - 1))


def train_film_conditional(
    model,
    dataset,
    n_epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 5e-4,
    device=None,
    loss_weighting: str = 'uniform',
    loss_type: str = 'rate_kl',
    ema_decay: float = 0.999,
) -> dict:
    """
    Training loop for FiLMConditionalGNNRateMatrixPredictor.

    Dataset returns (mu, tau, node_context, global_cond, R_target, edge_index, n_nodes).
    Samples sharing the same edge_index are batched together via forward_batch.

    loss_weighting: 'uniform', 'original', or 'linear' -- same as train().
    Returns: dict with 'losses' and 'ema' keys.
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.train()

    def _collate_variable(batch):
        return batch

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False, collate_fn=_collate_variable)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4, foreach=True)
    ema = EMA(model, decay=ema_decay)

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
            groups: dict = {}
            for sample in batch:
                mu, tau, node_ctx, global_ctx, R_target, edge_index, n_nodes = sample
                key = edge_index.data_ptr()
                if key not in groups:
                    groups[key] = []
                groups[key].append(sample)

            group_losses = []

            for group in groups.values():
                mu_b    = torch.stack([s[0] for s in group]).to(device)   # (G, N)
                tau_b   = torch.stack([s[1] for s in group]).to(device)   # (G, 1)
                nctx_b  = torch.stack([s[2] for s in group]).to(device)   # (G, N, ctx)
                gctx_b  = torch.stack([s[3] for s in group]).to(device)   # (G, global_dim)
                R_b     = torch.stack([s[4] for s in group]).to(device)   # (G, N, N)
                ei      = group[0][5].to(device)
                N       = int(group[0][6])

                R_pred = model.forward_batch(mu_b, tau_b, nctx_b, gctx_b, ei)

                off_diag_mask = _get_mask(N)
                per_sample = _per_sample_loss(R_pred, R_b, off_diag_mask, loss_type)  # (G,)

                tau_vals = tau_b.squeeze(-1)
                if loss_weighting == 'original':
                    weights = 1.0 / (1.0 - tau_vals).clamp(min=0.001) ** 2
                elif loss_weighting == 'linear':
                    weights = 1.0 / (1.0 - tau_vals).clamp(min=0.001)
                else:
                    weights = torch.ones_like(tau_vals)

                group_losses.append(weights * per_sample)

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
