"""
Diagnose the loss spike in ex10 training (~epoch 700/750).

Hypothesis: Adam's effective learning rate grows without bound when gradients
become very small after convergence. With eps=1e-8 (default), once v_t -> 0,
lr_eff = lr / (sqrt(v_t) + eps) -> lr / eps = 1e-3 / 1e-8 = 1e5.

Usage:
    uv run python benchmark_loss_spike.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.linalg import expm

from graph_ot_fm import make_grid_graph, GraphStructure
from meta_fm import (
    FlexibleConditionalGNNRateMatrixPredictor,
    TopologyGeneralizationDataset,
    get_device,
)

torch.manual_seed(42)
device = get_device()

# ── Minimal version of ex10 training to reproduce the spike ──────────────────

GRAPHS_SMALL = [
    ('grid_3x3', make_grid_graph(3, 3, weighted=False)),
    ('grid_4x4', make_grid_graph(4, 4, weighted=False)),
    ('cycle_8',  __import__('graph_ot_fm').make_cycle_graph(8)),
    ('cycle_10', __import__('graph_ot_fm').make_cycle_graph(10)),
    ('path_6',   __import__('graph_ot_fm').make_path_graph(6)),
    ('star_7',   __import__('graph_ot_fm').make_star_graph(7)),
]

print("Building small dataset (6 graphs × 500 samples)...")
dataset = TopologyGeneralizationDataset(
    graphs=GRAPHS_SMALL,
    n_samples_per_graph=500,
    n_pairs_per_graph=20,
    tau_diff_range=(0.3, 1.5),
    seed=42,
)
print(f"Dataset: {len(dataset)} samples")

def _collate_variable(batch):
    return batch

dataloader = DataLoader(dataset, batch_size=128, shuffle=True,
                        drop_last=False, collate_fn=_collate_variable)

_off_diag_masks = {}
def _get_mask(N):
    if N not in _off_diag_masks:
        _off_diag_masks[N] = ~torch.eye(N, dtype=torch.bool, device=device)
    return _off_diag_masks[N]

model = FlexibleConditionalGNNRateMatrixPredictor(
    context_dim=2, hidden_dim=64, n_layers=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-4)  # eps=1e-4, clip 1.0

print("\nTraining with Adam(eps=1e-4) + grad_clip=1.0 — tracking grad norms...")
print(f"{'Epoch':>6}  {'Loss':>10}  {'GradNorm':>10}  {'EffLR_max':>12}  {'v_min':>12}")
print("-" * 60)

N_EPOCHS = 200
for epoch in range(N_EPOCHS):
    epoch_loss = 0.0
    epoch_grad_norm = 0.0
    n_batches = 0

    for batch in dataloader:
        groups = {}
        for sample in batch:
            key = sample[4].data_ptr()
            groups.setdefault(key, []).append(sample)

        group_losses = []
        for group in groups.values():
            mu_b  = torch.stack([s[0] for s in group]).to(device)
            tau_b = torch.stack([s[1] for s in group]).to(device)
            ctx_b = torch.stack([s[2] for s in group]).to(device)
            R_b   = torch.stack([s[3] for s in group]).to(device)
            ei    = group[0][4].to(device)
            N     = int(group[0][5])

            R_pred = model.forward_batch(mu_b, tau_b, ctx_b, ei)
            off_diag_mask = _get_mask(N)
            diff_sq = (R_pred - R_b) ** 2
            per_sample = (diff_sq * off_diag_mask.float()).sum(dim=(-2, -1))
            group_losses.append(per_sample)

        batch_loss = torch.cat(group_losses).mean()
        optimizer.zero_grad()
        batch_loss.backward()

        # Measure gradient norm, then clip to 1.0
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += batch_loss.item()
        epoch_grad_norm += grad_norm.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    avg_grad_norm = epoch_grad_norm / n_batches

    # Check Adam's internal state: minimum v (second moment) across all params
    v_values = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state and 'exp_avg_sq' in optimizer.state[p]:
                v_values.append(optimizer.state[p]['exp_avg_sq'].min().item())
    v_min = min(v_values) if v_values else float('nan')
    # Effective max LR = lr / (sqrt(v_min) + eps)
    eps = optimizer.param_groups[0].get('eps', 1e-8)
    eff_lr_max = 1e-3 / (v_min**0.5 + eps) if v_min >= 0 else float('nan')

    if (epoch + 1) % 10 == 0:
        flag = "  *** SPIKE ***" if avg_loss > 0.5 else ""
        print(f"{epoch+1:6d}  {avg_loss:10.6f}  {avg_grad_norm:10.6f}  "
              f"{eff_lr_max:12.1f}  {v_min:12.2e}{flag}")

# ── Show what effective LR looks like at various stages ──────────────────────
print("\n--- Adam effective LR = lr / (sqrt(v) + eps) ---")
print("When gradients are small, v -> 0, and eff_LR -> lr/eps:")
for v in [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 0.0]:
    for eps in [1e-8, 1e-4]:
        eff = 1e-3 / (v**0.5 + eps)
        print(f"  v={v:.0e}  eps={eps:.0e}  eff_LR = {eff:.1f}")
    print()
