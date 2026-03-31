"""
Diagnose which parameters have zero gradients and why.

Usage:
    uv run python benchmark_dead_neurons.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from graph_ot_fm import make_grid_graph, make_cycle_graph, make_path_graph, make_star_graph
from meta_fm import FlexibleConditionalGNNRateMatrixPredictor, TopologyGeneralizationDataset, get_device
from meta_fm.model import rate_matrix_to_edge_index

torch.manual_seed(42)
device = get_device()

# ── Build a small dataset ─────────────────────────────────────────────────────

GRAPHS = [
    ('grid_3x3', make_grid_graph(3, 3, weighted=False)),
    ('cycle_8',  make_cycle_graph(8)),
    ('path_6',   make_path_graph(6)),
    ('star_7',   make_star_graph(7)),
]

print("Building dataset...")
dataset = TopologyGeneralizationDataset(
    graphs=GRAPHS, n_samples_per_graph=500, n_pairs_per_graph=20, seed=42)
print(f"  {len(dataset)} samples\n")

model = FlexibleConditionalGNNRateMatrixPredictor(
    context_dim=2, hidden_dim=64, n_layers=4).to(device)

# ── 1. Check which neurons fire (pre-ReLU activation > 0) across a big batch ─

print("=" * 65)
print("  1. Dead neuron detection: which ReLU inputs are ALWAYS <= 0?")
print("=" * 65)

# Hook to capture pre-ReLU activations
pre_relu: dict = {}
hooks = []

def make_hook(name):
    def hook(module, inp, out):
        # inp[0] is the pre-activation tensor
        x = inp[0].detach().cpu()
        if name not in pre_relu:
            pre_relu[name] = []
        pre_relu[name].append(x)
    return hook

# Register hooks on every ReLU
for name, module in model.named_modules():
    if isinstance(module, nn.ReLU):
        hooks.append(module.register_forward_hook(make_hook(name)))

# Run ~1000 samples through
model.eval()
_off_diag_masks = {}
def _get_mask(N):
    if N not in _off_diag_masks:
        _off_diag_masks[N] = ~torch.eye(N, dtype=torch.bool, device=device)
    return _off_diag_masks[N]

def _collate_variable(batch): return batch

dl = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=_collate_variable)
with torch.no_grad():
    for i, batch in enumerate(dl):
        groups = {}
        for sample in batch:
            groups.setdefault(sample[4].data_ptr(), []).append(sample)
        for group in groups.values():
            mu_b  = torch.stack([s[0] for s in group]).to(device)
            tau_b = torch.stack([s[1] for s in group]).to(device)
            ctx_b = torch.stack([s[2] for s in group]).to(device)
            ei    = group[0][4].to(device)
            model.forward_batch(mu_b, tau_b, ctx_b, ei)
        if i >= 7:  # ~2000 samples
            break

for h in hooks:
    h.remove()

# Concatenate all captured activations and check
total_dead = 0
total_neurons = 0
print(f"\n  {'Layer':<55} {'Dead':>5}  {'Total':>6}  {'%Dead':>6}")
print(f"  {'-'*55}  {'-'*5}  {'-'*6}  {'-'*6}")
for name, tensors in sorted(pre_relu.items()):
    # Each tensor may have shape (B*N, hidden) with varying N — flatten all to (?, hidden)
    flat = [t.view(-1, t.shape[-1]) for t in tensors]
    x = torch.cat(flat, dim=0)   # (total_node_slots, hidden)
    max_per_neuron = x.max(dim=0).values   # (hidden,)
    n_dead = (max_per_neuron <= 0).sum().item()
    n_total = max_per_neuron.numel()
    pct = 100 * n_dead / n_total
    total_dead += n_dead
    total_neurons += n_total
    print(f"  {name:<55} {n_dead:5d}  {n_total:6d}  {pct:5.1f}%")

print(f"\n  Total dead neurons: {total_dead}/{total_neurons} "
      f"({100*total_dead/total_neurons:.1f}%)")

# ── 2. Show which parameters have zero gradients ──────────────────────────────

print("\n" + "=" * 65)
print("  2. Parameters with zero gradient after one backward pass")
print("=" * 65)

model.train()
batch = list(dl)[0]
groups = {}
for sample in batch:
    groups.setdefault(sample[4].data_ptr(), []).append(sample)

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
    per_sample = ((R_pred - R_b)**2 * off_diag_mask.float()).sum(dim=(-2,-1))
    group_losses.append(per_sample)

loss = torch.cat(group_losses).mean()
model.zero_grad()
loss.backward()

print(f"\n  {'Parameter':<55} {'Shape':>16}  {'ZeroGrad':>8}  {'MaxAbsGrad':>12}")
print(f"  {'-'*55}  {'-'*16}  {'-'*8}  {'-'*12}")

for name, param in model.named_parameters():
    if param.grad is None:
        print(f"  {name:<55} {str(tuple(param.shape)):>16}  {'NO GRAD':>8}")
        continue
    g = param.grad.cpu()
    n_zero = (g == 0).sum().item()
    frac = n_zero / g.numel()
    max_abs = g.abs().max().item()
    flag = "  <-- ALL ZERO" if frac == 1.0 else (f"  {frac*100:.0f}% zero" if frac > 0.5 else "")
    print(f"  {name:<55} {str(tuple(param.shape)):>16}  {frac*100:7.1f}%  {max_abs:12.2e}{flag}")

# ── 3. Which specific neurons are dead and why ────────────────────────────────

print("\n" + "=" * 65)
print("  3. Dead neuron pre-activation statistics")
print("=" * 65)

# Look at the first ReLU's input distribution
first_key = list(pre_relu.keys())[0]
x_first = torch.cat(pre_relu[first_key], dim=0)
if x_first.dim() > 2:
    x_first = x_first.view(-1, x_first.shape[-1])
max_vals = x_first.max(dim=0).values
dead_idx = (max_vals <= 0).nonzero().squeeze(-1)
if len(dead_idx) > 0:
    print(f"\n  Layer '{first_key}': {len(dead_idx)} dead neurons")
    print(f"  Pre-activation max for dead neurons: "
          f"min={max_vals[dead_idx].min():.4f}, "
          f"max={max_vals[dead_idx].max():.4f}")
    # Show the input feature range to understand initialization
    print(f"  Pre-activation stats for ALL neurons: "
          f"mean={x_first.mean():.3f}, std={x_first.std():.3f}, "
          f"min={x_first.min():.3f}, max={x_first.max():.3f}")
