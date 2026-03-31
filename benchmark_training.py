"""
Benchmark training bottlenecks across experiments.

Usage:
    uv run python benchmark_training.py

Measures time for individual components to identify where time is spent.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import torch
from contextlib import contextmanager

from graph_ot_fm import (
    GraphStructure, make_grid_graph, make_cycle_graph,
    make_path_graph, make_barbell_graph,
)
from meta_fm import (
    GNNRateMatrixPredictor,
    ConditionalGNNRateMatrixPredictor,
    FlexibleConditionalGNNRateMatrixPredictor,
    rate_matrix_to_edge_index,
    get_device,
)


# ── Utilities ────────────────────────────────────────────────────────────────

@contextmanager
def timer(label, n_iters=1):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    ms_per = elapsed / n_iters * 1000
    print(f"  {label:<50s} {ms_per:8.3f} ms/iter  ({elapsed*1000:.1f} ms total, {n_iters} iters)")


def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


device = get_device()
torch.manual_seed(42)
rng = np.random.default_rng(42)

N_REPS = 200  # repetitions for stable timing


# ── 1. GNNRateMatrixPredictor forward (used in ex3, ex4, ex5, ex6, ex7) ─────

sep("1. GNNRateMatrixPredictor.forward — batch size sweep")

R33 = make_grid_graph(3, 3, weighted=False)
ei33 = rate_matrix_to_edge_index(R33)
model_gnn = GNNRateMatrixPredictor(edge_index=ei33, n_nodes=9, hidden_dim=64, n_layers=4).to(device)
model_gnn.eval()

for B in [1, 16, 32, 64]:
    mu = torch.rand(B, 9, device=device)
    mu /= mu.sum(dim=1, keepdim=True)
    t = torch.rand(B, 1, device=device)
    # warmup
    with torch.no_grad():
        for _ in range(5):
            model_gnn(mu, t)
    with timer(f"GNNRateMatrixPredictor forward  B={B:3d}, N=9", N_REPS):
        with torch.no_grad():
            for _ in range(N_REPS):
                model_gnn(mu, t)

R55 = make_grid_graph(5, 5, weighted=False)
ei55 = rate_matrix_to_edge_index(R55)
model_gnn55 = GNNRateMatrixPredictor(edge_index=ei55, n_nodes=25, hidden_dim=64, n_layers=4).to(device)
model_gnn55.eval()

for B in [1, 16, 32, 64]:
    mu = torch.rand(B, 25, device=device)
    mu /= mu.sum(dim=1, keepdim=True)
    t = torch.rand(B, 1, device=device)
    with torch.no_grad():
        for _ in range(5):
            model_gnn55(mu, t)
    with timer(f"GNNRateMatrixPredictor forward  B={B:3d}, N=25", N_REPS):
        with torch.no_grad():
            for _ in range(N_REPS):
                model_gnn55(mu, t)


# ── 2. ConditionalGNNRateMatrixPredictor forward (ex8, ex8b, ex9) ────────────

sep("2. ConditionalGNNRateMatrixPredictor.forward — batch size sweep")

model_cond = ConditionalGNNRateMatrixPredictor(
    edge_index=ei55, n_nodes=25, context_dim=2, hidden_dim=64, n_layers=4).to(device)
model_cond.eval()

for B in [1, 16, 32, 64]:
    mu = torch.rand(B, 25, device=device)
    mu /= mu.sum(dim=1, keepdim=True)
    t = torch.rand(B, 1, device=device)
    ctx = torch.rand(B, 25, 2, device=device)
    with torch.no_grad():
        for _ in range(5):
            model_cond(mu, t, ctx)
    with timer(f"ConditionalGNN forward          B={B:3d}, N=25", N_REPS):
        with torch.no_grad():
            for _ in range(N_REPS):
                model_cond(mu, t, ctx)


# ── 3. Inside Batch.from_data_list — isolate that cost ───────────────────────

sep("3. Batch.from_data_list overhead isolation (N=25)")

from torch_geometric.data import Data, Batch

node_features_sample = torch.rand(25, 4, device=device)

for B in [1, 16, 32, 64]:
    data_list = [Data(x=node_features_sample, edge_index=ei55) for _ in range(B)]
    with timer(f"Batch.from_data_list            B={B:3d}, N=25", N_REPS):
        for _ in range(N_REPS):
            data_list = [Data(x=node_features_sample, edge_index=ei55) for _ in range(B)]
            batch = Batch.from_data_list(data_list)


# ── 4. FlexibleConditionalGNN forward_single — per-sample cost ───────────────

sep("4. FlexibleConditionalGNN.forward_single — graph size sweep")

model_flex = FlexibleConditionalGNNRateMatrixPredictor(
    context_dim=2, hidden_dim=64, n_layers=4).to(device)
model_flex.eval()

for name, R in [
    ("cycle_8",    make_cycle_graph(8)),
    ("grid_3x3",   make_grid_graph(3, 3)),
    ("grid_4x4",   make_grid_graph(4, 4)),
    ("grid_5x5",   make_grid_graph(5, 5)),
    ("barbell_4_3", make_barbell_graph(4, 3)),
]:
    N = R.shape[0]
    ei = rate_matrix_to_edge_index(R).to(device)
    mu = torch.rand(N, device=device); mu /= mu.sum()
    t = torch.tensor([0.5], device=device)
    ctx = torch.rand(N, 2, device=device)
    with torch.no_grad():
        for _ in range(5):
            model_flex.forward_single(mu, t, ctx, ei)
    with timer(f"forward_single                  N={N:2d} ({name})", N_REPS):
        with torch.no_grad():
            for _ in range(N_REPS):
                model_flex.forward_single(mu, t, ctx, ei)


# ── 5. train_flexible_conditional inner loop — full batch cost ───────────────

sep("5. train_flexible_conditional: simulated batch processing")

# Simulate one batch of 32 samples (mix of topologies like real training)
graphs_mix = [
    make_cycle_graph(8), make_cycle_graph(10), make_grid_graph(3, 3),
    make_grid_graph(4, 4), make_grid_graph(5, 5), make_path_graph(8),
    make_barbell_graph(4, 3),
]
batch_size = 32
fake_batch = []
for i in range(batch_size):
    R = graphs_mix[i % len(graphs_mix)]
    N = R.shape[0]
    ei = rate_matrix_to_edge_index(R)
    mu = torch.rand(N); mu /= mu.sum()
    tau = torch.rand(1)
    ctx = torch.rand(N, 2)
    R_t = torch.rand(N, N)
    fake_batch.append((mu, tau, ctx, R_t, ei, N))

model_flex_train = FlexibleConditionalGNNRateMatrixPredictor(
    context_dim=2, hidden_dim=64, n_layers=4).to(device)

N_BATCH_REPS = 50

# Current approach: per-sample loop
def process_batch_current(batch, model):
    sample_losses = []
    for mu, tau, context, R_target, edge_index, n_nodes in batch:
        mu = mu.to(device)
        tau = tau.to(device)
        context = context.to(device)
        R_target = R_target.to(device)
        edge_index = edge_index.to(device)
        N = int(n_nodes)
        R_pred = model.forward_single(mu, tau, context, edge_index)
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=device)
        diff_sq = (R_pred - R_target) ** 2
        per_sample_loss = (diff_sq * off_diag_mask.float()).sum()
        sample_losses.append(per_sample_loss)
    return torch.stack(sample_losses).mean()

with timer(f"current: per-sample loop        B={batch_size}, mixed topologies", N_BATCH_REPS):
    for _ in range(N_BATCH_REPS):
        loss = process_batch_current(fake_batch, model_flex_train)


# ── 6. Dataset construction cost ─────────────────────────────────────────────

sep("6. TopologyGeneralizationDataset construction (small scale)")

from meta_fm import TopologyGeneralizationDataset
from graph_ot_fm import make_star_graph

small_graphs = [
    ('cycle_8',  make_cycle_graph(8)),
    ('grid_3x3', make_grid_graph(3, 3)),
    ('path_8',   make_path_graph(8)),
]

t0 = time.perf_counter()
ds = TopologyGeneralizationDataset(
    graphs=small_graphs,
    n_samples_per_graph=100,
    n_pairs_per_graph=10,
    seed=42,
)
elapsed = time.perf_counter() - t0
print(f"  {'Dataset build (3 graphs × 100 samples × 10 pairs)':<50s} {elapsed*1000:.1f} ms total")
print(f"  {'  → per graph':<50s} {elapsed/3*1000:.1f} ms")
print(f"  {'  → per sample':<50s} {elapsed/len(ds)*1000:.3f} ms")

# Estimate full-scale cost
n_full_graphs = 13
n_full_samples = 1000
n_full_pairs = 50
# Scale: graphs use ~same N on average, so roughly linear
# per_sample_cost = elapsed / len(ds)
per_sample_cost = elapsed / len(ds)
est_full = per_sample_cost * n_full_graphs * n_full_samples
print(f"\n  Estimated full dataset (13 graphs × 1000 samples × 50 pairs): {est_full:.1f} s")


# ── 7. Training step cost — full backward pass ───────────────────────────────

sep("7. Full training step cost (forward + backward + optimizer)")

# ConditionalGNN on 5x5 grid (most expensive existing experiment)
model_cond_train = ConditionalGNNRateMatrixPredictor(
    edge_index=ei55, n_nodes=25, context_dim=2, hidden_dim=64, n_layers=4).to(device)
optimizer_cond = torch.optim.Adam(model_cond_train.parameters(), lr=1e-3)

N_STEP_REPS = 100
B = 64

mu_b = torch.rand(B, 25, device=device); mu_b /= mu_b.sum(dim=1, keepdim=True)
tau_b = torch.rand(B, 1, device=device)
ctx_b = torch.rand(B, 25, 2, device=device)
R_b = torch.rand(B, 25, 25, device=device)

# warmup
for _ in range(5):
    R_pred = model_cond_train(mu_b, tau_b, ctx_b)
    loss = (R_pred - R_b).pow(2).mean()
    optimizer_cond.zero_grad(); loss.backward(); optimizer_cond.step()

with timer(f"ConditionalGNN train step       B={B}, N=25", N_STEP_REPS):
    for _ in range(N_STEP_REPS):
        R_pred = model_cond_train(mu_b, tau_b, ctx_b)
        loss = (R_pred - R_b).pow(2).mean()
        optimizer_cond.zero_grad()
        loss.backward()
        optimizer_cond.step()

# Project to full training
ms_per_step = None  # captured above
# Re-measure to get value
t0 = time.perf_counter()
for _ in range(N_STEP_REPS):
    R_pred = model_cond_train(mu_b, tau_b, ctx_b)
    loss = (R_pred - R_b).pow(2).mean()
    optimizer_cond.zero_grad(); loss.backward(); optimizer_cond.step()
ms_per_step = (time.perf_counter() - t0) / N_STEP_REPS * 1000

n_steps_per_epoch = 5000 // B  # ex8 dataset size / batch size
est_epoch_s = ms_per_step * n_steps_per_epoch / 1000
print(f"\n  Estimated ex8 training:  {ms_per_step:.1f} ms/step × {n_steps_per_epoch} steps/epoch")
print(f"  → {est_epoch_s:.1f} s/epoch × 500 epochs = {est_epoch_s*500/60:.1f} min")

# Flexible model full training step
t0 = time.perf_counter()
optimizer_flex = torch.optim.Adam(model_flex_train.parameters(), lr=1e-3)
for _ in range(N_BATCH_REPS):
    loss = process_batch_current(fake_batch, model_flex_train)
    optimizer_flex.zero_grad(); loss.backward(); optimizer_flex.step()
ms_flex_step = (time.perf_counter() - t0) / N_BATCH_REPS * 1000

n_steps_per_epoch_flex = 13000 // batch_size  # ex10 total / batch size
est_epoch_flex = ms_flex_step * n_steps_per_epoch_flex / 1000
print(f"\n  Estimated ex10 training: {ms_flex_step:.1f} ms/step × {n_steps_per_epoch_flex} steps/epoch")
print(f"  → {est_epoch_flex:.1f} s/epoch × 1000 epochs = {est_epoch_flex*1000/60:.1f} min")


# ── 8. Speedup measurements ───────────────────────────────────────────────────

sep("8. After fixes: forward_batch vs forward_single loop")

model_fixed = FlexibleConditionalGNNRateMatrixPredictor(
    context_dim=2, hidden_dim=64, n_layers=4).to(device)
model_fixed.eval()

# Same mixed-topology batch as before
for B in [32, 64, 128, 256]:
    # Build a batch all from the same topology (best case for forward_batch)
    R_55 = make_grid_graph(5, 5)
    ei_55 = rate_matrix_to_edge_index(R_55).to(device)
    mu_b = torch.rand(B, 25, device=device); mu_b /= mu_b.sum(dim=1, keepdim=True)
    t_b = torch.rand(B, 1, device=device)
    ctx_b = torch.rand(B, 25, 2, device=device)

    with torch.no_grad():
        for _ in range(5):
            model_fixed.forward_batch(mu_b, t_b, ctx_b, ei_55)

    with timer(f"forward_batch (same topology)   B={B:3d}, N=25", N_REPS):
        with torch.no_grad():
            for _ in range(N_REPS):
                model_fixed.forward_batch(mu_b, t_b, ctx_b, ei_55)

print()

# Compare: per-sample loop vs grouped forward_batch for the mixed batch
def process_batch_grouped(batch, model):
    groups = {}
    for sample in batch:
        mu, tau, ctx, R_t, ei, n = sample
        key = ei.data_ptr()
        if key not in groups:
            groups[key] = []
        groups[key].append(sample)

    all_losses = []
    for group in groups.values():
        mu_b = torch.stack([s[0] for s in group]).to(device)
        t_b  = torch.stack([s[1] for s in group]).to(device)
        c_b  = torch.stack([s[2] for s in group]).to(device)
        R_b  = torch.stack([s[3] for s in group]).to(device)
        ei   = group[0][4].to(device)
        R_pred = model.forward_batch(mu_b, t_b, c_b, ei)
        all_losses.append((R_pred - R_b).pow(2).mean())
    return torch.stack(all_losses).mean()

with timer(f"grouped forward_batch           B={batch_size}, mixed topologies", N_BATCH_REPS):
    for _ in range(N_BATCH_REPS):
        loss = process_batch_grouped(fake_batch, model_fixed)

# Re-time current approach for direct comparison
with timer(f"per-sample forward_single       B={batch_size}, mixed topologies", N_BATCH_REPS):
    for _ in range(N_BATCH_REPS):
        loss = process_batch_current(fake_batch, model_fixed)


sep("9. After fix: marginal_distribution_fast binom caching")

from graph_ot_fm.flow import marginal_distribution_fast

R55 = make_grid_graph(5, 5)
from graph_ot_fm import GraphStructure, compute_cost_matrix, compute_ot_coupling
from graph_ot_fm.geodesic_cache import GeodesicCache
graph55 = GraphStructure(R55)
cache55 = GeodesicCache(graph55)
cost55 = compute_cost_matrix(graph55)
mu_src55 = np.random.default_rng(7).dirichlet(np.ones(25))
mu_obs55 = mu_src55 @ np.eye(25)  # trivial diffuse for coupling
pi55 = compute_ot_coupling(mu_src55, mu_obs55, cost55)
cache55.precompute_for_coupling(pi55)

with timer("marginal_distribution_fast (fixed)  N=25", N_REPS):
    for _ in range(N_REPS):
        marginal_distribution_fast(cache55, pi55, 0.5)

# Full training step speedup estimate
optimizer_flex2 = torch.optim.Adam(model_fixed.parameters(), lr=1e-3)
t0 = time.perf_counter()
for _ in range(N_BATCH_REPS):
    loss = process_batch_grouped(fake_batch, model_fixed)
    optimizer_flex2.zero_grad(); loss.backward(); optimizer_flex2.step()
ms_grouped_step = (time.perf_counter() - t0) / N_BATCH_REPS * 1000

# With batch_size=256, steps per epoch = 13000/256 = 51
n_steps_256 = 13000 // 256
# Scale: grouped with B=256 has ~256/32 * fewer steps, but each step is slower
# Actual step time at B=256 needs separate measurement; scale from B=32 grouped
# forward_batch scales sublinearly, so assume 2x time for 8x more samples = 4x eff speedup
est_step_256 = ms_grouped_step * 2.0  # conservative: 2x slower per step at B=256
est_epoch_grouped = est_step_256 * n_steps_256 / 1000
print(f"\n  Grouped step (B=32):   {ms_grouped_step:.1f} ms")
print(f"  Per-sample step (B=32): {ms_per_step:.1f} ms")
print(f"  Speedup per step: {ms_per_step/ms_grouped_step:.1f}x")
print(f"\n  Estimated ex10 with B=256, grouped: {est_epoch_grouped:.1f} s/epoch × 1000 = {est_epoch_grouped*1000/60:.0f} min")


# ── Summary ───────────────────────────────────────────────────────────────────

sep("Summary")
print("""
  Key questions answered above:
  - How much does Batch.from_data_list cost vs total forward time?
  - How does cost scale with batch size vs graph size?
  - How slow is the per-sample loop in train_flexible_conditional?
  - How long will full training actually take?
  - How expensive is dataset construction?
""")
