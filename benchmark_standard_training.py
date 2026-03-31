"""
Benchmark standard training loop bottlenecks (ex7, ex8, ex8b, ex9).

Usage:
    uv run python benchmark_standard_training.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from torch_geometric.data import Data, Batch

from graph_ot_fm import make_grid_graph, make_cycle_graph
from meta_fm import (
    GNNRateMatrixPredictor, ConditionalGNNRateMatrixPredictor,
    rate_matrix_to_edge_index, get_device,
)
from meta_fm.model import RateMessagePassing


@contextmanager
def timer(label, n=1):
    t0 = time.perf_counter()
    yield
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"  {label:<55s} {ms:8.3f} ms/iter")

def sep(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

device = get_device()
torch.manual_seed(42)
N_REPS = 200

R55 = make_grid_graph(5, 5)
ei55 = rate_matrix_to_edge_index(R55).to(device)
N, E = 25, ei55.shape[1]

# ── 1. Dissect forward pass into components ──────────────────────────────────

sep("1. Forward pass dissection  (B=64, N=25, 4 layers, hidden=64)")

B = 64
mu = torch.rand(B, N, device=device); mu /= mu.sum(1, keepdim=True)
t  = torch.rand(B, 1, device=device)
ctx = torch.rand(B, N, 2, device=device)

hidden_dim = 64

# --- Component A: node feature assembly ---
with timer("A  node feature assembly", N_REPS):
    for _ in range(N_REPS):
        t_exp = t.expand(B, N)
        base = torch.stack([mu, t_exp], dim=-1)
        nf = torch.cat([base, ctx], dim=-1)   # (B, N, 4)

t_exp = t.expand(B, N)
base  = torch.stack([mu, t_exp], dim=-1)
nf    = torch.cat([base, ctx], dim=-1)

# --- Component B: Data list + Batch.from_data_list ---
with timer("B  Data list + Batch.from_data_list", N_REPS):
    for _ in range(N_REPS):
        dl = [Data(x=nf[b], edge_index=ei55) for b in range(B)]
        batch = Batch.from_data_list(dl)

# --- Component C: manual edge-index tiling (alternative to B) ---
with timer("C  manual edge-index tiling (alternative to B)", N_REPS):
    for _ in range(N_REPS):
        src, dst = ei55
        offsets = torch.arange(B, device=device) * N
        src_b = (src.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        dst_b = (dst.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
        batch_ei = torch.stack([src_b, dst_b])
        h_flat = nf.reshape(B * N, -1)

src, dst = ei55
offsets = torch.arange(B, device=device) * N
src_b = (src.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
dst_b = (dst.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
batch_ei = torch.stack([src_b, dst_b])
h_flat = nf.reshape(B * N, -1)

# --- Component D: one RateMessagePassing layer (PyG propagate) ---
# Use in_dim=hidden_dim (layers 2-4 shape), feed hidden features
mp_hidden = RateMessagePassing(in_dim=hidden_dim, hidden_dim=hidden_dim).to(device)
h_mp = torch.rand(B * N, hidden_dim, device=device)

with timer("D  one MP layer via PyG propagate  (B*N nodes)", N_REPS):
    for _ in range(N_REPS):
        _ = mp_hidden(h_mp, batch_ei)

# --- Component E: same MP layer manually (no PyG propagate) ---
msg_mlp    = mp_hidden.msg_mlp
update_mlp = mp_hidden.update_mlp

# Note: PyG source_to_target flow: x_i=target(dst), x_j=source(src)
with timer("E  one MP layer manual scatter_add  (B*N nodes)", N_REPS):
    for _ in range(N_REPS):
        msgs = msg_mlp(torch.cat([h_mp[dst_b], h_mp[src_b]], dim=-1))
        aggr = torch.zeros(B * N, hidden_dim, device=device)
        aggr.scatter_add_(0, dst_b.unsqueeze(-1).expand_as(msgs), msgs)
        _ = update_mlp(torch.cat([h_mp, aggr], dim=-1))

# --- Component F: 4 MP layers total (PyG) ---
in_dim_first = 4  # [mu, t, ctx_1, ctx_2]
mp_layers = nn.ModuleList([
    RateMessagePassing(in_dim=in_dim_first if i == 0 else hidden_dim, hidden_dim=hidden_dim)
    for i in range(4)
]).to(device)
h0 = torch.rand(B * N, in_dim_first, device=device)

with timer("F  4 MP layers via PyG propagate", N_REPS):
    for _ in range(N_REPS):
        h = h0
        for mp_layer in mp_layers:
            h = mp_layer(h, batch_ei)

# --- Component G: 4 MP layers manually ---
all_msg_mlps    = [mp_layers[i].msg_mlp    for i in range(4)]
all_update_mlps = [mp_layers[i].update_mlp for i in range(4)]

with timer("G  4 MP layers manual scatter_add", N_REPS):
    for _ in range(N_REPS):
        h = h0
        for msg_mlp_i, update_mlp_i in zip(all_msg_mlps, all_update_mlps):
            msgs = msg_mlp_i(torch.cat([h[dst_b], h[src_b]], dim=-1))
            aggr = torch.zeros(B * N, hidden_dim, device=device)
            aggr.scatter_add_(0, dst_b.unsqueeze(-1).expand_as(msgs), msgs)
            h = update_mlp_i(torch.cat([h, aggr], dim=-1))

# --- Component H: edge MLP + rate matrix assembly ---
model_cond = ConditionalGNNRateMatrixPredictor(
    edge_index=ei55, n_nodes=N, context_dim=2, hidden_dim=hidden_dim, n_layers=4
).to(device)

# get h after message passing
dl = [Data(x=nf[b], edge_index=ei55) for b in range(B)]
batch_obj = Batch.from_data_list(dl)
h_after_mp = batch_obj.x
for mp_layer in model_cond.mp_layers:
    h_after_mp = mp_layer(h_after_mp, batch_obj.edge_index)
h_reshaped = h_after_mp.view(B, N, hidden_dim)

with timer("H  edge MLP + rate matrix assembly", N_REPS):
    for _ in range(N_REPS):
        h_src = h_reshaped[:, src, :]
        h_dst = h_reshaped[:, dst, :]
        ef = torch.cat([h_src, h_dst], dim=-1)
        edge_rates = F.softplus(model_cond.edge_mlp(ef).squeeze(-1))
        rm = torch.zeros(B, N, N, device=device)
        rm[:, src, dst] = edge_rates
        rm[:, range(N), range(N)] = -rm.sum(dim=-1)

# --- Full forward for reference ---
with timer("FULL forward (current implementation)", N_REPS):
    with torch.no_grad():
        for _ in range(N_REPS):
            model_cond(mu, t, ctx)

# ── 2. Forward + backward breakdown ──────────────────────────────────────────

sep("2. Forward vs backward vs optimizer step  (B=64, N=25)")

optimizer = torch.optim.Adam(model_cond.parameters(), lr=1e-3)
R_dummy = torch.rand(B, N, N, device=device)
off_diag = ~torch.eye(N, dtype=torch.bool, device=device)

# warmup
for _ in range(5):
    R_pred = model_cond(mu, t, ctx)
    loss = (R_pred - R_dummy).pow(2)[:, off_diag].mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()

with timer("forward only", N_REPS):
    for _ in range(N_REPS):
        with torch.no_grad():
            model_cond(mu, t, ctx)

t0 = time.perf_counter()
for _ in range(N_REPS):
    R_pred = model_cond(mu, t, ctx)
fwd_ms = (time.perf_counter() - t0) / N_REPS * 1000

t0 = time.perf_counter()
for _ in range(N_REPS):
    R_pred = model_cond(mu, t, ctx)
    loss = (R_pred - R_dummy).pow(2)[:, off_diag].mean()
    loss.backward()
fwd_bwd_ms = (time.perf_counter() - t0) / N_REPS * 1000

t0 = time.perf_counter()
for _ in range(N_REPS):
    R_pred = model_cond(mu, t, ctx)
    loss = (R_pred - R_dummy).pow(2)[:, off_diag].mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
full_step_ms = (time.perf_counter() - t0) / N_REPS * 1000

print(f"  {'forward':<55s} {fwd_ms:8.3f} ms/iter")
print(f"  {'forward + backward':<55s} {fwd_bwd_ms:8.3f} ms/iter")
print(f"  {'full step (fwd+bwd+optim)':<55s} {full_step_ms:8.3f} ms/iter")
print(f"  {'  backward alone':<55s} {fwd_bwd_ms - fwd_ms:8.3f} ms/iter")
print(f"  {'  optimizer.step alone':<55s} {full_step_ms - fwd_bwd_ms:8.3f} ms/iter")

# ── 3. Batch size effect ───────────────────────────────────────────────────────

sep("3. Full training step cost vs batch size  (N=25)")

for B_test in [16, 32, 64, 128, 256]:
    mu_t = torch.rand(B_test, N, device=device); mu_t /= mu_t.sum(1, keepdim=True)
    t_t  = torch.rand(B_test, 1, device=device)
    c_t  = torch.rand(B_test, N, 2, device=device)
    R_t  = torch.rand(B_test, N, N, device=device)
    opt  = torch.optim.Adam(model_cond.parameters(), lr=1e-3)
    for _ in range(3):  # warmup
        rp = model_cond(mu_t, t_t, c_t)
        (rp - R_t).pow(2)[:, off_diag].mean().backward()
        opt.zero_grad(); opt.step()
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        rp = model_cond(mu_t, t_t, c_t)
        loss = (rp - R_t).pow(2)[:, off_diag].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    ms = (time.perf_counter() - t0) / N_REPS * 1000
    n_steps = 5000 // B_test   # ex8: 5000 samples
    est_epoch = ms * n_steps / 1000
    print(f"  B={B_test:3d}  step={ms:6.2f}ms  steps/epoch={n_steps:4d}  "
          f"→ {est_epoch:.2f}s/epoch × 500ep = {est_epoch*500/60:.1f}min  (ex8)")

# ── 4. DataLoader overhead ────────────────────────────────────────────────────

sep("4. DataLoader __iter__ overhead")

from meta_fm import ConditionalMetaFlowMatchingDataset
from graph_ot_fm import GraphStructure, compute_cost_matrix
from scipy.linalg import expm

R55_np = make_grid_graph(5, 5)
graph55 = GraphStructure(R55_np)
rng = np.random.default_rng(42)

pairs = []
for _ in range(20):
    src_nodes = rng.choice(N, size=2, replace=False)
    weights = rng.dirichlet([2.0, 2.0])
    mu_src = np.ones(N) * 0.2 / N
    for nd, w in zip(src_nodes, weights): mu_src[nd] += 0.8 * w
    mu_src /= mu_src.sum()
    tau_d = 0.8
    mu_obs = mu_src @ expm(tau_d * R55_np)
    mu_obs = np.clip(mu_obs, 1e-12, None); mu_obs /= mu_obs.sum()
    pairs.append({'mu_source': mu_src, 'mu_obs': mu_obs, 'tau_diff': tau_d})

from torch.utils.data import DataLoader
ds = ConditionalMetaFlowMatchingDataset(graph55, pairs, n_samples=1000, seed=42)
dl_0w = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
dl_2w = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2, persistent_workers=True)

def drain(dl, n_batches=16):
    it = iter(dl)
    for _ in range(n_batches):
        next(it)

# warmup
drain(dl_0w); drain(dl_2w)

t0 = time.perf_counter()
for _ in range(5): drain(dl_0w, 16)
ms_0w = (time.perf_counter() - t0) / 5 / 16 * 1000
print(f"  {'DataLoader batch  num_workers=0':<55s} {ms_0w:8.3f} ms/batch")

t0 = time.perf_counter()
for _ in range(5): drain(dl_2w, 16)
ms_2w = (time.perf_counter() - t0) / 5 / 16 * 1000
print(f"  {'DataLoader batch  num_workers=2':<55s} {ms_2w:8.3f} ms/batch")

# ── Summary ───────────────────────────────────────────────────────────────────

sep("Summary: where does the 14ms go?")
print(f"""
  Component breakdown (B=64, N=25):
    B  Batch.from_data_list      ~0.33 ms  ({0.33/full_step_ms*100:.0f}% of full step)
    F  4×PyG MP layers           (see above)
    G  4×manual MP layers        (see above)
    H  edge MLP + assembly       (see above)
    Total forward                {fwd_ms:.2f} ms
    Backward                     {fwd_bwd_ms - fwd_ms:.2f} ms
    Optimizer                    {full_step_ms - fwd_bwd_ms:.2f} ms
    Full step                    {full_step_ms:.2f} ms
""")
