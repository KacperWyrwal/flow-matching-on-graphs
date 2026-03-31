"""
Benchmark remaining bottlenecks after MPS + batch_size fixes.

Usage:
    uv run python benchmark_remaining.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.linalg import expm

from graph_ot_fm import make_grid_graph, GraphStructure
from meta_fm import (
    ConditionalGNNRateMatrixPredictor,
    ConditionalMetaFlowMatchingDataset,
    get_device,
)
from meta_fm.model import rate_matrix_to_edge_index

torch.manual_seed(42)
N_REPS = 100
device = get_device()

R55 = make_grid_graph(5, 5)
ei = rate_matrix_to_edge_index(R55).to(device)
N = 25
off_diag = ~torch.eye(N, dtype=torch.bool, device=device)


def sep(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def sync():
    if device.type == "mps":
        torch.mps.synchronize()


def timeit(fn, n=N_REPS):
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t0) / n * 1000


# ── 1. Training step breakdown on MPS ────────────────────────────────────────

sep(f"1. Training step breakdown  (B=256, N=25, device={device})")

model = ConditionalGNNRateMatrixPredictor(
    edge_index=ei, n_nodes=N, context_dim=2, hidden_dim=64, n_layers=4
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

B = 256
mu  = torch.rand(B, N, device=device); mu /= mu.sum(1, keepdim=True)
t   = torch.rand(B, 1, device=device)
ctx = torch.rand(B, N, 2, device=device)
R_d = torch.rand(B, N, N, device=device)

# warmup
for _ in range(5):
    R_p = model(mu, t, ctx)
    (R_p - R_d).pow(2)[:, off_diag].mean().backward()
    optimizer.zero_grad(); optimizer.step()

fwd_ms   = timeit(lambda: model(mu, t, ctx))
R_p = model(mu, t, ctx)
loss = (R_p - R_d).pow(2)[:, off_diag].mean()
bwd_ms   = timeit(lambda: loss.backward(retain_graph=True))
opt_ms   = timeit(lambda: optimizer.step())
step_ms  = timeit(lambda: (
    model(mu, t, ctx),
    (R_p - R_d).pow(2)[:, off_diag].mean().backward(),
    optimizer.zero_grad(),
    optimizer.step()
))

print(f"  {'forward':<40} {fwd_ms:7.2f} ms")
print(f"  {'backward':<40} {bwd_ms:7.2f} ms")
print(f"  {'optimizer.step':<40} {opt_ms:7.2f} ms")
print(f"  {'full step (measured together)':<40} {step_ms:7.2f} ms  (← use this one)")

# ── 2. .to(device) cost per batch ────────────────────────────────────────────

sep("2. .to(device) cost  (B=256, dataset on CPU)")

mu_cpu  = torch.rand(B, N)
t_cpu   = torch.rand(B, 1)
ctx_cpu = torch.rand(B, N, 2)
R_cpu   = torch.rand(B, N, N)

to_ms = timeit(lambda: (
    mu_cpu.to(device),
    t_cpu.to(device),
    ctx_cpu.to(device),
    R_cpu.to(device),
))
print(f"  {'4× .to(device) per batch  (B=256, N=25)':<50} {to_ms:7.3f} ms/batch")
print(f"  {'as % of full step':<50} {to_ms/step_ms*100:6.1f}%")

# Pre-pin on device — what if dataset lives on GPU?
mu_dev  = mu_cpu.to(device)
t_dev   = t_cpu.to(device)
ctx_dev = ctx_cpu.to(device)
R_dev   = R_cpu.to(device)
already_ms = timeit(lambda: (mu_dev, t_dev, ctx_dev, R_dev))  # no-op
print(f"  {'no transfer (already on device)':<50} {already_ms:7.3f} ms/batch")

# ── 3. DataLoader overhead ────────────────────────────────────────────────────

sep("3. DataLoader batch overhead  (B=256, num_workers=0)")

graph55 = GraphStructure(R55)
rng = np.random.default_rng(42)
pairs = []
for _ in range(50):
    src_nodes = rng.choice(N, size=2, replace=False)
    w = rng.dirichlet([2.0, 2.0])
    mu_src = np.ones(N) * 0.2 / N
    for nd, ww in zip(src_nodes, w): mu_src[nd] += 0.8 * ww
    mu_src /= mu_src.sum()
    tau_d = 0.8
    mu_obs = mu_src @ expm(tau_d * R55)
    mu_obs = np.clip(mu_obs, 1e-12, None); mu_obs /= mu_obs.sum()
    pairs.append({'mu_source': mu_src, 'mu_obs': mu_obs, 'tau_diff': tau_d})

print("  Building dataset (5000 samples)...", end=' ', flush=True)
t0 = time.perf_counter()
ds = ConditionalMetaFlowMatchingDataset(graph55, pairs, n_samples=5000, seed=42)
build_ms = (time.perf_counter() - t0) * 1000
print(f"{build_ms:.0f} ms")

dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
batches = list(dl)  # pre-fetch all batches

t0 = time.perf_counter()
for _ in range(20):
    for b in batches:
        pass
loader_ms = (time.perf_counter() - t0) / 20 / len(batches) * 1000
print(f"  {'DataLoader.__iter__ per batch (num_workers=0)':<50} {loader_ms:7.3f} ms/batch")
print(f"  {'as % of full step':<50} {loader_ms/step_ms*100:6.1f}%")

# ── 4. Pre-load dataset to device ────────────────────────────────────────────

sep("4. Pre-loaded TensorDataset (all data on device)")

mu_all  = torch.stack([s[0] for s in ds]).to(device)
tau_all = torch.stack([s[1] for s in ds]).to(device)
ctx_all = torch.stack([s[2] for s in ds]).to(device)
R_all   = torch.stack([s[3] for s in ds]).to(device)

ds_dev = TensorDataset(mu_all, tau_all, ctx_all, R_all)
dl_dev = DataLoader(ds_dev, batch_size=256, shuffle=True, num_workers=0)
batches_dev = list(dl_dev)

t0 = time.perf_counter()
for _ in range(20):
    for b in batches_dev:
        pass
loader_dev_ms = (time.perf_counter() - t0) / 20 / len(batches_dev) * 1000
print(f"  {'TensorDataset on device per batch':<50} {loader_dev_ms:7.3f} ms/batch")
print(f"  {'speedup vs DataLoader from CPU':<50} {loader_ms/loader_dev_ms:7.2f}×")

# ── 5. Projected training time ────────────────────────────────────────────────

sep(f"5. Projected ex8 training time  (5000 samples, 500 epochs, {device})")

n_steps = len(batches)  # steps per epoch at B=256
overhead_per_step = to_ms + loader_ms
overhead_no_transfer = loader_dev_ms

print(f"  steps per epoch (B=256):  {n_steps}")
print(f"  step time (compute only): {step_ms:.2f} ms")
print(f"  + .to(device) per batch:  {to_ms:.3f} ms")
print(f"  + DataLoader overhead:    {loader_ms:.3f} ms")
total_ms = step_ms + to_ms + loader_ms
total_min = total_ms * n_steps * 500 / 1000 / 60
print(f"  total per step:           {total_ms:.2f} ms  →  {total_min:.1f} min")

total_ms_preload = step_ms + loader_dev_ms  # no .to() if on device
total_min_preload = total_ms_preload * n_steps * 500 / 1000 / 60
print(f"  with pre-loaded dataset:  {total_ms_preload:.2f} ms  →  {total_min_preload:.1f} min")
