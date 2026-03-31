"""
Breakdown of training step time vs number of GNN layers.

Usage:
    uv run python benchmark_layers.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import torch
from torch.utils.data import DataLoader, TensorDataset

from graph_ot_fm import make_grid_graph
from meta_fm import ConditionalGNNRateMatrixPredictor, get_device
from meta_fm.model import rate_matrix_to_edge_index

torch.manual_seed(42)
N_REPS = 200
device = get_device()

R55 = make_grid_graph(5, 5)
ei   = rate_matrix_to_edge_index(R55).to(device)
N    = 25
B    = 256
off_diag = ~torch.eye(N, dtype=torch.bool, device=device)

mu  = torch.rand(B, N, device=device); mu /= mu.sum(1, keepdim=True)
t   = torch.rand(B, 1, device=device)
ctx = torch.rand(B, N, 2, device=device)
R_d = torch.rand(B, N, N, device=device)


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


def sep(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ── 1. Time vs n_layers ───────────────────────────────────────────────────────

sep(f"1. Training step breakdown vs n_layers  (B=256, N=25, hidden=64, {device})")
print(f"  {'n_layers':>8}  {'fwd':>8}  {'bwd':>8}  {'opt':>8}  {'total':>8}  "
      f"{'fwd%':>6}  {'bwd%':>6}  {'ex8 500ep':>10}")
print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*10}")

for n_layers in [1, 2, 3, 4, 6, 8]:
    model = ConditionalGNNRateMatrixPredictor(
        edge_index=ei, n_nodes=N, context_dim=2, hidden_dim=64, n_layers=n_layers
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-4)

    # warmup
    for _ in range(5):
        R_p = model(mu, t, ctx)
        (R_p - R_d).pow(2)[:, off_diag].mean().backward()
        opt.zero_grad(); opt.step()

    fwd_ms  = timeit(lambda: model(mu, t, ctx))

    R_p  = model(mu, t, ctx)
    loss = (R_p - R_d).pow(2)[:, off_diag].mean()
    bwd_ms  = timeit(lambda: loss.backward(retain_graph=True))

    opt_ms  = timeit(lambda: opt.step())

    step_ms = timeit(lambda: (
        setattr(model, '_rp', model(mu, t, ctx)),
        (model._rp - R_d).pow(2)[:, off_diag].mean().backward(),
        opt.zero_grad(), opt.step()
    ))

    n_steps     = 5000 // B
    ep_s        = step_ms * n_steps / 1000
    total_min   = ep_s * 500 / 60

    print(f"  {n_layers:>8}  {fwd_ms:>7.2f}ms {bwd_ms:>7.2f}ms {opt_ms:>7.2f}ms "
          f"{step_ms:>7.2f}ms  {fwd_ms/step_ms*100:>5.1f}%  {bwd_ms/step_ms*100:>5.1f}%  "
          f"{total_min:>8.1f}min")

# ── 2. Time vs hidden_dim ─────────────────────────────────────────────────────

sep(f"2. Training step breakdown vs hidden_dim  (B=256, N=25, 4 layers, {device})")
print(f"  {'hidden':>8}  {'fwd':>8}  {'bwd':>8}  {'opt':>8}  {'total':>8}  "
      f"{'params':>8}  {'ex8 500ep':>10}")
print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

for hidden_dim in [32, 64, 96, 128, 192, 256]:
    model = ConditionalGNNRateMatrixPredictor(
        edge_index=ei, n_nodes=N, context_dim=2, hidden_dim=hidden_dim, n_layers=4
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-4)
    n_params = sum(p.numel() for p in model.parameters())

    for _ in range(5):
        R_p = model(mu, t, ctx)
        (R_p - R_d).pow(2)[:, off_diag].mean().backward()
        opt.zero_grad(); opt.step()

    fwd_ms  = timeit(lambda: model(mu, t, ctx))
    R_p  = model(mu, t, ctx)
    loss = (R_p - R_d).pow(2)[:, off_diag].mean()
    bwd_ms  = timeit(lambda: loss.backward(retain_graph=True))
    opt_ms  = timeit(lambda: opt.step())
    step_ms = timeit(lambda: (
        setattr(model, '_rp', model(mu, t, ctx)),
        (model._rp - R_d).pow(2)[:, off_diag].mean().backward(),
        opt.zero_grad(), opt.step()
    ))

    n_steps   = 5000 // B
    total_min = step_ms * n_steps * 500 / 1000 / 60

    print(f"  {hidden_dim:>8}  {fwd_ms:>7.2f}ms {bwd_ms:>7.2f}ms {opt_ms:>7.2f}ms "
          f"{step_ms:>7.2f}ms  {n_params:>8,}  {total_min:>8.1f}min")

# ── 3. What fraction is data loading vs compute? ──────────────────────────────

sep(f"3. DataLoader overhead vs compute  (B=256, 4 layers, hidden=64, {device})")

# Simulate the training loop — how much time is data loading vs step?
model = ConditionalGNNRateMatrixPredictor(
    edge_index=ei, n_nodes=N, context_dim=2, hidden_dim=64, n_layers=4).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-4)

# Pre-load data to device (TensorDataset)
n_train = 5000
mu_all  = torch.rand(n_train, N, device=device)
mu_all /= mu_all.sum(1, keepdim=True)
t_all   = torch.rand(n_train, 1, device=device)
ctx_all = torch.rand(n_train, N, 2, device=device)
R_all   = torch.rand(n_train, N, N, device=device)

ds_dev = TensorDataset(mu_all, t_all, ctx_all, R_all)
dl_dev = DataLoader(ds_dev, batch_size=B, shuffle=True, num_workers=0)

# CPU dataset with .to(device) per batch
mu_cpu  = mu_all.cpu(); t_cpu = t_all.cpu()
ctx_cpu = ctx_all.cpu(); R_cpu = R_all.cpu()
ds_cpu = TensorDataset(mu_cpu, t_cpu, ctx_cpu, R_cpu)
dl_cpu = DataLoader(ds_cpu, batch_size=B, shuffle=True, num_workers=0)

def run_epoch(dl, on_device):
    total = 0.0
    n = 0
    for mu_b, t_b, ctx_b, R_b in dl:
        if not on_device:
            mu_b = mu_b.to(device); t_b = t_b.to(device)
            ctx_b = ctx_b.to(device); R_b = R_b.to(device)
        R_p = model(mu_b, t_b, ctx_b)
        loss = (R_p - R_b).pow(2)[:, off_diag].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item(); n += 1
    return total / n

# warmup
run_epoch(dl_dev, True); run_epoch(dl_dev, True)

sync()
t0 = time.perf_counter()
for _ in range(10): run_epoch(dl_dev, True)
sync()
dev_ms = (time.perf_counter() - t0) / 10 / (n_train // B) * 1000

sync()
t0 = time.perf_counter()
for _ in range(10): run_epoch(dl_cpu, False)
sync()
cpu_ms = (time.perf_counter() - t0) / 10 / (n_train // B) * 1000

# Pure compute (no DataLoader iteration)
batches = [(mu_all[i:i+B], t_all[i:i+B], ctx_all[i:i+B], R_all[i:i+B])
           for i in range(0, n_train, B)]
sync()
t0 = time.perf_counter()
for _ in range(10):
    for mu_b, t_b, ctx_b, R_b in batches:
        R_p = model(mu_b, t_b, ctx_b)
        loss = (R_p - R_b).pow(2)[:, off_diag].mean()
        opt.zero_grad(); loss.backward(); opt.step()
sync()
compute_ms = (time.perf_counter() - t0) / 10 / len(batches) * 1000

print(f"  {'pure compute (pre-sliced batches)':<45} {compute_ms:7.2f} ms/step")
print(f"  {'TensorDataset on device (DataLoader)':<45} {dev_ms:7.2f} ms/step  "
      f"({dev_ms/compute_ms:.2f}× overhead)")
print(f"  {'CPU dataset + .to(device) per batch':<45} {cpu_ms:7.2f} ms/step  "
      f"({cpu_ms/compute_ms:.2f}× overhead)")
