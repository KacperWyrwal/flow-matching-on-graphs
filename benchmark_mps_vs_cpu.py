"""
Benchmark MPS vs CPU for the GNN training loop.

Usage:
    uv run python benchmark_mps_vs_cpu.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from graph_ot_fm import make_grid_graph
from meta_fm import ConditionalGNNRateMatrixPredictor
from meta_fm.model import rate_matrix_to_edge_index

torch.manual_seed(42)
N_REPS = 100

R55 = make_grid_graph(5, 5)
ei55_cpu = rate_matrix_to_edge_index(R55)
N = 25
off_diag = ~torch.eye(N, dtype=torch.bool)


def sep(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def benchmark_device(device_name, B_values=(64, 128, 256)):
    device = torch.device(device_name)
    ei = ei55_cpu.to(device)

    model = ConditionalGNNRateMatrixPredictor(
        edge_index=ei, n_nodes=N, context_dim=2, hidden_dim=64, n_layers=4
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    off_diag_dev = off_diag.to(device)

    results = {}
    for B in B_values:
        mu  = torch.rand(B, N, device=device); mu /= mu.sum(1, keepdim=True)
        t   = torch.rand(B, 1, device=device)
        ctx = torch.rand(B, N, 2, device=device)
        R_dummy = torch.rand(B, N, N, device=device)

        # Warmup
        for _ in range(5):
            R_pred = model(mu, t, ctx)
            loss = (R_pred - R_dummy).pow(2)[:, off_diag_dev].mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        # Sync helper
        def sync():
            if device_name == "mps":
                torch.mps.synchronize()
            # cpu has no async ops

        # --- Forward only ---
        sync()
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            with torch.no_grad():
                model(mu, t, ctx)
        sync()
        fwd_ms = (time.perf_counter() - t0) / N_REPS * 1000

        # --- Full step ---
        sync()
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            R_pred = model(mu, t, ctx)
            loss = (R_pred - R_dummy).pow(2)[:, off_diag_dev].mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        sync()
        step_ms = (time.perf_counter() - t0) / N_REPS * 1000

        results[B] = (fwd_ms, step_ms)

    return results


sep("Forward pass (no_grad) — N=25, 4 GNN layers, hidden=64")
print(f"  {'B':>4}  {'CPU fwd':>10}  {'MPS fwd':>10}  {'speedup':>8}")
print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*8}")

cpu_results = benchmark_device("cpu")
mps_results = benchmark_device("mps")

for B in (64, 128, 256):
    cpu_fwd, _ = cpu_results[B]
    mps_fwd, _ = mps_results[B]
    print(f"  {B:4d}  {cpu_fwd:9.2f}ms  {mps_fwd:9.2f}ms  {cpu_fwd/mps_fwd:7.2f}×")

sep("Full training step (fwd+bwd+optim) — N=25")
print(f"  {'B':>4}  {'CPU step':>10}  {'MPS step':>10}  {'speedup':>8}  {'ex8 CPU':>8}  {'ex8 MPS':>8}")
print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")

for B in (64, 128, 256):
    _, cpu_step = cpu_results[B]
    _, mps_step = mps_results[B]
    n_steps = 5000 // B
    cpu_epoch = cpu_step * n_steps / 1000  # seconds
    mps_epoch = mps_step * n_steps / 1000
    cpu_total = cpu_epoch * 500 / 60  # minutes
    mps_total = mps_epoch * 500 / 60
    print(f"  {B:4d}  {cpu_step:9.2f}ms  {mps_step:9.2f}ms  {cpu_step/mps_step:7.2f}×"
          f"  {cpu_total:6.1f}min  {mps_total:6.1f}min")

sep("Message passing breakdown — B=64, N=25")

from meta_fm.model import RateMessagePassing
import torch.nn as nn

B = 64
device_cpu = torch.device("cpu")
device_mps = torch.device("mps")

ei_cpu = ei55_cpu
ei_mps = ei55_cpu.to(device_mps)
src, dst = ei_cpu

offsets_cpu = torch.arange(B) * N
src_b_cpu = (src.unsqueeze(0) + offsets_cpu.unsqueeze(1)).reshape(-1)
dst_b_cpu = (dst.unsqueeze(0) + offsets_cpu.unsqueeze(1)).reshape(-1)
batch_ei_cpu = torch.stack([src_b_cpu, dst_b_cpu])

offsets_mps = torch.arange(B, device=device_mps) * N
src_b_mps = (ei_mps[0].unsqueeze(0) + offsets_mps.unsqueeze(1)).reshape(-1)
dst_b_mps = (ei_mps[1].unsqueeze(0) + offsets_mps.unsqueeze(1)).reshape(-1)
batch_ei_mps = torch.stack([src_b_mps, dst_b_mps])

hidden_dim = 64
mp_cpu = RateMessagePassing(in_dim=hidden_dim, hidden_dim=hidden_dim).to(device_cpu)
mp_mps = RateMessagePassing(in_dim=hidden_dim, hidden_dim=hidden_dim).to(device_mps)
# copy weights so comparison is fair
mp_mps.load_state_dict({k: v.to(device_mps) for k, v in mp_cpu.state_dict().items()})

h_cpu = torch.rand(B * N, hidden_dim)
h_mps = h_cpu.to(device_mps)

# CPU
t0 = time.perf_counter()
for _ in range(N_REPS):
    mp_cpu(h_cpu, batch_ei_cpu)
cpu_mp = (time.perf_counter() - t0) / N_REPS * 1000

# MPS
torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(N_REPS):
    mp_mps(h_mps, batch_ei_mps)
torch.mps.synchronize()
mps_mp = (time.perf_counter() - t0) / N_REPS * 1000

print(f"  {'':55}  {'CPU':>8}  {'MPS':>8}  {'speedup':>8}")
print(f"  {'1 MP layer (B=64, N=25, hidden=64)':<55}  {cpu_mp:7.2f}ms  {mps_mp:7.2f}ms  {cpu_mp/mps_mp:7.2f}×")

mp4_cpu = nn.ModuleList([
    RateMessagePassing(in_dim=4 if i==0 else hidden_dim, hidden_dim=hidden_dim)
    for i in range(4)
])
mp4_mps = nn.ModuleList([
    RateMessagePassing(in_dim=4 if i==0 else hidden_dim, hidden_dim=hidden_dim)
    for i in range(4)
]).to(device_mps)
for l_cpu, l_mps in zip(mp4_cpu, mp4_mps):
    l_mps.load_state_dict({k: v.to(device_mps) for k, v in l_cpu.state_dict().items()})

h0_cpu = torch.rand(B * N, 4)
h0_mps = h0_cpu.to(device_mps)

t0 = time.perf_counter()
for _ in range(N_REPS):
    h = h0_cpu
    for mp_l in mp4_cpu:
        h = mp_l(h, batch_ei_cpu)
cpu_4mp = (time.perf_counter() - t0) / N_REPS * 1000

torch.mps.synchronize()
t0 = time.perf_counter()
for _ in range(N_REPS):
    h = h0_mps
    for mp_l in mp4_mps:
        h = mp_l(h, batch_ei_mps)
torch.mps.synchronize()
mps_4mp = (time.perf_counter() - t0) / N_REPS * 1000

print(f"  {'4 MP layers (B=64, N=25, hidden=64)':<55}  {cpu_4mp:7.2f}ms  {mps_4mp:7.2f}ms  {cpu_4mp/mps_4mp:7.2f}×")
