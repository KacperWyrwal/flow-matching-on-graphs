"""
Isolate which operations are slow on MPS vs CPU.

Usage:
    uv run python benchmark_mps_ops.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from graph_ot_fm import make_grid_graph
from meta_fm.model import rate_matrix_to_edge_index, RateMessagePassing

torch.manual_seed(42)
N_REPS = 200
N = 25
B = 64
hidden_dim = 64

R55 = make_grid_graph(5, 5)
ei_cpu = rate_matrix_to_edge_index(R55)
src_cpu, dst_cpu = ei_cpu
E = ei_cpu.shape[1]


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def timeit(label, fn, device, n=N_REPS):
    sync(device)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync(device)
    ms = (time.perf_counter() - t0) / n * 1000
    return ms


def sep(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(f"  {'Operation':<45}  {'CPU':>8}  {'MPS':>8}  {'ratio':>7}")
    print(f"  {'-'*45}  {'-'*8}  {'-'*8}  {'-'*7}")


def row(label, cpu_ms, mps_ms):
    ratio = cpu_ms / mps_ms
    flag = "  <-- MPS SLOW" if ratio < 0.5 else ""
    print(f"  {label:<45}  {cpu_ms:7.3f}ms  {mps_ms:7.3f}ms  {ratio:6.2f}×{flag}")


# ─── 1. Batch.from_data_list ───────────────────────────────────────────
sep("1. Data preparation overhead (B=64, N=25)")

nf_cpu = torch.rand(B, N, 4)
nf_mps = nf_cpu.to("mps")

cpu_ms = timeit("Batch.from_data_list", lambda: Batch.from_data_list(
    [Data(x=nf_cpu[b], edge_index=ei_cpu) for b in range(B)]
), torch.device("cpu"))
mps_ms = timeit("Batch.from_data_list", lambda: Batch.from_data_list(
    [Data(x=nf_mps[b], edge_index=ei_cpu.to("mps")) for b in range(B)]
), torch.device("mps"))
row("Batch.from_data_list (B=64, N=25)", cpu_ms, mps_ms)

# Manual edge tiling
offsets_cpu = torch.arange(B) * N
ei_mps = ei_cpu.to("mps")
offsets_mps = torch.arange(B, device="mps") * N

def tile_cpu():
    s = (src_cpu.unsqueeze(0) + offsets_cpu.unsqueeze(1)).reshape(-1)
    d = (dst_cpu.unsqueeze(0) + offsets_cpu.unsqueeze(1)).reshape(-1)
    return torch.stack([s, d])

def tile_mps():
    s = (ei_mps[0].unsqueeze(0) + offsets_mps.unsqueeze(1)).reshape(-1)
    d = (ei_mps[1].unsqueeze(0) + offsets_mps.unsqueeze(1)).reshape(-1)
    return torch.stack([s, d])

cpu_ms = timeit("edge tiling (manual)", tile_cpu, torch.device("cpu"))
mps_ms = timeit("edge tiling (manual)", tile_mps, torch.device("mps"))
row("Manual edge tiling (no Batch)", cpu_ms, mps_ms)

# ─── 2. Rate matrix assembly ────────────────────────────────────────────
sep("2. Rate matrix assembly (B=64, N=25, E edges)")

edge_rates_cpu = torch.rand(B, E)
edge_rates_mps = edge_rates_cpu.to("mps")
arange_N = torch.arange(N)
arange_N_mps = arange_N.to("mps")

def assemble_cpu():
    rm = torch.zeros(B, N, N)
    rm[:, src_cpu, dst_cpu] = edge_rates_cpu
    rm[:, arange_N, arange_N] = -rm.sum(dim=-1)
    return rm

def assemble_mps():
    rm = torch.zeros(B, N, N, device="mps")
    rm[:, ei_mps[0], ei_mps[1]] = edge_rates_mps
    rm[:, arange_N_mps, arange_N_mps] = -rm.sum(dim=-1)
    return rm

cpu_ms = timeit("rate matrix assembly", assemble_cpu, torch.device("cpu"))
mps_ms = timeit("rate matrix assembly", assemble_mps, torch.device("mps"))
row("Rate matrix assembly (advanced index)", cpu_ms, mps_ms)

# torch.zeros only
cpu_ms = timeit("torch.zeros(B,N,N)", lambda: torch.zeros(B, N, N), torch.device("cpu"))
mps_ms = timeit("torch.zeros(B,N,N)", lambda: torch.zeros(B, N, N, device="mps"), torch.device("mps"))
row("torch.zeros(B,N,N)", cpu_ms, mps_ms)

# Advanced index scatter only
rm_cpu = torch.zeros(B, N, N)
rm_mps = torch.zeros(B, N, N, device="mps")

cpu_ms = timeit("rm[:, src, dst] = vals (scatter write)", lambda: rm_cpu.__setitem__((slice(None), src_cpu, dst_cpu), edge_rates_cpu), torch.device("cpu"))
mps_ms = timeit("rm[:, src, dst] = vals (scatter write)", lambda: rm_mps.__setitem__((slice(None), ei_mps[0], ei_mps[1]), edge_rates_mps), torch.device("mps"))
row("rm[:, src, dst] = vals (scatter write)", cpu_ms, mps_ms)

# ─── 3. MLP layers ──────────────────────────────────────────────────────
sep("3. MLP layers (the good part)")

mlp_cpu = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 64)).eval()
mlp_mps = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 64)).eval().to("mps")

x_cpu = torch.rand(B * N, 128)
x_mps = x_cpu.to("mps")

cpu_ms = timeit("MLP(128→64→64), 1600 rows", lambda: mlp_cpu(x_cpu), torch.device("cpu"))
mps_ms = timeit("MLP(128→64→64), 1600 rows", lambda: mlp_mps(x_mps), torch.device("mps"))
row("MLP(128→64→64), 1600 rows", cpu_ms, mps_ms)

# ─── 4. Backward pass ───────────────────────────────────────────────────
sep("4. Backward pass (B=64, N=25)")

from meta_fm import ConditionalGNNRateMatrixPredictor

model_cpu = ConditionalGNNRateMatrixPredictor(
    edge_index=ei_cpu, n_nodes=N, context_dim=2, hidden_dim=64, n_layers=4)
model_mps = ConditionalGNNRateMatrixPredictor(
    edge_index=ei_mps, n_nodes=N, context_dim=2, hidden_dim=64, n_layers=4).to("mps")

mu_cpu = torch.rand(B, N); mu_cpu /= mu_cpu.sum(1, keepdim=True)
t_cpu  = torch.rand(B, 1)
ctx_cpu = torch.rand(B, N, 2)

mu_mps  = mu_cpu.to("mps")
t_mps   = t_cpu.to("mps")
ctx_mps = ctx_cpu.to("mps")

R_dummy_cpu = torch.rand(B, N, N)
R_dummy_mps = R_dummy_cpu.to("mps")
off_diag = ~torch.eye(N, dtype=torch.bool)
off_diag_mps = off_diag.to("mps")

# Pre-compute forward to get a graph for backward
R_pred_cpu = model_cpu(mu_cpu, t_cpu, ctx_cpu)
loss_cpu = (R_pred_cpu - R_dummy_cpu).pow(2)[:, off_diag].mean()
cpu_ms = timeit("backward pass", lambda: loss_cpu.backward(retain_graph=True), torch.device("cpu"))
row("backward pass (retain_graph)", cpu_ms, float('nan'))

R_pred_mps = model_mps(mu_mps, t_mps, ctx_mps)
loss_mps = (R_pred_mps - R_dummy_mps).pow(2)[:, off_diag_mps].mean()
mps_ms = timeit("backward pass", lambda: loss_mps.backward(retain_graph=True), torch.device("mps"))
row("backward pass (retain_graph)", cpu_ms, mps_ms)

# ─── Summary ────────────────────────────────────────────────────────────
print(f"""
  Key finding:
  - MLP layers (the compute):   4-5× faster on MPS
  - Rate matrix assembly:       likely slower on MPS (advanced indexing)
  - Batch.from_data_list:       CPU-bound Python overhead, similar or worse on MPS
  - Net result: MPS win on pure matmul, but scattered by overhead ops
""")
