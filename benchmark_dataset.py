"""
Benchmark dataset construction pipeline — specifically the rate matrix computation.

Usage:
    uv run python benchmark_dataset.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
from contextlib import contextmanager
from scipy.linalg import expm

from graph_ot_fm import (
    GraphStructure, make_grid_graph, make_cycle_graph, make_path_graph,
)
from graph_ot_fm.geodesic_cache import GeodesicCache
from graph_ot_fm.flow import marginal_distribution_fast, marginal_rate_matrix_fast
from graph_ot_fm import compute_cost_matrix, compute_ot_coupling


@contextmanager
def timer(label, n_iters=1):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    ms = elapsed / n_iters * 1000
    print(f"  {label:<55s} {ms:8.3f} ms/iter")


def sep(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def make_multipeak_dist(N, n_peaks, rng):
    nodes = rng.choice(N, size=n_peaks, replace=False)
    weights = rng.dirichlet(np.full(n_peaks, 2.0))
    dist = np.ones(N) * 0.2 / N
    for node, w in zip(nodes, weights):
        dist[node] += 0.8 * w
    dist = np.clip(dist, 1e-6, None)
    return dist / dist.sum()


rng = np.random.default_rng(42)
N_REPS = 200

# ── Test graphs ──────────────────────────────────────────────────────────────
graphs = {
    'grid_3x3 (N=9)':  make_grid_graph(3, 3),
    'grid_4x4 (N=16)': make_grid_graph(4, 4),
    'grid_5x5 (N=25)': make_grid_graph(5, 5),
    'cycle_12 (N=12)': make_cycle_graph(12),
    'path_10  (N=10)': make_path_graph(10),
}

# ── 1. Cache + OT coupling construction ──────────────────────────────────────

sep("1. One-time setup cost (per graph, paid once at dataset build)")

for name, R in graphs.items():
    N = R.shape[0]
    graph = GraphStructure(R)

    t0 = time.perf_counter()
    cache = GeodesicCache(graph)
    cost = compute_cost_matrix(graph)
    elapsed_setup = (time.perf_counter() - t0) * 1000
    print(f"  GeodesicCache + cost_matrix  {name:<20s}  {elapsed_setup:.2f} ms")

# ── 2. Per-pair cost: OT coupling + precompute_for_coupling ──────────────────

sep("2. Per source-obs pair cost (paid n_pairs_per_graph times per graph)")

for name, R in graphs.items():
    N = R.shape[0]
    graph = GraphStructure(R)
    cache = GeodesicCache(graph)
    cost = compute_cost_matrix(graph)

    # Pre-make distributions
    mu_a = make_multipeak_dist(N, 2, rng)
    mu_b = make_multipeak_dist(N, 1, rng)
    tau_diff = 0.8
    mu_obs = mu_a @ expm(tau_diff * R)
    mu_obs = np.clip(mu_obs, 1e-12, None); mu_obs /= mu_obs.sum()

    # OT coupling
    with timer(f"compute_ot_coupling            {name}", N_REPS):
        for _ in range(N_REPS):
            pi = compute_ot_coupling(mu_obs, mu_a, cost)

    pi = compute_ot_coupling(mu_obs, mu_a, cost)
    n_nonzero = np.sum(pi > 1e-12)

    # precompute_for_coupling
    with timer(f"precompute_for_coupling        {name} ({n_nonzero} pairs)", N_REPS):
        for _ in range(N_REPS):
            fresh_cache = GeodesicCache(graph)
            fresh_cache.precompute_for_coupling(pi)

# ── 3. Per-sample cost: marginal_distribution_fast + marginal_rate_matrix_fast

sep("3. Per-sample cost (paid n_samples_per_graph times per graph)")

for name, R in graphs.items():
    N = R.shape[0]
    graph = GraphStructure(R)
    cache = GeodesicCache(graph)
    cost = compute_cost_matrix(graph)

    mu_a = make_multipeak_dist(N, 2, rng)
    mu_b = make_multipeak_dist(N, 1, rng)
    tau_diff = 0.8
    mu_obs = mu_a @ expm(tau_diff * R)
    mu_obs = np.clip(mu_obs, 1e-12, None); mu_obs /= mu_obs.sum()
    pi = compute_ot_coupling(mu_obs, mu_a, cost)
    cache.precompute_for_coupling(pi)
    n_nonzero = np.sum(pi > 1e-12)
    tau = 0.5

    with timer(f"marginal_distribution_fast     {name} ({n_nonzero} pairs)", N_REPS):
        for _ in range(N_REPS):
            marginal_distribution_fast(cache, pi, tau)

    with timer(f"marginal_rate_matrix_fast      {name} ({n_nonzero} pairs)", N_REPS):
        for _ in range(N_REPS):
            marginal_rate_matrix_fast(cache, pi, tau)

    # Breakdown: what fraction of rate_matrix_fast is the inner branch loop?
    R_mat = marginal_rate_matrix_fast(cache, pi, tau)
    total_branches = sum(
        len(cache.branch_structure.get((int(i), int(j)), []))
        for i, j in np.argwhere(pi > 1e-12)
        if int(i) != int(j)
    )
    print(f"    └─ total branch_structure entries for this coupling: {total_branches}")

# ── 4. marginal_rate_matrix_fast inner loop breakdown ────────────────────────

sep("4. Inner loop breakdown in marginal_rate_matrix_fast (grid_5x5)")

R55 = make_grid_graph(5, 5)
N = 25
graph55 = GraphStructure(R55)
cache55 = GeodesicCache(graph55)
cost55 = compute_cost_matrix(graph55)

mu_src = make_multipeak_dist(N, 3, rng)
mu_obs55 = mu_src @ expm(0.8 * R55)
mu_obs55 = np.clip(mu_obs55, 1e-12, None); mu_obs55 /= mu_obs55.sum()
pi55 = compute_ot_coupling(mu_obs55, mu_src, cost55)
cache55.precompute_for_coupling(pi55)
tau = 0.5

# Time each part of marginal_rate_matrix_fast separately
from scipy.stats import binom as binom_dist

nonzero_pairs = np.argwhere(pi55 > 1e-12)
print(f"  Nonzero pairs in coupling: {len(nonzero_pairs)}")

# Part A: conditional marginal computation
with timer("  Part A: build p_t (all conditionals)", N_REPS):
    for _ in range(N_REPS):
        p_t = np.zeros(N)
        conditionals = {}
        binom_cache = {}
        for i, j in nonzero_pairs:
            i, j = int(i), int(j)
            pi_ij = pi55[i, j]
            if i == j:
                cond = np.zeros(N); cond[i] = 1.0
            else:
                d = int(graph55.dist[i, j])
                if d not in binom_cache:
                    binom_cache[d] = binom_dist.pmf(np.arange(d + 1), d, tau)
                cond = cache55.conditional_marginal(i, j, tau, binom_pmf=binom_cache[d])
            conditionals[(i, j)] = cond
            p_t += pi_ij * cond

# Part B: branch structure accumulation (the inner Python loop)
p_t = np.zeros(N)
conditionals = {}
binom_cache = {}
for i, j in nonzero_pairs:
    i, j = int(i), int(j)
    pi_ij = pi55[i, j]
    if i == j:
        cond = np.zeros(N); cond[i] = 1.0
    else:
        d = int(graph55.dist[i, j])
        if d not in binom_cache:
            binom_cache[d] = binom_dist.pmf(np.arange(d + 1), d, tau)
        cond = cache55.conditional_marginal(i, j, tau, binom_pmf=binom_cache[d])
    conditionals[(i, j)] = cond
    p_t += pi_ij * cond

inv_1mt = 1.0 / (1.0 - tau)
with timer("  Part B: branch loop accumulation", N_REPS):
    for _ in range(N_REPS):
        u_t = np.zeros((N, N))
        for i, j in nonzero_pairs:
            i, j = int(i), int(j)
            if i == j:
                continue
            if (i, j) not in cache55.branch_structure:
                continue
            pi_ij = pi55[i, j]
            cond = conditionals[(i, j)]
            for a, b, weight in cache55.branch_structure[(i, j)]:
                if p_t[a] > 1e-12:
                    u_t[a, b] += pi_ij * (cond[a] / p_t[a]) * weight * inv_1mt

total_branches = sum(
    len(cache55.branch_structure.get((int(i), int(j)), []))
    for i, j in nonzero_pairs if int(i) != int(j)
)
print(f"  Total branch entries iterated per call: {total_branches}")
print(f"  Nonzero off-diagonal pairs: {sum(1 for i,j in nonzero_pairs if int(i)!=int(j))}")

# ── 5. Full dataset construction scaling ────────────────────────────────────

sep("5. Full dataset construction time estimates")

for name, R in graphs.items():
    N = R.shape[0]
    graph = GraphStructure(R)
    cache = GeodesicCache(graph)
    cost = compute_cost_matrix(graph)

    # Use a representative coupling
    mu_src = make_multipeak_dist(N, 2, rng)
    mu_obs = mu_src @ expm(0.8 * R)
    mu_obs = np.clip(mu_obs, 1e-12, None); mu_obs /= mu_obs.sum()
    pi = compute_ot_coupling(mu_obs, mu_src, cost)
    cache.precompute_for_coupling(pi)

    # Measure per-sample cost
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        marginal_distribution_fast(cache, pi, 0.5)
        marginal_rate_matrix_fast(cache, pi, 0.5)
    per_sample_ms = (time.perf_counter() - t0) / N_REPS * 1000

    # Measure per-pair cost (ot + precompute)
    t0 = time.perf_counter()
    for _ in range(50):
        p = compute_ot_coupling(mu_obs, mu_src, cost)
        fc = GeodesicCache(graph); fc.precompute_for_coupling(p)
    per_pair_ms = (time.perf_counter() - t0) / 50 * 1000

    n_samples = 1000
    n_pairs = 50
    est_ms = n_pairs * per_pair_ms + n_samples * per_sample_ms
    print(f"  {name:<20s}  pair={per_pair_ms:.2f}ms  sample={per_sample_ms:.3f}ms  "
          f"total/graph≈{est_ms/1000:.1f}s")

print(f"\n  For 13 graphs: multiply above by 13")
