"""
Experiment 17: OT Transport Generalization Across Topologies and Sizes.

Trains a FlexibleConditionalGNNRateMatrixPredictor on optimal transport flow
matching across a diverse zoo of graph topologies, then tests generalization
to unseen topologies and unseen graph sizes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import networkx as nx
from scipy.linalg import expm
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from graph_ot_fm import (
    GraphStructure,
    GeodesicCache,
    compute_cost_matrix,
    total_variation,
)
from graph_ot_fm.ot_solver import compute_ot_coupling
from graph_ot_fm.flow import marginal_distribution_fast, marginal_rate_matrix_fast

from meta_fm import (
    FlexibleConditionalGNNRateMatrixPredictor,
    train_flexible_conditional,
    EMA,
    get_device,
    sample_trajectory_flexible,
)
from meta_fm.model import rate_matrix_to_edge_index

# Reuse mesh generation from ex16
from experiments.ex16_heat_mesh import (
    generate_mesh,
    mesh_to_graph,
    generate_initial_condition,
    IC_TYPES,
)


# ── Graph Zoo ────────────────────────────────────────────────────────────────

def _ensure_connected(R, pos):
    """Ensure graph is connected by adding edges between nearest nodes of
    different components."""
    adj = (np.abs(R) > 0).astype(float)
    np.fill_diagonal(adj, 0)
    n_comp, labels = connected_components(csr_matrix(adj), directed=False)
    if n_comp <= 1:
        return R
    # Connect components pairwise
    R_new = R.copy()
    np.fill_diagonal(R_new, 0)
    for c in range(1, n_comp):
        nodes_prev = np.where(labels < c)[0]
        nodes_cur = np.where(labels == c)[0]
        best_dist = np.inf
        best_i, best_j = nodes_prev[0], nodes_cur[0]
        for i in nodes_prev:
            for j in nodes_cur:
                d = np.linalg.norm(pos[i] - pos[j])
                if d < best_dist:
                    best_dist = d
                    best_i, best_j = i, j
        R_new[best_i, best_j] = 1.0
        R_new[best_j, best_i] = 1.0
    np.fill_diagonal(R_new, -R_new.sum(axis=1))
    return R_new


def make_grid_graph(rows, cols):
    """2D grid graph. N = rows * cols."""
    N = rows * cols
    pos = np.array([(i % cols, i // cols) for i in range(N)], dtype=np.float32)
    R = np.zeros((N, N))
    for i in range(N):
        r, c = i // cols, i % cols
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                j = nr * cols + nc
                R[i, j] = 1.0
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos


def make_cycle_graph(n):
    """Cycle graph with N nodes on a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    R = np.zeros((n, n))
    for i in range(n):
        R[i, (i + 1) % n] = 1.0
        R[i, (i - 1) % n] = 1.0
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos


def make_random_geometric_graph(n, radius=0.3, rng=None):
    """Random geometric graph: connect points within radius."""
    if rng is None:
        rng = np.random.default_rng()
    pos = rng.uniform(0, 1, size=(n, 2)).astype(np.float32)
    dists = cdist(pos, pos)
    R = np.zeros((n, n))
    R[dists < radius] = 1.0
    np.fill_diagonal(R, 0)
    # Ensure connected
    R = _ensure_connected(R, pos)
    # Set diagonal = -row sum
    np.fill_diagonal(R, 0)
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos


def make_barabasi_albert_graph(n, m=3, rng=None):
    """Barabasi-Albert preferential attachment graph."""
    G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(100000)))
    pos_dict = nx.spring_layout(G, seed=42)
    pos = np.array([pos_dict[i] for i in range(n)], dtype=np.float32)
    A = nx.to_numpy_array(G)
    R = A.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos


def make_watts_strogatz_graph(n, k=4, p=0.2, rng=None):
    """Watts-Strogatz small-world graph."""
    G = nx.watts_strogatz_graph(n, k, p, seed=int(rng.integers(100000)))
    pos_dict = nx.spring_layout(G, seed=42)
    pos = np.array([pos_dict[i] for i in range(n)], dtype=np.float32)
    A = nx.to_numpy_array(G)
    R = A.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos


def make_random_tree(n, rng=None):
    """Random tree via Prufer sequence."""
    G = nx.random_labeled_tree(n, seed=int(rng.integers(100000)))
    pos_dict = nx.spring_layout(G, seed=42)
    pos = np.array([pos_dict[i] for i in range(n)], dtype=np.float32)
    A = nx.to_numpy_array(G)
    R = A.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos


def make_barbell_graph(n_clique, bridge_len=3, rng=None):
    """Two cliques connected by a bridge path."""
    G = nx.barbell_graph(n_clique, bridge_len)
    pos_dict = nx.spring_layout(G, seed=42)
    N = G.number_of_nodes()
    pos = np.array([pos_dict[i] for i in range(N)], dtype=np.float32)
    A = nx.to_numpy_array(G)
    R = A.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos


def make_sbm_graph(sizes, p_in=0.5, p_out=0.05, rng=None):
    """Stochastic block model."""
    k = len(sizes)
    probs = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
    G = nx.stochastic_block_model(sizes, probs,
                                  seed=int(rng.integers(100000)))
    pos_dict = nx.spring_layout(G, seed=42)
    N = G.number_of_nodes()
    pos = np.array([pos_dict[i] for i in range(N)], dtype=np.float32)
    A = nx.to_numpy_array(G)
    R = A.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos


def make_petersen_graph():
    """Petersen graph (10 nodes, 3-regular, non-planar)."""
    G = nx.petersen_graph()
    pos_dict = nx.spring_layout(G, seed=42)
    pos = np.array([pos_dict[i] for i in range(10)], dtype=np.float32)
    A = nx.to_numpy_array(G)
    R = A.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos


def make_ladder_graph(n):
    """Ladder graph: two rows of n nodes each, total 2n nodes."""
    G = nx.ladder_graph(n)
    pos_dict = nx.spring_layout(G, seed=42)
    N = G.number_of_nodes()
    pos = np.array([pos_dict[i] for i in range(N)], dtype=np.float32)
    A = nx.to_numpy_array(G)
    R = A.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos


def make_mesh_graph(shape, n_points=40, rng=None):
    """Mesh graph from 2D triangulation."""
    seed = int(rng.integers(100000)) if rng is not None else 42
    points, triangles, _ = generate_mesh(shape, n_points, seed=seed)
    R = mesh_to_graph(points, triangles)
    return R, points.astype(np.float32)


# ── Graph Generators ─────────────────────────────────────────────────────────

def generate_training_graphs(seed=42):
    """Generate a diverse collection of training graphs.
    Returns list of (name, R, positions) tuples."""
    rng = np.random.default_rng(seed)
    graphs = []

    # 1. Grid graphs
    for rows, cols in [(3, 5), (4, 4), (5, 5), (4, 6), (6, 6)]:
        R, pos = make_grid_graph(rows, cols)
        graphs.append((f'grid_{rows}x{cols}', R, pos))

    # 2. Cycle graphs
    for n in [15, 20, 30]:
        R, pos = make_cycle_graph(n)
        graphs.append((f'cycle_{n}', R, pos))

    # 3. Random geometric graphs
    for n in [20, 30, 40, 50]:
        R, pos = make_random_geometric_graph(n, radius=0.35, rng=rng)
        graphs.append((f'rgg_{n}', R, pos))

    # 4. Barabasi-Albert
    for n in [20, 30, 40]:
        R, pos = make_barabasi_albert_graph(n, m=3, rng=rng)
        graphs.append((f'ba_{n}', R, pos))

    # 5. Small-world (Watts-Strogatz)
    for n in [20, 30, 40]:
        R, pos = make_watts_strogatz_graph(n, k=4, p=0.2, rng=rng)
        graphs.append((f'ws_{n}', R, pos))

    # 6. Random trees
    for n in [15, 25, 35]:
        R, pos = make_random_tree(n, rng=rng)
        graphs.append((f'tree_{n}', R, pos))

    # 7. Mesh graphs from 2D shapes
    for shape in ['square', 'circle', 'triangle']:
        R, pos = make_mesh_graph(shape, n_points=40, rng=rng)
        graphs.append((f'mesh_{shape}_40', R, pos))

    # 8. Barbell graph
    for n_clique in [8, 10, 12]:
        R, pos = make_barbell_graph(n_clique, bridge_len=3, rng=rng)
        graphs.append((f'barbell_{n_clique}', R, pos))

    return graphs


def generate_test_graphs_topology(seed=9000):
    """Unseen topologies at similar sizes to training."""
    rng = np.random.default_rng(seed)
    graphs = []

    # 1. Grid sizes not seen
    for rows, cols in [(3, 7), (5, 6), (7, 4)]:
        R, pos = make_grid_graph(rows, cols)
        graphs.append((f'grid_{rows}x{cols}', R, pos))

    # 2. Ladder graph
    R, pos = make_ladder_graph(15)
    graphs.append(('ladder_30', R, pos))

    # 3. Random geometric with different radius
    for n in [25, 40]:
        R, pos = make_random_geometric_graph(n, radius=0.45, rng=rng)
        graphs.append((f'rgg_r045_{n}', R, pos))

    # 4. Mesh graphs from unseen shapes
    for shape in ['L_shape', 'star', 'annulus']:
        R, pos = make_mesh_graph(shape, n_points=40, rng=rng)
        graphs.append((f'mesh_{shape}_40', R, pos))

    # 5. Stochastic block model
    R, pos = make_sbm_graph(sizes=[10, 10, 10], p_in=0.5, p_out=0.05, rng=rng)
    graphs.append(('sbm_3x10', R, pos))

    # 6. Petersen graph
    R, pos = make_petersen_graph()
    graphs.append(('petersen', R, pos))

    return graphs


def generate_test_graphs_size(seed=9500):
    """Unseen sizes -- both smaller and larger than training."""
    rng = np.random.default_rng(seed)
    graphs = []

    # SMALLER (N=8-12)
    R, pos = make_grid_graph(2, 4)
    graphs.append(('grid_2x4', R, pos))

    R, pos = make_cycle_graph(8)
    graphs.append(('cycle_8', R, pos))

    R, pos = make_random_geometric_graph(10, radius=0.5, rng=rng)
    graphs.append(('rgg_10', R, pos))

    R, pos = make_random_tree(10, rng=rng)
    graphs.append(('tree_10', R, pos))

    # LARGER (N=80-150)
    R, pos = make_grid_graph(8, 10)
    graphs.append(('grid_8x10', R, pos))

    R, pos = make_grid_graph(10, 10)
    graphs.append(('grid_10x10', R, pos))

    R, pos = make_random_geometric_graph(100, radius=0.20, rng=rng)
    graphs.append(('rgg_100', R, pos))

    R, pos = make_random_geometric_graph(150, radius=0.18, rng=rng)
    graphs.append(('rgg_150', R, pos))

    R, pos = make_mesh_graph('square', n_points=100, rng=rng)
    graphs.append(('mesh_square_100', R, pos))

    R, pos = make_mesh_graph('circle', n_points=120, rng=rng)
    graphs.append(('mesh_circle_120', R, pos))

    R, pos = make_barabasi_albert_graph(100, m=3, rng=rng)
    graphs.append(('ba_100', R, pos))

    R, pos = make_watts_strogatz_graph(100, k=4, p=0.2, rng=rng)
    graphs.append(('ws_100', R, pos))

    return graphs


# ── Distribution Pairs ───────────────────────────────────────────────────────

def generate_distribution_pair(N, points, rng):
    """Generate a (source, target) pair of distributions on N nodes."""
    ic_type_src = rng.choice(['single_peak', 'multi_peak',
                              'gradient', 'smooth_random'])
    mu_src = generate_initial_condition(N, points, rng, ic_type_src)

    ic_type_tgt = rng.choice(['single_peak', 'multi_peak',
                              'gradient', 'smooth_random'])
    mu_tgt = generate_initial_condition(N, points, rng, ic_type_tgt)

    return mu_src, mu_tgt


# ── Dataset ──────────────────────────────────────────────────────────────────

class OTTransportDataset(torch.utils.data.Dataset):
    """
    Training data for OT transport across varying topologies.

    Returns 6-tuple for train_flexible_conditional:
        (mu_tau, tau, node_context, R_target, edge_index, N)
    """

    def __init__(self, graphs, n_pairs_per_graph=20,
                 n_samples=20000, seed=42):
        rng = np.random.default_rng(seed)
        all_items = []
        self.all_pairs = []

        for name, R, pos in graphs:
            N = R.shape[0]

            graph_struct = GraphStructure(R)
            cost = compute_cost_matrix(graph_struct)
            geo_cache = GeodesicCache(graph_struct)
            edge_index = rate_matrix_to_edge_index(R)

            for pair_idx in range(n_pairs_per_graph):
                mu_src, mu_tgt = generate_distribution_pair(N, pos, rng)

                # OT coupling
                coupling = compute_ot_coupling(mu_src, mu_tgt, graph_struct=graph_struct)
                geo_cache.precompute_for_coupling(coupling)

                # Node context: target distribution only
                node_ctx = mu_tgt[:, None].astype(np.float32)  # (N, 1)

                self.all_pairs.append({
                    'name': name,
                    'N': N,
                    'R': R,
                    'positions': pos,
                    'edge_index': edge_index,
                    'mu_source': mu_src,
                    'mu_target': mu_tgt,
                    'cost': cost,
                })

                # Sample flow tuples
                n_per = max(1, n_samples // (len(graphs) * n_pairs_per_graph))

                for _ in range(n_per):
                    tau = float(rng.uniform(0.0, 0.999))
                    mu_tau = marginal_distribution_fast(
                        geo_cache, coupling, tau)
                    R_target = marginal_rate_matrix_fast(
                        geo_cache, coupling, tau)
                    u_tilde = R_target * (1.0 - tau)

                    all_items.append((
                        torch.tensor(mu_tau, dtype=torch.float32),
                        torch.tensor([tau], dtype=torch.float32),
                        torch.tensor(node_ctx, dtype=torch.float32),
                        torch.tensor(u_tilde, dtype=torch.float32),
                        edge_index, N,
                    ))

        idx = rng.permutation(len(all_items))
        self.samples = [all_items[i] for i in idx[:n_samples]]
        print(f"  Dataset: {len(self.samples)} samples from "
              f"{len(self.all_pairs)} pairs on {len(graphs)} graphs",
              flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Sampling / Inference ─────────────────────────────────────────────────────

def sample_ot_transport(model, mu_start, node_ctx, edge_index,
                        n_steps=100, device=None, record_times=None):
    """ODE integration for FlexibleConditionalGNNRateMatrixPredictor.

    If record_times is provided (list of floats in [0,1]), returns dict mapping
    t -> mu_t (N,) numpy array. Otherwise returns final distribution.
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    N = len(mu_start)
    dt = 0.999 / n_steps

    ctx_tensor = torch.tensor(node_ctx, dtype=torch.float32, device=device)
    ei = edge_index.to(device)
    mu = mu_start.copy().astype(float)

    trajectory = {}
    if record_times is not None:
        # Record t=0 if requested
        for rt in record_times:
            if rt <= 0.0:
                trajectory[rt] = mu.copy()

    with torch.no_grad():
        for step in range(n_steps):
            t = step * dt
            mu_tensor = torch.tensor(mu, dtype=torch.float32, device=device)
            t_tensor = torch.tensor([t], dtype=torch.float32, device=device)

            R_pred = model.forward_single(mu_tensor, t_tensor, ctx_tensor, ei)
            # Model predicts u_tilde = (1-t)*R; recover R
            R_np = R_pred.cpu().numpy() / (1.0 - t + 1e-10)

            dp = mu @ R_np
            mu = mu + dt * dp
            mu = np.clip(mu, 0.0, None)
            s = mu.sum()
            if s > 1e-15:
                mu /= s

            # Check if current time matches any record_times
            if record_times is not None:
                t_next = (step + 1) * dt
                for rt in record_times:
                    if rt > 0.0 and abs(t_next - rt) < dt / 2 and rt not in trajectory:
                        trajectory[rt] = mu.copy()

    if record_times is not None:
        # Ensure all requested times have entries (use final mu for any missing)
        for rt in record_times:
            if rt not in trajectory:
                trajectory[rt] = mu.copy()
        return trajectory

    return mu


# ── Baselines ────────────────────────────────────────────────────────────────

EVAL_TIMES = [0.0, 0.25, 0.5, 0.75, 1.0]


def compute_exact_interpolation(mu_0, mu_1, R, t_values):
    """Compute the exact OT interpolation at specified flow times.

    Returns: dict mapping t -> mu_t_exact (N,) array
    """
    graph_struct = GraphStructure(R)
    geo_cache = GeodesicCache(graph_struct)

    coupling = compute_ot_coupling(mu_0, mu_1, graph_struct=graph_struct)
    geo_cache.precompute_for_coupling(coupling)

    interpolations = {}
    for t in t_values:
        if t <= 0.0:
            interpolations[t] = mu_0.copy()
        elif t >= 0.999:
            interpolations[t] = mu_1.copy()
        else:
            mu_t = marginal_distribution_fast(geo_cache, coupling, min(t, 0.999))
            interpolations[t] = mu_t
    return interpolations


def interpolation_baseline(mu_src, mu_tgt, alpha):
    """Linear interpolation in distribution space."""
    mu_pred = (1 - alpha) * mu_src + alpha * mu_tgt
    mu_pred = np.clip(mu_pred, 0, None)
    mu_pred /= mu_pred.sum() + 1e-15
    return mu_pred


def diffusion_baseline(mu_src, R, tau_star):
    """Diffusion toward target: heat equation."""
    P = expm(tau_star * R)
    mu_pred = mu_src @ P
    mu_pred = np.clip(mu_pred, 0, None)
    mu_pred /= mu_pred.sum() + 1e-15
    return mu_pred


def compute_exact_interpolation(mu_0, mu_1, R, t_values):
    """Compute exact OT interpolation at specified flow times.
    Returns dict mapping t -> mu_t_exact (N,) array."""
    graph_struct = GraphStructure(R)
    geo_cache = GeodesicCache(graph_struct)
    coupling = compute_ot_coupling(mu_0, mu_1, graph_struct=graph_struct)
    geo_cache.precompute_for_coupling(coupling)
    interpolations = {}
    for t in t_values:
        if t <= 0.0:
            interpolations[t] = mu_0.copy()
        else:
            mu_t = marginal_distribution_fast(geo_cache, coupling, min(t, 0.999))
            interpolations[t] = mu_t
    return interpolations


def tune_interpolation_alpha(pairs, alphas=None):
    """Tune alpha on training pairs."""
    if alphas is None:
        alphas = np.linspace(0, 1, 51)
    best_alpha = 0.5
    best_tv = np.inf
    for alpha in alphas:
        tvs = []
        for p in pairs:
            mu_pred = interpolation_baseline(p['mu_source'], p['mu_target'], alpha)
            tvs.append(total_variation(mu_pred, p['mu_target']))
        avg_tv = np.mean(tvs)
        if avg_tv < best_tv:
            best_tv = avg_tv
            best_alpha = alpha
    return best_alpha


def tune_diffusion_tau(pairs, taus=None):
    """Tune tau_star on training pairs."""
    if taus is None:
        taus = np.logspace(-3, 1, 30)
    best_tau = 0.1
    best_tv = np.inf
    for tau in taus:
        tvs = []
        for p in pairs:
            try:
                mu_pred = diffusion_baseline(p['mu_source'], p['R'], tau)
                tvs.append(total_variation(mu_pred, p['mu_target']))
            except Exception:
                tvs.append(1.0)
        avg_tv = np.mean(tvs)
        if avg_tv < best_tv:
            best_tv = avg_tv
            best_tau = tau
    return best_tau


# ── Sinkhorn OT for large graphs ────────────────────────────────────────────

def sinkhorn_coupling(mu_src, mu_tgt, cost, reg=0.1, max_iter=500):
    """Entropic OT via Sinkhorn algorithm for large graphs."""
    n = len(mu_src)
    K = np.exp(-cost / reg)
    u = np.ones(n)
    v = np.ones(n)
    a = mu_src.copy() + 1e-10
    b = mu_tgt.copy() + 1e-10
    for _ in range(max_iter):
        u = a / (K @ v + 1e-10)
        v = b / (K.T @ u + 1e-10)
    coupling = np.diag(u) @ K @ np.diag(v)
    return coupling


# ── Evaluation ───────────────────────────────────────────────────────────────

T_VALUES = [0.25, 0.5, 0.75, 1.0]


def evaluate_on_pairs(model, pairs, device, n_steps=100,
                      alpha_interp=0.5, tau_diff=0.1):
    """Evaluate FM model (and baselines) on a set of (graph, distribution) pairs.
    Returns dict of metric lists including path-level metrics."""
    results = {
        'fm_path_tv': [],      # list of dicts {t: tv_value}
        'fm_mean_path_tv': [], # list of floats (mean over T_VALUES)
        'fm_endpoint_tv': [],  # TV at t=1.0 only
        'fm_peak': [],
        'interp_endpoint_tv': [],
        'interp_peak': [],
        'diff_endpoint_tv': [],
        'diff_peak': [],
        'names': [], 'sizes': [],
        # Keep legacy keys for backward compat
        'fm_tv': [], 'interp_tv': [], 'diff_tv': [],
        'fm_wass': [], 'interp_wass': [], 'diff_wass': [],
        'interp_peak_legacy': [], 'diff_peak_legacy': [],
    }

    model.eval()

    for p in pairs:
        name = p['name']
        N = p['N']
        R = p['R']
        pos = p['positions']
        edge_index = p['edge_index']
        mu_src = p['mu_source']
        mu_tgt = p['mu_target']
        cost = p['cost']

        results['names'].append(name)
        results['sizes'].append(N)

        # Node context: target distribution
        node_ctx = mu_tgt[:, None].astype(np.float32)

        # Compute exact OT interpolation at T_VALUES
        try:
            exact_interp = compute_exact_interpolation(mu_src, mu_tgt, R, T_VALUES)
        except Exception as e:
            print(f"  Exact OT interp failed on {name}: {e}", flush=True)
            exact_interp = {}
            for t in T_VALUES:
                mu_lin = (1 - t) * mu_src + t * mu_tgt
                mu_lin /= mu_lin.sum() + 1e-15
                exact_interp[t] = mu_lin

        # FM prediction with trajectory
        try:
            fm_traj = sample_ot_transport(model, mu_src, node_ctx, edge_index,
                                          n_steps=n_steps, device=device,
                                          record_times=T_VALUES)
            path_tv = {}
            for t in T_VALUES:
                path_tv[t] = total_variation(fm_traj[t], exact_interp[t])
            results['fm_path_tv'].append(path_tv)
            results['fm_mean_path_tv'].append(float(np.mean(list(path_tv.values()))))
            results['fm_endpoint_tv'].append(path_tv[1.0])
            results['fm_tv'].append(path_tv[1.0])
            results['fm_wass'].append(float(np.sum(fm_traj[1.0][:, None] * mu_tgt[None, :] * cost)))
            results['fm_peak'].append(1.0 if np.argmax(fm_traj[1.0]) == np.argmax(mu_tgt) else 0.0)
        except Exception as e:
            print(f"  FM failed on {name}: {e}", flush=True)
            dummy_path_tv = {t: 1.0 for t in T_VALUES}
            results['fm_path_tv'].append(dummy_path_tv)
            results['fm_mean_path_tv'].append(1.0)
            results['fm_endpoint_tv'].append(1.0)
            results['fm_tv'].append(1.0)
            results['fm_wass'].append(1.0)
            results['fm_peak'].append(0.0)

        # Interpolation baseline (endpoint only)
        mu_interp = interpolation_baseline(mu_src, mu_tgt, alpha_interp)
        results['interp_endpoint_tv'].append(total_variation(mu_interp, mu_tgt))
        results['interp_tv'].append(total_variation(mu_interp, mu_tgt))
        results['interp_wass'].append(
            float(np.sum(mu_interp[:, None] * mu_tgt[None, :] * cost)))
        results['interp_peak'].append(
            1.0 if np.argmax(mu_interp) == np.argmax(mu_tgt) else 0.0)

        # Diffusion baseline (endpoint only)
        try:
            mu_diff = diffusion_baseline(mu_src, R, tau_diff)
            results['diff_endpoint_tv'].append(total_variation(mu_diff, mu_tgt))
            results['diff_tv'].append(total_variation(mu_diff, mu_tgt))
            results['diff_wass'].append(
                float(np.sum(mu_diff[:, None] * mu_tgt[None, :] * cost)))
            results['diff_peak'].append(
                1.0 if np.argmax(mu_diff) == np.argmax(mu_tgt) else 0.0)
        except Exception:
            results['diff_endpoint_tv'].append(1.0)
            results['diff_tv'].append(1.0)
            results['diff_wass'].append(1.0)
            results['diff_peak'].append(0.0)

    return results


def prepare_eval_pairs(graphs, n_pairs=5, seed=7777):
    """Generate evaluation distribution pairs for a set of graphs."""
    rng = np.random.default_rng(seed)
    pairs = []
    for name, R, pos in graphs:
        N = R.shape[0]
        graph_struct = GraphStructure(R)
        cost = compute_cost_matrix(graph_struct)
        edge_index = rate_matrix_to_edge_index(R)
        for _ in range(n_pairs):
            mu_src, mu_tgt = generate_distribution_pair(N, pos, rng)
            pairs.append({
                'name': name,
                'N': N,
                'R': R,
                'positions': pos,
                'edge_index': edge_index,
                'mu_source': mu_src,
                'mu_target': mu_tgt,
                'cost': cost,
            })
    return pairs


# ── Plotting ─────────────────────────────────────────────────────────────────

def draw_graph_with_dist(ax, R, pos, mu=None, title='', cmap='hot_r',
                         node_size=None, vmin=None, vmax=None,
                         edge_alpha=0.15, edge_lw=0.4):
    """Draw a graph with optional distribution coloring.

    Returns the scatter handle (for colorbars) or None.
    """
    N = R.shape[0]
    if node_size is None:
        node_size = max(30, min(80, 3000 / max(N, 1)))

    # Draw edges
    for i in range(N):
        for j in range(i + 1, N):
            if abs(R[i, j]) > 1e-10 and i != j:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color='lightgray', alpha=edge_alpha, linewidth=edge_lw,
                        zorder=1)
    # Draw nodes
    sc = None
    if mu is not None:
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=mu, cmap=cmap,
                        s=node_size, zorder=5, edgecolors='gray',
                        linewidths=0.3, vmin=vmin, vmax=vmax)
    else:
        ax.scatter(pos[:, 0], pos[:, 1], c='steelblue', s=node_size,
                   zorder=5, edgecolors='gray', linewidths=0.3)
    ax.set_title(title, fontsize=8)
    ax.set_aspect('equal')
    ax.axis('off')
    return sc


def plot_results(losses, train_graphs, id_results, topo_results,
                 size_results, topo_pairs, out_path, device, model):
    """Create the 2x3 results figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel A: Training loss
    ax = axes[0, 0]
    ax.plot(losses, 'b-', alpha=0.7, linewidth=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(A) Training Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Panel B: Graph zoo (subset)
    ax = axes[0, 1]
    n_show = min(12, len(train_graphs))
    ncols = 4
    nrows = (n_show + ncols - 1) // ncols
    for idx in range(n_show):
        name, R, pos = train_graphs[idx]
        N = R.shape[0]
        # Shift positions for subplot layout
        offset_x = (idx % ncols) * 2.5
        offset_y = (idx // ncols) * 2.5
        pos_s = pos.copy()
        # Normalize to [0,1]
        pos_s = pos_s - pos_s.min(0)
        rng_pos = pos_s.max(0) - pos_s.min(0) + 1e-8
        pos_s = pos_s / rng_pos.max()
        pos_s[:, 0] += offset_x
        pos_s[:, 1] += offset_y
        for i in range(N):
            for j in range(i + 1, N):
                if abs(R[i, j]) > 1e-10:
                    ax.plot([pos_s[i, 0], pos_s[j, 0]],
                            [pos_s[i, 1], pos_s[j, 1]],
                            'b-', alpha=0.2, linewidth=0.3)
        ax.scatter(pos_s[:, 0], pos_s[:, 1], s=5, c='steelblue', zorder=5)
    ax.set_title('(B) Graph Zoo (training subset)')
    ax.axis('off')

    # Panel C: Mean path TV bars (FM uses mean_path_tv, others use endpoint_tv)
    ax = axes[0, 2]
    methods = ['FM (path)', 'Interp', 'Diffusion']
    splits = ['ID', 'OOD-topo', 'OOD-size']
    all_results = [id_results, topo_results, size_results]
    x = np.arange(len(splits))
    width = 0.25
    for i, (meth, key) in enumerate(zip(methods,
                                        ['fm_mean_path_tv',
                                         'interp_endpoint_tv', 'diff_endpoint_tv'])):
        vals = [np.nanmean(r[key]) for r in all_results]
        ax.bar(x + i * width, vals, width, label=meth, alpha=0.8)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(splits)
    ax.set_ylabel('TV Distance')
    ax.set_title('(C) Mean Path TV / Endpoint TV by Split')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel D: TV vs flow time (path quality over flow time)
    # Compute path baselines: linear interp toward true target
    ax = axes[1, 0]
    t_plot = [0.0, 0.25, 0.5, 0.75, 1.0]

    fm_means, fm_stds = [], []
    lin_true_means, lin_true_stds = [], []

    for t in t_plot:
        if t == 0.0:
            fm_means.append(0.0); fm_stds.append(0.0)
            lin_true_means.append(0.0); lin_true_stds.append(0.0)
        else:
            # FM path TV (already computed)
            tvs = [d[t] for d in topo_results['fm_path_tv']]
            fm_means.append(np.mean(tvs)); fm_stds.append(np.std(tvs))

            # Linear baseline: compute on-the-fly from topo_pairs
            lt_tvs = []
            for p in topo_pairs:
                mu_src = p['mu_source']
                mu_tgt = p['mu_target']
                R = p['R']
                # Exact OT interpolation at this t
                try:
                    exact_t = compute_exact_interpolation(
                        mu_src, mu_tgt, R, [t])[t]
                except Exception:
                    exact_t = (1 - t) * mu_src + t * mu_tgt
                    exact_t /= exact_t.sum() + 1e-15

                # Linear toward true target
                mu_lin_true = (1 - t) * mu_src + t * mu_tgt
                mu_lin_true = np.clip(mu_lin_true, 0, None)
                mu_lin_true /= mu_lin_true.sum() + 1e-15
                lt_tvs.append(total_variation(mu_lin_true, exact_t))

            lin_true_means.append(np.mean(lt_tvs))
            lin_true_stds.append(np.std(lt_tvs))

    fm_means = np.array(fm_means); fm_stds = np.array(fm_stds)
    lt_m = np.array(lin_true_means); lt_s = np.array(lin_true_stds)

    ax.plot(t_plot, fm_means, 'o-', color='tab:blue', label='FM (learned path)', lw=1.5)
    ax.fill_between(t_plot, fm_means - fm_stds, fm_means + fm_stds,
                    alpha=0.15, color='tab:blue')
    ax.plot(t_plot, lt_m, 's--', color='tab:green', label='Linear (true target)', lw=1.2)
    ax.fill_between(t_plot, lt_m - lt_s, lt_m + lt_s, alpha=0.1, color='tab:green')

    ax.set_xlabel('Flow Time t')
    ax.set_ylabel('TV from Exact OT Interpolation')
    ax.set_title('(D) Path Quality Over Flow Time')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # Panel E: Mean path TV vs graph size (FM), endpoint TV for others
    ax = axes[1, 1]
    all_sizes = (size_results['sizes'] + topo_results['sizes']
                 + id_results['sizes'])
    all_fm_path_tv = (size_results['fm_mean_path_tv'] + topo_results['fm_mean_path_tv']
                      + id_results['fm_mean_path_tv'])
    all_interp_tv = (size_results['interp_endpoint_tv'] + topo_results['interp_endpoint_tv']
                     + id_results['interp_endpoint_tv'])
    ax.scatter(all_sizes, all_fm_path_tv, alpha=0.5, s=15,
               label='FM (mean path TV)', c='tab:blue')
    ax.scatter(all_sizes, all_interp_tv, alpha=0.5, s=15, label='Interp (endpoint)',
               c='tab:green', marker='s')
    ax.set_xlabel('Graph Size (N)')
    ax.set_ylabel('TV Distance')
    ax.set_title('(E) Path/Endpoint TV vs Graph Size')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel F: TV by graph family
    ax = axes[1, 2]
    all_names = (id_results['names'] + topo_results['names']
                 + size_results['names'])
    all_fm = (id_results['fm_tv'] + topo_results['fm_tv']
              + size_results['fm_tv'])
    families = {}
    for nm, tv in zip(all_names, all_fm):
        fam = nm.split('_')[0]
        if fam not in families:
            families[fam] = []
        families[fam].append(tv)
    fam_names = sorted(families.keys())
    fam_means = [np.mean(families[f]) for f in fam_names]
    fam_stds = [np.std(families[f]) for f in fam_names]
    x_fam = np.arange(len(fam_names))
    ax.bar(x_fam, fam_means, yerr=fam_stds, capsize=3, alpha=0.7,
           color='steelblue')
    ax.set_xticks(x_fam)
    ax.set_xticklabels(fam_names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('TV Distance')
    ax.set_title('(F) TV by Graph Family')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def plot_transport_gallery(model, pairs, out_path, device, n_show=4):
    """Gallery: n_show graphs x 2 rows (Exact OT / FM Learned) x 5 cols.

    Features:
    - Larger node markers scaled by graph size
    - Faint edges showing graph structure
    - Row labels, column headers, separator lines, graph name labels
    - Per-group colorbar
    """
    import matplotlib.gridspec as gridspec

    t_cols = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_show = min(n_show, len(pairs))
    rows_per_graph = 2
    n_rows = n_show * rows_per_graph
    n_cols = len(t_cols)

    fig = plt.figure(figsize=(3.0 * n_cols + 0.8, 2.6 * n_rows))
    # Extra column for colorbar
    gs = gridspec.GridSpec(n_rows, n_cols + 1,
                           width_ratios=[1] * n_cols + [0.05],
                           hspace=0.25, wspace=0.12)

    row_labels = ['Exact OT', 'FM (Learned)']

    for g_idx in range(n_show):
        p = pairs[g_idx]
        R = p['R']
        pos = p['positions']
        N = R.shape[0]
        mu_src = p['mu_source']
        mu_tgt = p['mu_target']
        node_ctx = mu_tgt[:, None].astype(np.float32)

        # Compute exact OT interpolation
        try:
            exact = compute_exact_interpolation(mu_src, mu_tgt, R, t_cols)
        except Exception:
            exact = {t: (1 - t) * mu_src + t * mu_tgt for t in t_cols}
            for t in t_cols:
                exact[t] /= exact[t].sum() + 1e-15

        # Compute FM trajectory
        try:
            fm_traj = sample_ot_transport(model, mu_src, node_ctx,
                                          p['edge_index'], n_steps=100,
                                          device=device, record_times=t_cols)
        except Exception:
            fm_traj = {t: mu_src.copy() for t in t_cols}

        base_row = g_idx * rows_per_graph

        # Shared vmin/vmax across all panels for this graph
        all_mus = ([exact.get(t, mu_src) for t in t_cols]
                   + [fm_traj.get(t, mu_src) for t in t_cols])
        vmin = 0.0
        vmax = max(m.max() for m in all_mus)

        ns = max(30, min(80, 3000 / max(N, 1)))
        last_sc = None  # for colorbar

        for col_idx, t in enumerate(t_cols):
            # Row 1: Exact OT
            ax = fig.add_subplot(gs[base_row, col_idx])
            mu_exact = exact.get(t, mu_src)
            last_sc = draw_graph_with_dist(
                ax, R, pos, mu=mu_exact, cmap='hot_r',
                node_size=ns, vmin=vmin, vmax=vmax,
                title=f't={t:.2f}' if g_idx == 0 else '')
            if col_idx == 0:
                ax.set_ylabel(row_labels[0], fontsize=8, fontweight='bold')

            # Row 2: FM Learned
            ax = fig.add_subplot(gs[base_row + 1, col_idx])
            mu_fm = fm_traj.get(t, mu_src)
            draw_graph_with_dist(
                ax, R, pos, mu=mu_fm, cmap='hot_r',
                node_size=ns, vmin=vmin, vmax=vmax)
            if col_idx == 0:
                ax.set_ylabel(row_labels[1], fontsize=8, fontweight='bold')

        # Colorbar for this graph group
        if last_sc is not None:
            cbar_ax = fig.add_subplot(gs[base_row:base_row + rows_per_graph, n_cols])
            fig.colorbar(last_sc, cax=cbar_ax)

        # Graph name label (left margin)
        fig.text(0.01, 1.0 - (base_row + 1.0) / n_rows,
                 p['name'], fontsize=9, fontweight='bold',
                 rotation=90, va='center', ha='left')

        # Separator line between graph groups (except after last)
        if g_idx < n_show - 1:
            sep_y = 1.0 - (base_row + rows_per_graph) / n_rows
            fig.add_artist(plt.Line2D(
                [0.05, 0.95], [sep_y, sep_y],
                transform=fig.transFigure, color='gray',
                linewidth=0.5, alpha=0.5))

    fig.suptitle('Transport Gallery: Exact OT / FM Learned',
                 fontsize=11, y=0.995)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


# ── Console Output ───────────────────────────────────────────────────────────

def print_results_table(id_res, topo_res, size_res):
    """Print formatted results table with path-level and endpoint metrics."""
    # Primary metric: mean path TV for FM, endpoint TV for others
    print("\nPath / Endpoint TV Results:", flush=True)
    header = (f"{'':22s} {'ID':>10s}  {'OOD-topo':>10s}  "
              f"{'OOD-small':>10s}  {'OOD-large':>10s}")
    print(header, flush=True)

    for meth, key in [('FM (mean path TV)', 'fm_mean_path_tv'),
                      ('FM (endpoint TV)', 'fm_endpoint_tv'),
                      ('Interpolation', 'interp_endpoint_tv'),
                      ('Diffusion', 'diff_endpoint_tv')]:
        id_v = np.nanmean(id_res[key])
        topo_v = np.nanmean(topo_res[key])
        sm_vals = [size_res[key][i] for i, sz in enumerate(size_res['sizes']) if sz <= 20]
        lg_vals = [size_res[key][i] for i, sz in enumerate(size_res['sizes']) if sz > 20]
        sm_v = np.nanmean(sm_vals) if sm_vals else float('nan')
        lg_v = np.nanmean(lg_vals) if lg_vals else float('nan')
        print(f"  {meth:20s} {id_v:10.4f}  {topo_v:10.4f}  {sm_v:10.4f}  {lg_v:10.4f}",
              flush=True)

    # FM path TV at each time step
    print("\nFM Path TV by flow time (OOD-topo):", flush=True)
    for t in T_VALUES:
        tvs = [d[t] for d in topo_res['fm_path_tv']]
        print(f"  t={t:.2f}: mean={np.mean(tvs):.4f}, std={np.std(tvs):.4f}", flush=True)

    # TV by size bucket
    all_sizes = id_res['sizes'] + topo_res['sizes'] + size_res['sizes']
    all_fm = id_res['fm_mean_path_tv'] + topo_res['fm_mean_path_tv'] + size_res['fm_mean_path_tv']

    buckets = [(8, 12), (15, 30), (31, 60), (80, 100), (101, 200)]
    print("\nPath TV by graph size:", flush=True)
    for lo, hi in buckets:
        fm_vals = [v for s, v in zip(all_sizes, all_fm) if lo <= s <= hi]
        if fm_vals:
            fm_m = np.nanmean(fm_vals)
            print(f"  N={lo}-{hi}: FM(path)={fm_m:.2f}",
                  flush=True)

    # Peak recovery
    fm_peak = np.nanmean(id_res['fm_peak'] + topo_res['fm_peak']
                         + size_res['fm_peak'])
    interp_peak = np.nanmean(id_res['interp_peak'] + topo_res['interp_peak']
                             + size_res['interp_peak'])
    print(f"\nPeak recovery:", flush=True)
    print(f"  FM: {fm_peak*100:.0f}%, "
          f"Interpolation: {interp_peak*100:.0f}%", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ex17: OT Transport Generalization')
    parser.add_argument('--n-pairs-per-graph', type=int, default=20)
    parser.add_argument('--n-samples', type=int, default=20000)
    parser.add_argument('--n-epochs', type=int, default=1000)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--loss-type', type=str, default='rate_kl')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint-every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--n-eval-pairs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()

    device = args.device or get_device()
    print(f"Device: {device}", flush=True)

    # ── Generate graphs ──
    print("\n=== Experiment 17: OT Transport Generalization ===", flush=True)

    print("\nGenerating training graphs...", flush=True)
    train_graphs = generate_training_graphs(seed=42)
    print(f"  {len(train_graphs)} training graphs, sizes "
          f"{min(g[1].shape[0] for g in train_graphs)}-"
          f"{max(g[1].shape[0] for g in train_graphs)}", flush=True)

    print("Generating test graphs (topology)...", flush=True)
    test_topo = generate_test_graphs_topology(seed=9000)
    print(f"  {len(test_topo)} OOD-topo graphs", flush=True)

    print("Generating test graphs (size)...", flush=True)
    test_size = generate_test_graphs_size(seed=9500)
    print(f"  {len(test_size)} OOD-size graphs", flush=True)

    # ── Build dataset ──
    print("\nBuilding OT transport dataset...", flush=True)
    dataset = OTTransportDataset(
        train_graphs,
        n_pairs_per_graph=args.n_pairs_per_graph,
        n_samples=args.n_samples,
        seed=42,
    )
    n_train_pairs = len(dataset.all_pairs)
    print(f"Training: {len(train_graphs)} graphs, {n_train_pairs} distribution pairs",
          flush=True)
    print(f"Testing: ID ({len(train_graphs)} graphs), "
          f"OOD-topo ({len(test_topo)} graphs), "
          f"OOD-size ({len(test_size)} graphs)", flush=True)

    # ── Checkpointing ──
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'ex17_fm_model.pt')

    # ── FM Model ──
    print("\n--- Training FM model ---", flush=True)
    model = FlexibleConditionalGNNRateMatrixPredictor(
        context_dim=1,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    )

    if args.resume and os.path.exists(ckpt_path):
        print(f"  Resuming from {ckpt_path}", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        losses = []
    else:
        result = train_flexible_conditional(
            model, dataset,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            loss_weighting='uniform',
            loss_type=args.loss_type,
            ema_decay=0.999,
        )
        losses = result['losses']
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved FM model to {ckpt_path}", flush=True)

    # ── Tune baselines ──
    print("\nTuning baselines...", flush=True)
    # Use a subset of training pairs for tuning
    tune_pairs = dataset.all_pairs[:min(50, len(dataset.all_pairs))]
    alpha_interp = tune_interpolation_alpha(tune_pairs)
    tau_diff = tune_diffusion_tau(tune_pairs)
    print(f"  Interpolation alpha={alpha_interp:.3f}", flush=True)
    print(f"  Diffusion tau_star={tau_diff:.4f}", flush=True)

    # ── Evaluate ──
    print("\nPreparing evaluation pairs...", flush=True)
    id_pairs = prepare_eval_pairs(train_graphs, n_pairs=args.n_eval_pairs,
                                  seed=7777)
    topo_pairs = prepare_eval_pairs(test_topo, n_pairs=args.n_eval_pairs,
                                    seed=8888)
    size_pairs = prepare_eval_pairs(test_size, n_pairs=args.n_eval_pairs,
                                    seed=9999)
    print(f"  ID: {len(id_pairs)}, OOD-topo: {len(topo_pairs)}, "
          f"OOD-size: {len(size_pairs)}", flush=True)

    model.to(device)

    print("\nEvaluating ID...", flush=True)
    id_results = evaluate_on_pairs(model, id_pairs, device,
                                   alpha_interp=alpha_interp,
                                   tau_diff=tau_diff)

    print("Evaluating OOD-topo...", flush=True)
    topo_results = evaluate_on_pairs(model, topo_pairs, device,
                                     alpha_interp=alpha_interp,
                                     tau_diff=tau_diff)

    print("Evaluating OOD-size...", flush=True)
    size_results = evaluate_on_pairs(model, size_pairs, device,
                                     alpha_interp=alpha_interp,
                                     tau_diff=tau_diff)

    # ── Print results ──
    print_results_table(id_results, topo_results, size_results)

    # ── Plots ──
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("\nGenerating plots...", flush=True)
    plot_results(
        losses, train_graphs,
        id_results, topo_results, size_results,
        topo_pairs, os.path.join(out_dir, 'ex17_results.png'),
        device, model,
    )

    # Transport gallery: pick 3 diverse OOD graphs
    gallery_pairs = []
    seen_names = set()
    for p in topo_pairs + size_pairs:
        if p['name'] not in seen_names and len(gallery_pairs) < 4:
            gallery_pairs.append(p)
            seen_names.add(p['name'])
    plot_transport_gallery(
        model, gallery_pairs,
        os.path.join(out_dir, 'ex17_transport_gallery.png'),
        device, n_show=4,
    )

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
