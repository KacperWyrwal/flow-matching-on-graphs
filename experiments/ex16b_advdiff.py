"""
Experiment 16b: Advection-Diffusion Flow Matching on Mesh Geometries.

Trains a FiLM-conditional GNN flow matching model on advection-diffusion
simulations across 8 mesh geometries and tests topology generalization on
4 unseen geometries.  Velocity field type is part of the conditioning.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import expm

from graph_ot_fm import (
    GraphStructure,
    GeodesicCache,
)
from graph_ot_fm.ot_solver import compute_ot_coupling
from graph_ot_fm.flow import marginal_distribution_fast, marginal_rate_matrix_fast
from graph_ot_fm import total_variation

from meta_fm import (
    FiLMConditionalGNNRateMatrixPredictor,
    DirectGNNPredictor,
    train_film_conditional,
    train_direct_gnn,
    get_device,
)
from meta_fm.model import rate_matrix_to_edge_index, RateMessagePassing
from meta_fm.sample import sample_posterior_film

from ex16_heat_mesh import (
    generate_mesh, mesh_to_graph, generate_initial_condition,
    TRAIN_GEOS, TEST_GEOS, IC_TYPES, _is_connected,
)


# ── Constants ─────────────────────────────────────────────────────────────────

VELOCITY_FIELDS = ['uniform', 'vortex', 'source', 'sink', 'shear']


# ── Velocity Fields ───────────────────────────────────────────────────────────

def make_velocity_field(field_type, params=None):
    """
    Return a callable v(x) -> (2,) 2D velocity vector.

    field_type: 'uniform', 'vortex', 'source', 'sink', 'shear'
    params: dict of field parameters (optional; defaults provided)
    """
    if params is None:
        params = {}

    if field_type == 'uniform':
        theta = params.get('theta', 0.0)
        return lambda x: np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

    elif field_type == 'vortex':
        strength = params.get('strength', 1.0)
        cx = params.get('cx', 0.5)
        cy = params.get('cy', 0.5)
        def _vortex(x):
            dx = x[0] - cx
            dy = x[1] - cy
            return np.array([-strength * dy, strength * dx], dtype=np.float32)
        return _vortex

    elif field_type == 'source':
        strength = params.get('strength', 1.0)
        cx = params.get('cx', 0.5)
        cy = params.get('cy', 0.5)
        def _source(x):
            dx = x[0] - cx
            dy = x[1] - cy
            r = np.sqrt(dx**2 + dy**2) + 1e-10
            return np.array([strength * dx / r, strength * dy / r], dtype=np.float32)
        return _source

    elif field_type == 'sink':
        strength = params.get('strength', 1.0)
        cx = params.get('cx', 0.5)
        cy = params.get('cy', 0.5)
        def _sink(x):
            dx = x[0] - cx
            dy = x[1] - cy
            r = np.sqrt(dx**2 + dy**2) + 1e-10
            return np.array([-strength * dx / r, -strength * dy / r], dtype=np.float32)
        return _sink

    elif field_type == 'shear':
        strength = params.get('strength', 1.0)
        cy = params.get('cy', 0.5)
        def _shear(x):
            return np.array([strength * (x[1] - cy), 0.0], dtype=np.float32)
        return _shear

    else:
        raise ValueError(f"Unknown velocity field type: {field_type}")


def sample_velocity_params(field_type, rng):
    """Sample random parameters for a velocity field type."""
    if field_type == 'uniform':
        return {'theta': float(rng.uniform(0, 2 * np.pi))}
    elif field_type in ('vortex', 'source', 'sink'):
        return {
            'strength': float(rng.uniform(0.5, 2.0)),
            'cx': float(rng.uniform(0.3, 0.7)),
            'cy': float(rng.uniform(0.3, 0.7)),
        }
    elif field_type == 'shear':
        return {
            'strength': float(rng.uniform(0.5, 2.0)),
            'cy': float(rng.uniform(0.3, 0.7)),
        }
    else:
        return {}


# ── Advection-Diffusion Rate Matrix ───────────────────────────────────────────

def build_advection_diffusion_rate_matrix(points, triangles, v_field, D=1.0, alpha=1.0):
    """
    Build advection-diffusion rate matrix on a triangulated mesh.

    For each edge (a, b):
        R_ab = D / dist(a,b) + alpha * max(0, v(x_a) · (x_b - x_a) / dist) / dist

    Diagonal entries: -row_sum (so rows sum to zero).
    """
    N = len(points)
    R = np.zeros((N, N))

    for tri in triangles:
        for i in range(3):
            for j in range(3):
                if i != j:
                    a, b = tri[i], tri[j]
                    diff = points[b] - points[a]
                    dist = np.linalg.norm(diff) + 1e-12
                    direction = diff / dist

                    # Diffusion term
                    diff_term = D / dist

                    # Advection term: upwind scheme
                    va = v_field(points[a])
                    adv_proj = np.dot(va, direction)
                    adv_term = alpha * max(0.0, adv_proj) / dist

                    R[a, b] = max(R[a, b], diff_term + adv_term)

    # Set diagonal to negative row sum
    np.fill_diagonal(R, 0.0)
    np.fill_diagonal(R, -R.sum(axis=1))
    return R


# ── Exact Simulation ──────────────────────────────────────────────────────────

def simulate_exact(mu_init, R, T):
    """
    Simulate advection-diffusion using matrix exponential.

    mu_final = mu_init @ expm(T * R)
    """
    P = expm(T * R)
    mu = mu_init @ P
    mu = np.clip(mu, 0, None)
    mu /= mu.sum() + 1e-15
    return mu.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class AdvectionDiffusionDataset(torch.utils.data.Dataset):
    """
    Dataset for advection-diffusion flow matching on mesh geometries.

    Generates (mu_tau, tau, node_ctx, global_ctx, u_tilde, edge_index, N) tuples.
    node_ctx  = [mu_init, boundary, v_x_norm, v_y_norm]  (N, 4)
    global_ctx = [T, D, alpha]  (3,)
    """

    def __init__(self, geometries, n_meshes_per_geo=10, n_fields_per_mesh=3,
                 n_traj_per_field=3, T_range=(0.05, 0.2), n_samples=20000,
                 n_points=60, D=1.0, alpha_range=(0.5, 2.0), seed=42):
        self.samples = []
        self.all_pairs = []

        rng = np.random.default_rng(seed)
        all_items = []

        for geo in geometries:
            for mesh_idx in range(n_meshes_per_geo):
                mesh_seed = int(rng.integers(0, 100000))
                try:
                    points, triangles, boundary_mask = generate_mesh(geo, n_points, seed=mesh_seed)
                except Exception as e:
                    print(f"  Mesh gen failed for {geo} mesh {mesh_idx}: {e}", flush=True)
                    continue

                N = len(points)
                bnd = boundary_mask.astype(np.float32)

                for field_idx in range(n_fields_per_mesh):
                    field_type = VELOCITY_FIELDS[int(rng.integers(len(VELOCITY_FIELDS)))]
                    field_params = sample_velocity_params(field_type, rng)
                    v_field = make_velocity_field(field_type, field_params)

                    alpha = float(rng.uniform(alpha_range[0], alpha_range[1]))

                    try:
                        R = build_advection_diffusion_rate_matrix(
                            points, triangles, v_field, D=D, alpha=alpha)
                    except Exception as e:
                        print(f"  R build failed for {geo} field {field_idx}: {e}", flush=True)
                        continue

                    if not _is_connected(R):
                        print(f"  Skipping disconnected graph: {geo} mesh {mesh_idx} field {field_idx}", flush=True)
                        continue

                    try:
                        graph_struct = GraphStructure(R)
                        geo_cache = GeodesicCache(graph_struct)
                        edge_index = rate_matrix_to_edge_index(R)
                    except Exception as e:
                        print(f"  Graph struct failed for {geo} mesh {mesh_idx}: {e}", flush=True)
                        continue

                    # Compute normalized node velocity features
                    v_nodes = np.array([v_field(points[a]) for a in range(N)],
                                       dtype=np.float32)  # (N, 2)
                    v_max = np.abs(v_nodes).max() + 1e-10
                    v_nodes_norm = v_nodes / v_max

                    for traj_idx in range(n_traj_per_field):
                        ic_type = IC_TYPES[int(rng.integers(len(IC_TYPES)))]
                        mu_init = generate_initial_condition(N, points, rng, ic_type)
                        T_val = float(rng.uniform(T_range[0], T_range[1]))

                        try:
                            mu_final = simulate_exact(mu_init, R, T_val)
                            coupling = compute_ot_coupling(mu_init, mu_final, graph_struct=graph_struct)
                            geo_cache.precompute_for_coupling(coupling)
                        except Exception as e:
                            print(f"  OT failed for {geo} traj {traj_idx}: {e}", flush=True)
                            continue

                        # Store eval pair (without geo_cache/coupling to save memory)
                        self.all_pairs.append({
                            'geo': geo,
                            'field_type': field_type,
                            'N': N,
                            'R': R,
                            'points': points,
                            'triangles': triangles,
                            'boundary': bnd,
                            'edge_index': edge_index,
                            'mu_source': mu_init,
                            'mu_target': mu_final,
                            'v_nodes': v_nodes_norm,   # (N, 2)
                            'T': T_val,
                            'D': D,
                            'alpha': alpha,
                        })

                        # Sample training tuples from this trajectory
                        n_total_traj = (len(geometries) * n_meshes_per_geo *
                                        n_fields_per_mesh * n_traj_per_field)
                        n_per = max(1, n_samples // n_total_traj)

                        for _ in range(n_per):
                            tau = float(rng.uniform(0.0, 0.999))
                            try:
                                mu_tau = marginal_distribution_fast(geo_cache, coupling, tau)
                                R_target = marginal_rate_matrix_fast(geo_cache, coupling, tau)
                                u_tilde = R_target * (1.0 - tau)
                            except Exception:
                                continue

                            # node_ctx: (N, 4) = [mu_init, boundary, v_x_norm, v_y_norm]
                            node_ctx = np.stack(
                                [mu_init, bnd, v_nodes_norm[:, 0], v_nodes_norm[:, 1]],
                                axis=1).astype(np.float32)
                            # global_ctx: (3,) = [T, D, alpha]
                            global_ctx = np.array([T_val, D, alpha], dtype=np.float32)

                            all_items.append((
                                torch.tensor(mu_tau,    dtype=torch.float32),
                                torch.tensor([tau],     dtype=torch.float32),
                                torch.tensor(node_ctx,  dtype=torch.float32),
                                torch.tensor(global_ctx, dtype=torch.float32),
                                torch.tensor(u_tilde,   dtype=torch.float32),
                                edge_index,
                                N,
                            ))

        # Shuffle and trim
        idx = np.random.default_rng(seed + 1).permutation(len(all_items))
        n_actual = min(n_samples, len(all_items))
        self.samples = [all_items[i] for i in idx[:n_actual]]
        print(f"  Dataset: {len(self.samples)} training samples, "
              f"{len(self.all_pairs)} eval pairs", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Baselines ─────────────────────────────────────────────────────────────────

class AdvDiffMGN(nn.Module):
    """
    MeshGraphNet-style baseline for advection-diffusion.

    Input per node: [mu, boundary, T_broadcast, v_x, v_y] = 5 dims
    Output: softmax distribution
    """

    def __init__(self, hidden_dim=64, n_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(5, hidden_dim)
        self.mp_layers = nn.ModuleList([
            RateMessagePassing(in_dim=hidden_dim, hidden_dim=hidden_dim)
            for _ in range(n_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feats, edge_index):
        """
        node_feats: (N, 5)
        edge_index: (2, E)
        Returns: (N,) softmax distribution
        """
        h = self.input_proj(node_feats)
        for mp in self.mp_layers:
            h = mp(h, edge_index)
        logits = self.readout(h).squeeze(-1)
        return torch.softmax(logits, dim=0)

    def predict(self, mu_t, T_val, boundary_t, v_nodes_t, edge_index):
        """
        mu_t:       (N,) tensor
        T_val:      float
        boundary_t: (N,) tensor
        v_nodes_t:  (N, 2) tensor
        edge_index: (2, E) tensor
        Returns:    (N,) numpy array
        """
        self.eval()
        with torch.no_grad():
            T_broadcast = torch.full_like(mu_t, T_val)
            node_feats = torch.stack(
                [mu_t, boundary_t, T_broadcast,
                 v_nodes_t[:, 0], v_nodes_t[:, 1]], dim=1)
            pred = self.forward(node_feats, edge_index)
        return pred.cpu().numpy()


def train_mgn_advdiff(mgn, all_pairs, n_epochs=300, lr=5e-4, device=None):
    """Train AdvDiffMGN baseline on all_pairs."""
    if device is None:
        device = get_device()
    mgn = mgn.to(device)
    optimizer = torch.optim.Adam(mgn.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        idx = np.random.permutation(len(all_pairs))
        for i in idx:
            pair = all_pairs[i]
            mu_src = torch.tensor(pair['mu_source'], dtype=torch.float32, device=device)
            mu_tgt = torch.tensor(pair['mu_target'], dtype=torch.float32, device=device)
            bnd    = torch.tensor(pair['boundary'],  dtype=torch.float32, device=device)
            v_nodes_t = torch.tensor(pair['v_nodes'], dtype=torch.float32, device=device)
            ei     = pair['edge_index'].to(device)
            T_val  = pair['T']

            T_broadcast = torch.full_like(mu_src, T_val)
            node_feats = torch.stack(
                [mu_src, bnd, T_broadcast,
                 v_nodes_t[:, 0], v_nodes_t[:, 1]], dim=1)
            mu_pred = mgn(node_feats, ei)

            loss = F.kl_div(torch.log(mu_pred + 1e-8), mu_tgt, reduction='sum')

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mgn.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(all_pairs), 1)
        losses.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            print(f"  MGN Epoch {epoch+1}/{n_epochs} | KL Loss: {avg_loss:.6f}", flush=True)

    return losses


def build_direct_gnn_pairs_advdiff(all_pairs):
    """Convert all_pairs to list of (context_np, mu_target_np, edge_index) tuples.

    context per node: [mu_src, boundary, v_x, v_y, T_broadcast]  (N, 5)
    """
    pairs = []
    for p in all_pairs:
        N = p['N']
        mu_src = p['mu_source']
        bnd    = p['boundary']
        v_nodes = p['v_nodes']   # (N, 2)
        T_val   = p['T']
        T_broadcast = np.full(N, T_val, dtype=np.float32)
        context = np.stack(
            [mu_src, bnd, v_nodes[:, 0], v_nodes[:, 1], T_broadcast],
            axis=1)  # (N, 5)
        pairs.append((context, p['mu_target'], p['edge_index']))
    return pairs


# ── Euler Baseline ────────────────────────────────────────────────────────────

def baseline_euler(mu, R, T):
    """First-order Euler approximation: mu @ (I + T*R), clipped & normalized."""
    N = len(mu)
    mu_next = mu @ (np.eye(N) + T * R)
    mu_next = np.clip(mu_next, 0, None)
    return mu_next / (mu_next.sum() + 1e-15)


# ── Test Case Generation ──────────────────────────────────────────────────────

def generate_test_cases_advdiff(geometries, n_meshes=5, n_cases_per_mesh=5,
                                T_range=(0.05, 0.2), n_points=60,
                                D=1.0, alpha_range=(0.5, 2.0), seed=9000):
    """
    Generate test cases for advection-diffusion evaluation.

    Returns list of dicts with keys:
        geo, field_type, N, points, triangles, boundary, R, edge_index,
        mu_source, mu_target, v_nodes (N,2 normalized), T, D, alpha, ic_type
    """
    rng = np.random.default_rng(seed)
    test_cases = []

    for geo_idx, geo in enumerate(geometries):
        for mesh_idx in range(n_meshes):
            mesh_seed = seed + (geo_idx * n_meshes + mesh_idx) * 1000
            try:
                points, triangles, boundary_mask = generate_mesh(geo, n_points, seed=mesh_seed)
            except Exception as e:
                print(f"  Test mesh gen failed for {geo} mesh {mesh_idx}: {e}", flush=True)
                continue

            N = len(points)
            bnd = boundary_mask.astype(np.float32)

            for case_idx in range(n_cases_per_mesh):
                field_type = VELOCITY_FIELDS[int(rng.integers(len(VELOCITY_FIELDS)))]
                field_params = sample_velocity_params(field_type, rng)
                v_field = make_velocity_field(field_type, field_params)

                alpha = float(rng.uniform(alpha_range[0], alpha_range[1]))

                try:
                    R = build_advection_diffusion_rate_matrix(
                        points, triangles, v_field, D=D, alpha=alpha)
                except Exception as e:
                    print(f"  Test R build failed: {e}", flush=True)
                    continue

                if not _is_connected(R):
                    continue

                edge_index = rate_matrix_to_edge_index(R)

                # Node velocity features
                v_nodes = np.array([v_field(points[a]) for a in range(N)],
                                   dtype=np.float32)
                v_max = np.abs(v_nodes).max() + 1e-10
                v_nodes_norm = v_nodes / v_max

                ic_type = IC_TYPES[int(rng.integers(len(IC_TYPES)))]
                mu_init = generate_initial_condition(N, points, rng, ic_type)
                T_val = float(rng.uniform(T_range[0], T_range[1]))

                try:
                    mu_final = simulate_exact(mu_init, R, T_val)
                except Exception:
                    continue

                test_cases.append({
                    'geo': geo,
                    'field_type': field_type,
                    'N': N,
                    'points': points,
                    'triangles': triangles,
                    'boundary': bnd,
                    'R': R,
                    'edge_index': edge_index,
                    'mu_source': mu_init,
                    'mu_target': mu_final,
                    'v_nodes': v_nodes_norm,
                    'T': T_val,
                    'D': D,
                    'alpha': alpha,
                    'ic_type': ic_type,
                })

    print(f"  Generated {len(test_cases)} advdiff test cases for {geometries}", flush=True)
    return test_cases


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_one_step_advdiff(film_model, mgn, direct_gnn, test_cases, device,
                          n_ode_steps=50):
    """
    Evaluate one-step prediction for all methods on test_cases.

    Returns dict with arrays of TV values per method:
        {'learned': [...], 'mgn': [...], 'direct_gnn': [...], 'euler': [...]}
    """
    film_model.eval()
    mgn.eval()
    direct_gnn.eval()

    tv_learned  = []
    tv_mgn      = []
    tv_direct   = []
    tv_euler    = []

    for tc in test_cases:
        N        = tc['N']
        mu_src   = tc['mu_source']
        mu_tgt   = tc['mu_target']
        R        = tc['R']
        T        = tc['T']
        D        = tc['D']
        alpha    = tc['alpha']
        bnd      = tc['boundary']
        ei       = tc['edge_index']
        v_nodes  = tc['v_nodes']   # (N, 2) normalized

        node_ctx   = np.stack(
            [mu_src, bnd, v_nodes[:, 0], v_nodes[:, 1]],
            axis=1).astype(np.float32)   # (N, 4)
        global_ctx = np.array([T, D, alpha], dtype=np.float32)  # (3,)

        # Learned FM model
        try:
            mu_pred_learned = sample_posterior_film(
                film_model, mu_src[None], node_ctx, global_ctx, ei,
                n_steps=n_ode_steps, device=device
            )[0]
            tv_learned.append(0.5 * np.abs(mu_pred_learned - mu_tgt).sum())
        except Exception:
            tv_learned.append(float('nan'))

        # AdvDiffMGN
        try:
            mu_t    = torch.tensor(mu_src,   dtype=torch.float32, device=device)
            bnd_t   = torch.tensor(bnd,      dtype=torch.float32, device=device)
            vnodes_t = torch.tensor(v_nodes, dtype=torch.float32, device=device)
            ei_d    = ei.to(device)
            mu_pred_mgn = mgn.predict(mu_t, T, bnd_t, vnodes_t, ei_d)
            tv_mgn.append(0.5 * np.abs(mu_pred_mgn - mu_tgt).sum())
        except Exception:
            tv_mgn.append(float('nan'))

        # DirectGNN
        try:
            T_broadcast = np.full(N, T, dtype=np.float32)
            context = np.stack(
                [mu_src, bnd, v_nodes[:, 0], v_nodes[:, 1], T_broadcast],
                axis=1)  # (N, 5)
            ctx_t = torch.tensor(context, dtype=torch.float32, device=device)
            ei_d  = ei.to(device)
            with torch.no_grad():
                mu_pred_direct = direct_gnn(ctx_t, ei_d).cpu().numpy()
            tv_direct.append(0.5 * np.abs(mu_pred_direct - mu_tgt).sum())
        except Exception:
            tv_direct.append(float('nan'))

        # Euler baseline
        try:
            mu_euler = baseline_euler(mu_src, R, T)
            tv_euler.append(0.5 * np.abs(mu_euler - mu_tgt).sum())
        except Exception:
            tv_euler.append(float('nan'))

    return {
        'learned':    np.array(tv_learned),
        'mgn':        np.array(tv_mgn),
        'direct_gnn': np.array(tv_direct),
        'euler':      np.array(tv_euler),
    }


def eval_rollout_advdiff(film_model, mgn, direct_gnn, test_case, K=10,
                         device='cpu', n_ode_steps=30):
    """
    Evaluate long-horizon rollout for all methods on a single test_case.

    Returns dict with:
        'exact_traj'      : list of K+1 (N,) arrays
        'learned_traj'    : list of K+1 (N,) arrays
        'mgn_traj'        : list of K+1 (N,) arrays
        'direct_gnn_traj' : list of K+1 (N,) arrays
        'euler_traj'      : list of K+1 (N,) arrays
        'learned'         : (K+1,) TV array
        'mgn'             : (K+1,) TV array
        'direct_gnn'      : (K+1,) TV array
        'euler'           : (K+1,) TV array
    """
    film_model.eval()
    mgn.eval()
    direct_gnn.eval()

    N        = test_case['N']
    mu_init  = test_case['mu_source']
    R        = test_case['R']
    T        = test_case['T']
    D        = test_case['D']
    alpha    = test_case['alpha']
    bnd      = test_case['boundary']
    ei       = test_case['edge_index']
    v_nodes  = test_case['v_nodes']   # (N, 2) normalized

    # Exact trajectory
    exact_traj = [mu_init.copy()]
    mu_exact = mu_init.copy()
    for _ in range(K):
        mu_exact = simulate_exact(mu_exact, R, T)
        exact_traj.append(mu_exact.copy())

    def _tv(a, b):
        return 0.5 * np.abs(a - b).sum()

    # FM rollout (autoregressive: mu_current as node_ctx[0])
    learned_traj = [mu_init.copy()]
    learned_tv   = [0.0]
    mu_current   = mu_init.copy()
    for step in range(K):
        node_ctx   = np.stack(
            [mu_current, bnd, v_nodes[:, 0], v_nodes[:, 1]],
            axis=1).astype(np.float32)
        global_ctx = np.array([T, D, alpha], dtype=np.float32)
        try:
            mu_next = sample_posterior_film(
                film_model, mu_current[None], node_ctx, global_ctx, ei,
                n_steps=n_ode_steps, device=device
            )[0]
            mu_current = mu_next
        except Exception:
            pass
        learned_traj.append(mu_current.copy())
        learned_tv.append(_tv(mu_current, exact_traj[step + 1]))

    # MGN rollout
    mgn_traj = [mu_init.copy()]
    mgn_tv   = [0.0]
    mu_current_mgn = mu_init.copy()
    for step in range(K):
        mu_t     = torch.tensor(mu_current_mgn, dtype=torch.float32, device=device)
        bnd_t    = torch.tensor(bnd,    dtype=torch.float32, device=device)
        vnodes_t = torch.tensor(v_nodes, dtype=torch.float32, device=device)
        ei_d     = ei.to(device)
        mu_next_mgn = mgn.predict(mu_t, T, bnd_t, vnodes_t, ei_d)
        mu_current_mgn = mu_next_mgn
        mgn_traj.append(mu_current_mgn.copy())
        mgn_tv.append(_tv(mu_current_mgn, exact_traj[step + 1]))

    # DirectGNN rollout
    direct_traj = [mu_init.copy()]
    direct_tv   = [0.0]
    mu_current_direct = mu_init.copy()
    for step in range(K):
        T_broadcast = np.full(N, T, dtype=np.float32)
        context = np.stack(
            [mu_current_direct, bnd, v_nodes[:, 0], v_nodes[:, 1], T_broadcast],
            axis=1)
        ctx_t = torch.tensor(context, dtype=torch.float32, device=device)
        ei_d  = ei.to(device)
        with torch.no_grad():
            mu_next_direct = direct_gnn(ctx_t, ei_d).cpu().numpy()
        mu_current_direct = mu_next_direct
        direct_traj.append(mu_current_direct.copy())
        direct_tv.append(_tv(mu_current_direct, exact_traj[step + 1]))

    # Euler rollout
    euler_traj = [mu_init.copy()]
    euler_tv   = [0.0]
    mu_current_euler = mu_init.copy()
    for step in range(K):
        mu_next_euler = baseline_euler(mu_current_euler, R, T)
        mu_current_euler = mu_next_euler
        euler_traj.append(mu_current_euler.copy())
        euler_tv.append(_tv(mu_current_euler, exact_traj[step + 1]))

    return {
        'exact_traj':       exact_traj,
        'learned_traj':     learned_traj,
        'mgn_traj':         mgn_traj,
        'direct_gnn_traj':  direct_traj,
        'euler_traj':       euler_traj,
        'learned':          np.array(learned_tv),
        'mgn':              np.array(mgn_tv),
        'direct_gnn':       np.array(direct_tv),
        'euler':            np.array(euler_tv),
    }


# ── Results Printing ──────────────────────────────────────────────────────────

def print_results_advdiff(id_results, ood_results, rollout_results):
    """Print experiment results."""
    methods      = ['learned', 'mgn', 'direct_gnn', 'euler']
    method_names = ['Learned (FM)', 'AdvDiffMGN', 'DirectGNN', 'Euler']

    print("\n" + "="*60, flush=True)
    print("  EX16b Advection-Diffusion Results", flush=True)
    print("="*60, flush=True)
    print(f"\n{'Method':<20} {'ID TV':>10} {'OOD TV':>10} {'Gap':>10}", flush=True)
    print("-"*52, flush=True)

    for method, name in zip(methods, method_names):
        id_tv   = id_results[method]
        ood_tv  = ood_results[method]
        id_mean  = float(np.nanmean(id_tv))
        ood_mean = float(np.nanmean(ood_tv))
        gap = ood_mean - id_mean
        print(f"  {name:<18} {id_mean:>10.4f} {ood_mean:>10.4f} {gap:>10.4f}", flush=True)

    print("\nLong-horizon Rollout (final TV at K steps):", flush=True)
    print("-"*52, flush=True)
    for method, name in zip(methods, method_names):
        final_tv = float(rollout_results[method][-1])
        print(f"  {name:<18} {final_tv:>10.4f}", flush=True)
    print("="*60 + "\n", flush=True)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_main_figure_advdiff(train_losses, id_results, ood_results,
                             rollout_results, train_test_meshes,
                             rollout_test_case, save_path):
    """
    2×3 figure summarising the advection-diffusion experiment.

    Panels:
        A  Training loss (semilogy)
        B  2×4 mesh thumbnails (train top, test bottom)
        C  Grouped bar chart: TV in-dist vs OOD for all 4 methods
        D  One test case: tripcolor + quiver velocity arrows
        E  Long-horizon rollout TV curves
        F  Generalization gap per OOD geometry
    """
    fig = plt.figure(figsize=(18, 11))
    gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    methods       = ['learned', 'mgn', 'direct_gnn', 'euler']
    method_names  = ['FM (Learned)', 'AdvDiffMGN', 'DirectGNN', 'Euler']
    method_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # ── Panel A: Training loss ─────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.semilogy(train_losses, color='steelblue', lw=1.5)
    ax_a.set_xlabel('Epoch')
    ax_a.set_ylabel('Loss')
    ax_a.set_title('A  Training Loss')
    ax_a.grid(True, alpha=0.3)

    # ── Panel B: Mesh thumbnails ───────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_visible(False)
    inner_gs = gs[0, 1].subgridspec(2, 4, hspace=0.05, wspace=0.05)

    train_show = ['square', 'circle', 'triangle', 'hexagon']
    test_show  = ['L_shape', 'annulus', 'star', 'trapezoid']

    for col_idx, geo in enumerate(train_show):
        ax = fig.add_subplot(inner_gs[0, col_idx])
        if geo in train_test_meshes.get('train', {}):
            pts, tris = train_test_meshes['train'][geo]
            ax.triplot(pts[:, 0], pts[:, 1], tris, lw=0.3, color='navy')
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(geo.replace('_', '\n'), fontsize=5, pad=1)

    for col_idx, geo in enumerate(test_show):
        ax = fig.add_subplot(inner_gs[1, col_idx])
        if geo in train_test_meshes.get('test', {}):
            pts, tris = train_test_meshes['test'][geo]
            ax.triplot(pts[:, 0], pts[:, 1], tris, lw=0.3, color='darkred')
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(geo.replace('_', '\n'), fontsize=5, pad=1)

    fig.text(0.38, 0.92, 'B  Mesh Geometries (top: train, bottom: test)',
             fontsize=9, ha='center')

    # ── Panel C: Grouped bar chart ─────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    x     = np.arange(len(methods))
    width = 0.35
    id_means  = [float(np.nanmean(id_results[m]))  for m in methods]
    ood_means = [float(np.nanmean(ood_results[m])) for m in methods]
    ax_c.bar(x - width/2, id_means,  width, label='In-dist',
             color=method_colors, alpha=0.85)
    ax_c.bar(x + width/2, ood_means, width, label='OOD',
             color=method_colors, alpha=0.45, hatch='//')
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(
        [n.replace(' (Learned)', '').replace('AdvDiff', '') for n in method_names],
        fontsize=7, rotation=15)
    ax_c.set_ylabel('Total Variation')
    ax_c.set_title('C  TV: In-dist vs OOD')
    ax_c.legend(fontsize=7)
    ax_c.grid(True, axis='y', alpha=0.3)

    # ── Panel D: Test case distribution + velocity arrows ─────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    if rollout_test_case is not None:
        tc     = rollout_test_case
        pts    = tc['points']
        tris   = tc['triangles']
        mu_src = tc['mu_source']
        v_nrm  = tc['v_nodes']   # (N, 2) normalized
        triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
        im = ax_d.tripcolor(triang, mu_src, shading='gouraud', cmap='hot_r')
        # Quiver velocity arrows
        scale  = (pts[:, 0].max() - pts[:, 0].min()) * 0.08
        ax_d.quiver(pts[:, 0], pts[:, 1],
                    v_nrm[:, 0] * scale, v_nrm[:, 1] * scale,
                    color='cyan', alpha=0.7, scale=1.0, scale_units='xy',
                    width=0.003)
        ax_d.set_aspect('equal')
        ax_d.axis('off')
        ax_d.set_title(
            f'D  {tc["geo"]} / {tc["field_type"]}\nInitial dist + velocity',
            fontsize=8)
        plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    else:
        ax_d.set_title('D  (no test case)', fontsize=8)

    # ── Panel E: Long-horizon rollout TV ──────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    K     = len(rollout_results['learned']) - 1
    steps = np.arange(K + 1)
    for method, name, color in zip(methods, method_names, method_colors):
        ax_e.plot(steps, rollout_results[method], label=name, color=color, lw=1.5)
    ax_e.set_xlabel('Rollout Step')
    ax_e.set_ylabel('Total Variation')
    ax_e.set_title('E  Long-horizon Rollout')
    ax_e.legend(fontsize=7)
    ax_e.grid(True, alpha=0.3)

    # ── Panel F: Generalization gap per OOD geometry ──────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    if isinstance(ood_results.get('learned'), dict):
        test_geos = list(ood_results['learned'].keys())
        x = np.arange(len(test_geos))
        w = 0.2
        for i, (method, color) in enumerate(zip(methods, method_colors)):
            id_mean = float(np.nanmean(id_results[method]))
            geo_gaps = [
                float(np.nanmean(ood_results[method].get(g, [float('nan')]))) - id_mean
                for g in test_geos]
            ax_f.bar(x + (i - 1.5) * w, geo_gaps, w,
                     label=method_names[i], color=color, alpha=0.8)
        ax_f.set_xticks(x)
        ax_f.set_xticklabels(test_geos, rotation=20, fontsize=7)
        ax_f.set_ylabel('OOD - ID TV')
    else:
        id_means_f  = [float(np.nanmean(id_results[m]))  for m in methods]
        ood_means_f = [float(np.nanmean(ood_results[m])) for m in methods]
        gaps = [o - i for o, i in zip(ood_means_f, id_means_f)]
        ax_f.bar(range(len(methods)), gaps, color=method_colors, alpha=0.8)
        ax_f.set_xticks(range(len(methods)))
        ax_f.set_xticklabels(
            [n.replace(' (Learned)', '') for n in method_names],
            rotation=20, fontsize=7)
        ax_f.set_ylabel('OOD - ID TV')

    ax_f.axhline(0, color='black', lw=0.8, ls='--')
    ax_f.set_title('F  Generalization Gap')
    ax_f.grid(True, axis='y', alpha=0.3)
    ax_f.legend(fontsize=6)

    fig.suptitle('Ex16b: Advection-Diffusion Flow Matching on Mesh Geometries',
                 fontsize=12, y=0.98)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved main figure: {save_path}", flush=True)


def plot_rollout_comparison(rollout_results, test_case, save_path):
    """
    5 rows × (K+1) columns rollout figure.

    Rows: Exact / FM / MGN / DirectGNN / Euler
    Each cell: tripcolor + faint quiver arrows + TV annotation
    """
    row_keys  = ['exact_traj', 'learned_traj', 'mgn_traj', 'direct_gnn_traj', 'euler_traj']
    row_names = ['Exact', 'FM (Learned)', 'AdvDiffMGN', 'DirectGNN', 'Euler']
    tv_keys   = [None, 'learned', 'mgn', 'direct_gnn', 'euler']

    exact_traj = rollout_results['exact_traj']
    K1 = len(exact_traj)  # K+1

    pts    = test_case['points']
    tris   = test_case['triangles']
    v_nrm  = test_case['v_nodes']
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
    scale  = (pts[:, 0].max() - pts[:, 0].min()) * 0.06

    fig, axes = plt.subplots(5, K1, figsize=(2.5 * K1, 12))
    if K1 == 1:
        axes = axes[:, None]

    for row_idx, (rkey, rname, tvkey) in enumerate(zip(row_keys, row_names, tv_keys)):
        traj = rollout_results[rkey]
        vmax = max(np.abs(d).max() for d in traj) + 1e-10

        for col_idx in range(K1):
            ax = axes[row_idx, col_idx]
            mu = traj[col_idx]
            ax.tripcolor(triang, mu, shading='gouraud',
                         vmin=0, vmax=vmax, cmap='hot_r')
            ax.quiver(pts[:, 0], pts[:, 1],
                      v_nrm[:, 0] * scale, v_nrm[:, 1] * scale,
                      color='cyan', alpha=0.4, scale=1.0, scale_units='xy',
                      width=0.003)
            ax.set_aspect('equal')
            ax.axis('off')

            if tvkey is not None:
                tv_val = float(rollout_results[tvkey][col_idx])
                ax.set_title(f'TV={tv_val:.3f}', fontsize=6, pad=1)
            else:
                ax.set_title(f't={col_idx}', fontsize=6, pad=1)

        axes[row_idx, 0].set_ylabel(rname, fontsize=8)

    fig.suptitle(
        f'Rollout comparison — {test_case["geo"]} / {test_case["field_type"]}',
        fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved rollout figure: {save_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ex16b: Advection-Diffusion Flow Matching on Meshes')
    parser.add_argument('--n-points',         type=int,   default=60)
    parser.add_argument('--n-meshes-per-geo', type=int,   default=10)
    parser.add_argument('--n-fields-per-mesh',type=int,   default=3)
    parser.add_argument('--n-traj-per-field', type=int,   default=3)
    parser.add_argument('--n-samples',        type=int,   default=20000)
    parser.add_argument('--n-epochs',         type=int,   default=1000)
    parser.add_argument('--hidden-dim',       type=int,   default=64)
    parser.add_argument('--n-layers',         type=int,   default=4)
    parser.add_argument('--lr',               type=float, default=5e-4)
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--n-test-meshes',    type=int,   default=5)
    parser.add_argument('--n-test-cases',     type=int,   default=5)
    parser.add_argument('--rollout-steps',    type=int,   default=10)
    parser.add_argument('--alpha-range',      type=float, nargs=2, default=[0.5, 2.0])
    parser.add_argument('--diffusion-coeff',  type=float, default=1.0)
    parser.add_argument('--T-range',          type=float, nargs=2, default=[0.05, 0.2])
    args = parser.parse_args()

    device = get_device()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    meta_ckpt = os.path.join(
        ckpt_dir,
        f'meta_model_ex16b_advdiff_{args.n_epochs}ep_h{args.hidden_dim}_l{args.n_layers}.pt')
    mgn_ckpt       = os.path.join(ckpt_dir, f'mgn_ex16b_{args.n_epochs}ep.pt')
    direct_gnn_ckpt = os.path.join(ckpt_dir, f'direct_gnn_ex16b_{args.n_epochs}ep.pt')

    D = args.diffusion_coeff

    # ── Dataset ────────────────────────────────────────────────────────────────
    print("\nBuilding advection-diffusion training dataset...", flush=True)
    dataset = AdvectionDiffusionDataset(
        geometries=TRAIN_GEOS,
        n_meshes_per_geo=args.n_meshes_per_geo,
        n_fields_per_mesh=args.n_fields_per_mesh,
        n_traj_per_field=args.n_traj_per_field,
        T_range=tuple(args.T_range),
        n_samples=args.n_samples,
        n_points=args.n_points,
        D=D,
        alpha_range=tuple(args.alpha_range),
        seed=args.seed,
    )

    # ── FiLM model ─────────────────────────────────────────────────────────────
    # node_context_dim=4: [mu_init, boundary, v_x, v_y]
    # global_dim=3:       [T, D, alpha]
    film_model = FiLMConditionalGNNRateMatrixPredictor(
        node_context_dim=4,
        global_dim=3,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    )

    train_losses = []
    if os.path.exists(meta_ckpt):
        print(f"\nLoading FiLM model from {meta_ckpt}", flush=True)
        state = torch.load(meta_ckpt, map_location=device)
        film_model.load_state_dict(state['model'])
        train_losses = state.get('losses', [])
    else:
        print(f"\nTraining FiLM model ({args.n_epochs} epochs)...", flush=True)
        result = train_film_conditional(
            film_model, dataset,
            n_epochs=args.n_epochs,
            batch_size=256,
            lr=args.lr,
            device=device,
            loss_weighting='uniform',
            loss_type='rate_kl',
        )
        train_losses = result['losses']
        torch.save({'model': film_model.state_dict(), 'losses': train_losses}, meta_ckpt)
        print(f"  Saved FiLM model to {meta_ckpt}", flush=True)

    film_model = film_model.to(device)

    # ── AdvDiffMGN baseline ────────────────────────────────────────────────────
    mgn = AdvDiffMGN(hidden_dim=64, n_layers=3)

    if os.path.exists(mgn_ckpt):
        print(f"\nLoading AdvDiffMGN from {mgn_ckpt}", flush=True)
        mgn.load_state_dict(torch.load(mgn_ckpt, map_location=device))
    else:
        print(f"\nTraining AdvDiffMGN ({args.n_epochs} epochs)...", flush=True)
        train_mgn_advdiff(
            mgn, dataset.all_pairs,
            n_epochs=args.n_epochs,
            lr=5e-4,
            device=device,
        )
        torch.save(mgn.state_dict(), mgn_ckpt)
        print(f"  Saved AdvDiffMGN to {mgn_ckpt}", flush=True)

    mgn = mgn.to(device)

    # ── DirectGNN baseline ─────────────────────────────────────────────────────
    # context_dim=5: [mu_src, boundary, v_x, v_y, T_broadcast]
    direct_gnn = DirectGNNPredictor(context_dim=5, hidden_dim=64, n_layers=4)

    if os.path.exists(direct_gnn_ckpt):
        print(f"\nLoading DirectGNN from {direct_gnn_ckpt}", flush=True)
        direct_gnn.load_state_dict(torch.load(direct_gnn_ckpt, map_location=device))
    else:
        print(f"\nTraining DirectGNN ({args.n_epochs} epochs)...", flush=True)
        dgnn_pairs = build_direct_gnn_pairs_advdiff(dataset.all_pairs)
        result_dgnn = train_direct_gnn(
            direct_gnn, dgnn_pairs,
            n_epochs=args.n_epochs,
            lr=5e-4,
            device=device,
            seed=args.seed,
        )
        torch.save(direct_gnn.state_dict(), direct_gnn_ckpt)
        print(f"  Saved DirectGNN to {direct_gnn_ckpt}", flush=True)

    direct_gnn = direct_gnn.to(device)

    # ── Generate test cases ────────────────────────────────────────────────────
    print("\nGenerating in-distribution test cases...", flush=True)
    id_test_cases = generate_test_cases_advdiff(
        TRAIN_GEOS,
        n_meshes=args.n_test_meshes,
        n_cases_per_mesh=args.n_test_cases,
        T_range=tuple(args.T_range),
        n_points=args.n_points,
        D=D,
        alpha_range=tuple(args.alpha_range),
        seed=9000,
    )

    print("Generating OOD test cases...", flush=True)
    ood_test_cases = generate_test_cases_advdiff(
        TEST_GEOS,
        n_meshes=args.n_test_meshes,
        n_cases_per_mesh=args.n_test_cases,
        T_range=tuple(args.T_range),
        n_points=args.n_points,
        D=D,
        alpha_range=tuple(args.alpha_range),
        seed=9001,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\nEvaluating in-distribution...", flush=True)
    id_results = eval_one_step_advdiff(
        film_model, mgn, direct_gnn, id_test_cases, device, n_ode_steps=50)

    print("Evaluating OOD...", flush=True)
    ood_results = eval_one_step_advdiff(
        film_model, mgn, direct_gnn, ood_test_cases, device, n_ode_steps=50)

    # Pick rollout test case: prefer first OOD with known geo
    rollout_case = None
    for tc in ood_test_cases:
        if tc['geo'] == 'L_shape':
            rollout_case = tc
            break
    if rollout_case is None and ood_test_cases:
        rollout_case = ood_test_cases[0]

    rollout_results = {
        'exact_traj': [], 'learned_traj': [], 'mgn_traj': [],
        'direct_gnn_traj': [], 'euler_traj': [],
        'learned':    np.zeros(args.rollout_steps + 1),
        'mgn':        np.zeros(args.rollout_steps + 1),
        'direct_gnn': np.zeros(args.rollout_steps + 1),
        'euler':      np.zeros(args.rollout_steps + 1),
    }

    if rollout_case is not None:
        print(f"\nEvaluating rollout on {rollout_case['geo']} / "
              f"{rollout_case['field_type']}...", flush=True)
        rollout_results = eval_rollout_advdiff(
            film_model, mgn, direct_gnn, rollout_case,
            K=args.rollout_steps, device=device, n_ode_steps=30)

    # ── Print results ──────────────────────────────────────────────────────────
    print_results_advdiff(id_results, ood_results, rollout_results)

    # ── Collect meshes for visualization ──────────────────────────────────────
    train_test_meshes = {'train': {}, 'test': {}}
    for geo in ['square', 'circle', 'triangle', 'hexagon']:
        try:
            pts, tris, _ = generate_mesh(geo, args.n_points, seed=args.seed)
            train_test_meshes['train'][geo] = (pts, tris)
        except Exception:
            pass
    for geo in ['L_shape', 'annulus', 'star', 'trapezoid']:
        try:
            pts, tris, _ = generate_mesh(geo, args.n_points, seed=args.seed + 1)
            train_test_meshes['test'][geo] = (pts, tris)
        except Exception:
            pass

    # ── Main figure ────────────────────────────────────────────────────────────
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ex16b_advdiff.png')
    print("\nGenerating main figure...", flush=True)
    plot_main_figure_advdiff(
        train_losses=train_losses,
        id_results=id_results,
        ood_results=ood_results,
        rollout_results=rollout_results,
        train_test_meshes=train_test_meshes,
        rollout_test_case=rollout_case,
        save_path=save_path,
    )

    # ── Rollout figure ─────────────────────────────────────────────────────────
    if rollout_case is not None and rollout_results.get('exact_traj'):
        rollout_save = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'ex16b_rollout.png')
        print("\nGenerating rollout comparison figure...", flush=True)
        plot_rollout_comparison(rollout_results, rollout_case, rollout_save)

    print("Done.", flush=True)


if __name__ == '__main__':
    main()
