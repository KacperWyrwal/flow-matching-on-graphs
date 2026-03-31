"""
Experiment 18: Diffusion Source Recovery Across Graph Topologies and Sizes.

Given observation = source @ expm(tau * R), recover the source distribution.
Uses FiLMConditionalGNNRateMatrixPredictor with node_context_dim=1 (obs value
per node), global_dim=1 (tau).  Trains on diverse graph zoo and tests
generalization to unseen topologies, sizes, and tau ranges.
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
from scipy.linalg import expm
from scipy.stats import pearsonr

from graph_ot_fm import (
    GraphStructure,
    GeodesicCache,
    total_variation,
)
from graph_ot_fm.ot_solver import compute_ot_coupling
from graph_ot_fm.flow import marginal_distribution_fast, marginal_rate_matrix_fast

from meta_fm import (
    FiLMConditionalGNNRateMatrixPredictor,
    DirectGNNPredictor,
    train_film_conditional,
    train_direct_gnn,
    get_device,
    EMA,
)
from meta_fm.model import rate_matrix_to_edge_index
from meta_fm.sample import sample_posterior_film

from experiments.ex17_ot_generalization import (
    generate_training_graphs,
    generate_test_graphs_topology,
    generate_test_graphs_size,
    draw_graph_with_dist,
)


# -- Source distribution generation -------------------------------------------

def generate_source_distribution(N, points, rng, R=None):
    """Generate varied source distributions."""
    ic_type = rng.choice([
        'single_peak', 'multi_peak', 'gradient', 'smooth_random',
        'clustered', 'near_uniform', 'power_law',
    ])

    if ic_type == 'single_peak':
        mu = np.ones(N) * 0.01
        mu[rng.integers(N)] += 1.0
    elif ic_type == 'multi_peak':
        n_peaks = int(rng.integers(2, 6))
        peaks = rng.choice(N, size=min(n_peaks, N), replace=False)
        weights = rng.dirichlet(np.ones(len(peaks)))
        mu = np.ones(N) * 1e-3
        for p, w in zip(peaks, weights):
            mu[p] += w
    elif ic_type == 'gradient':
        direction = rng.standard_normal(2)
        direction /= np.linalg.norm(direction) + 1e-10
        proj = points @ direction
        proj = proj - proj.min()
        mu = proj / (proj.max() + 1e-10) + 0.05
    elif ic_type == 'smooth_random':
        n_bumps = int(rng.integers(2, 9))
        mu = np.zeros(N)
        for _ in range(n_bumps):
            center = rng.uniform(points.min(0), points.max(0))
            sigma = rng.uniform(0.05, 0.3)
            dists = np.linalg.norm(points - center, axis=1)
            mu += rng.uniform(0.5, 2.0) * np.exp(-0.5 * (dists / sigma) ** 2)
        mu += 1e-3
    elif ic_type == 'clustered':
        seed = int(rng.integers(N))
        k_hops = int(rng.integers(1, 4))
        if R is not None:
            R_off = R.copy()
            np.fill_diagonal(R_off, 0)
            visited = {seed}
            frontier = {seed}
            for _ in range(k_hops):
                new_frontier = set()
                for node in frontier:
                    for nb in range(N):
                        if R_off[node, nb] > 0:
                            new_frontier.add(nb)
                frontier = new_frontier - visited
                visited |= frontier
        else:
            visited = {seed}
        mu = np.ones(N) * 0.01
        for node in visited:
            mu[node] = rng.uniform(0.1, 1.0)
    elif ic_type == 'near_uniform':
        mu = np.ones(N) + rng.normal(0, 0.1, size=N)
        mu = np.clip(mu, 0.01, None)
    elif ic_type == 'power_law':
        mu = rng.pareto(a=1.5, size=N) + 0.01
    else:
        mu = np.ones(N)

    mu = np.clip(mu, 1e-6, None)
    mu /= mu.sum()
    return mu.astype(np.float32)


# -- Observation generation ---------------------------------------------------

def generate_observation(mu_source, R, tau):
    """Forward diffusion: observation = source @ expm(tau * R).

    Returns float64 to preserve precision for baseline evaluation.
    Cast to float32 only when creating model input tensors.
    """
    obs = mu_source.astype(np.float64) @ expm(tau * R.astype(np.float64))
    obs = np.clip(obs, 0, None)
    obs /= obs.sum() + 1e-15
    return obs  # keep float64


def compute_difficulty(mu_source, obs):
    """Difficulty = TV between source and observation."""
    return 0.5 * np.abs(mu_source - obs).sum()


def sample_dirichlet_start(N, obs, alpha, mode, rng):
    """Sample a starting distribution for posterior/training.

    mode='uniform':      Dirichlet(alpha, ..., alpha)
    mode='obs_centered': Dirichlet(alpha * N * obs_normalized)
    """
    if mode == 'obs_centered':
        obs_safe = np.clip(obs, 1e-4, None)
        obs_safe = obs_safe / obs_safe.sum()
        return rng.dirichlet(alpha * N * obs_safe).astype(np.float32)
    else:
        return rng.dirichlet(np.full(N, alpha)).astype(np.float32)


# -- Dataset ------------------------------------------------------------------

class DiffusionSourceDataset(torch.utils.data.Dataset):
    """Training data for diffusion source recovery.

    For each graph x n_pairs_per_graph:
      - Sample source, sample tau from tau_range, compute observation
      - node_ctx = obs[:, None]  (N, 1)
      - global_ctx = [tau]       (1,)
      - OT coupling: Dirichlet start -> source
      - Sample flow matching 7-tuples for train_film_conditional

    Returns 7-tuple:
        (mu_tau, tau_flow, node_ctx, global_ctx, u_tilde, edge_index, N)
    """

    def __init__(self, graphs, tau_range=(0.3, 1.5),
                 n_pairs_per_graph=20, n_starts_per_pair=5,
                 dirichlet_alpha=1.0, start_mode='uniform',
                 n_samples=20000, seed=42):
        rng = np.random.default_rng(seed)
        all_items = []
        self.all_pairs = []

        for name, R, pos in graphs:
            N = R.shape[0]
            graph_struct = GraphStructure(R)
            geo_cache = GeodesicCache(graph_struct)
            edge_index = rate_matrix_to_edge_index(R)

            for pair_idx in range(n_pairs_per_graph):
                # Sample source distribution
                mu_source = generate_source_distribution(N, pos, rng, R=R)

                # Sample diffusion time
                tau = float(rng.uniform(tau_range[0], tau_range[1]))

                # Compute observation
                obs = generate_observation(mu_source, R, tau)

                # Context
                node_ctx = obs[:, None].astype(np.float32)  # (N, 1)
                global_ctx = np.array([tau], dtype=np.float32)  # (1,)

                difficulty = compute_difficulty(mu_source, obs)

                self.all_pairs.append({
                    'name': name,
                    'N': N,
                    'R': R,
                    'positions': pos,
                    'edge_index': edge_index,
                    'mu_source': mu_source,
                    'obs': obs,
                    'tau': tau,
                    'difficulty': difficulty,
                })

                # Multiple Dirichlet starts per pair for epistemic diversity
                n_per = max(1, n_samples // (
                    len(graphs) * n_pairs_per_graph * n_starts_per_pair))

                for _ in range(n_starts_per_pair):
                    mu_start = sample_dirichlet_start(
                        N, obs, dirichlet_alpha, start_mode, rng)

                    # OT coupling: Dirichlet start -> source
                    coupling = compute_ot_coupling(
                        mu_start, mu_source, graph_struct=graph_struct)
                    geo_cache.precompute_for_coupling(coupling)

                    for _ in range(n_per):
                        tau_flow = float(rng.uniform(0.0, 0.999))
                        mu_tau = marginal_distribution_fast(
                            geo_cache, coupling, tau_flow)
                        R_target = marginal_rate_matrix_fast(
                            geo_cache, coupling, tau_flow)
                        u_tilde = R_target * (1.0 - tau_flow)

                        all_items.append((
                            torch.tensor(mu_tau, dtype=torch.float32),
                            torch.tensor([tau_flow], dtype=torch.float32),
                            torch.tensor(node_ctx, dtype=torch.float32),
                            torch.tensor(global_ctx, dtype=torch.float32),
                            torch.tensor(u_tilde, dtype=torch.float32),
                            edge_index,
                            N,
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


# -- Baselines ----------------------------------------------------------------

def baseline_backprojection(obs):
    """Baseline: return observation unchanged."""
    return obs.copy()


def baseline_laplacian_sharpening(obs, R, tau):
    """Baseline (oracle): obs @ expm(-tau * R), clip and normalize.

    Uses float64 throughout to avoid precision loss. This is an oracle
    baseline that has access to the exact rate matrix R.
    """
    try:
        obs64 = np.asarray(obs, dtype=np.float64)
        R64 = np.asarray(R, dtype=np.float64)
        mu_hat = obs64 @ expm(-tau * R64)
        mu_hat = np.clip(mu_hat, 0, None)
        s = mu_hat.sum()
        if s > 1e-15:
            mu_hat /= s
        else:
            return obs.copy()
        return mu_hat
    except Exception:
        return obs.copy()


# -- Test case generation -----------------------------------------------------

TAU_IN_RANGE = [0.3, 0.5, 0.8, 1.0, 1.5]
TAU_SHORT = [0.1, 0.15, 0.2]
TAU_LONG = [2.0, 2.5, 3.0]


def generate_test_cases(graphs, tau_values, n_per_graph=5, seed=8888):
    """Generate test cases: list of dicts with graph info, source, obs, tau."""
    rng = np.random.default_rng(seed)
    cases = []
    for name, R, pos in graphs:
        N = R.shape[0]
        edge_index = rate_matrix_to_edge_index(R)
        for _ in range(n_per_graph):
            mu_source = generate_source_distribution(N, pos, rng, R=R)
            tau = float(rng.choice(tau_values))
            obs = generate_observation(mu_source, R, tau)
            difficulty = compute_difficulty(mu_source, obs)
            cases.append({
                'name': name,
                'N': N,
                'R': R,
                'positions': pos,
                'edge_index': edge_index,
                'mu_source': mu_source,
                'obs': obs,
                'tau': tau,
                'difficulty': difficulty,
            })
    return cases


# -- Evaluation ---------------------------------------------------------------

def evaluate_source_recovery(model, test_cases, device,
                             posterior_k=20, dirichlet_alpha=1.0,
                             start_mode='uniform', direct_model=None):
    """Evaluate FM model and baselines on source recovery.

    For FM: K Dirichlet starts -> posterior mean.
    Also computes posterior std, calibration, diversity.

    Returns dict with per-case metrics + metadata.
    """
    results = {
        'fm_tv': [], 'fm_peak': [],
        'direct_tv': [], 'direct_peak': [],
        'backproj_tv': [], 'backproj_peak': [],
        'laplacian_tv': [], 'laplacian_peak': [],
        'names': [], 'sizes': [], 'taus': [], 'difficulties': [],
        # Posterior / calibration metrics
        'fm_posterior_std': [],
        'fm_posterior_err': [],
        'fm_calibration_r': [],
        'fm_diversity': [],
    }

    model.eval()
    if direct_model is not None:
        direct_model.eval()
    rng = np.random.default_rng(12345)

    for idx, case in enumerate(test_cases):
        name = case['name']
        N = case['N']
        R = case['R']
        pos = case['positions']
        edge_index = case['edge_index']
        mu_source = case['mu_source']
        obs = case['obs']
        tau = case['tau']
        difficulty = case['difficulty']

        results['names'].append(name)
        results['sizes'].append(N)
        results['taus'].append(tau)
        results['difficulties'].append(difficulty)

        node_ctx = obs[:, None].astype(np.float32)  # (N, 1)
        global_ctx = np.array([tau], dtype=np.float32)  # (1,)

        # FM: K posterior samples from Dirichlet starts
        mu_starts = np.array([
            sample_dirichlet_start(N, obs, dirichlet_alpha, start_mode, rng)
            for _ in range(posterior_k)
        ])  # (K, N)

        try:
            fm_samples = sample_posterior_film(
                model, mu_starts, node_ctx, global_ctx, edge_index,
                n_steps=100, device=device)
            fm_samples = np.clip(fm_samples, 0, None)
            # Normalize each sample
            sums = fm_samples.sum(axis=1, keepdims=True) + 1e-15
            fm_samples = (fm_samples / sums).astype(np.float32)
        except Exception:
            fm_samples = np.ones((posterior_k, N), dtype=np.float32) / N

        fm_mean = fm_samples.mean(axis=0)
        fm_mean /= fm_mean.sum() + 1e-15
        fm_std = fm_samples.std(axis=0)  # (N,)
        fm_err = np.abs(fm_mean - mu_source)  # (N,)

        # FM metrics
        tv_fm = total_variation(fm_mean, mu_source)
        results['fm_tv'].append(tv_fm)
        results['fm_peak'].append(
            1.0 if np.argmax(fm_mean) == np.argmax(mu_source) else 0.0)

        # Calibration: Pearson r between posterior std and |error|
        if fm_std.std() > 1e-12 and fm_err.std() > 1e-12:
            cal_r, _ = pearsonr(fm_std, fm_err)
            cal_r = cal_r if not np.isnan(cal_r) else 0.0
        else:
            cal_r = 0.0
        results['fm_calibration_r'].append(cal_r)
        results['fm_posterior_std'].append(fm_std)
        results['fm_posterior_err'].append(fm_err)

        # Diversity: mean pairwise TV among K samples
        pair_tvs = []
        for i in range(min(posterior_k, 10)):
            for j in range(i + 1, min(posterior_k, 10)):
                pair_tvs.append(total_variation(fm_samples[i], fm_samples[j]))
        results['fm_diversity'].append(
            np.mean(pair_tvs) if pair_tvs else 0.0)

        # Backprojection baseline
        mu_back = baseline_backprojection(obs)
        tv_back = total_variation(mu_back, mu_source)
        results['backproj_tv'].append(tv_back)
        results['backproj_peak'].append(
            1.0 if np.argmax(mu_back) == np.argmax(mu_source) else 0.0)

        # Laplacian sharpening baseline
        mu_lap = baseline_laplacian_sharpening(obs, R, tau)
        tv_lap = total_variation(mu_lap, mu_source)
        results['laplacian_tv'].append(tv_lap)
        results['laplacian_peak'].append(
            1.0 if np.argmax(mu_lap) == np.argmax(mu_source) else 0.0)

        # DirectGNN baseline
        if direct_model is not None:
            try:
                tau_broadcast = np.full(N, tau, dtype=np.float32)
                ctx_direct = np.stack([obs, tau_broadcast], axis=1)  # (N, 2)
                ctx_t = torch.tensor(ctx_direct, dtype=torch.float32, device=device)
                ei_dev = edge_index.to(device)
                with torch.no_grad():
                    mu_direct = direct_model(ctx_t, ei_dev).cpu().numpy()
            except Exception:
                mu_direct = obs.copy()
        else:
            mu_direct = obs.copy()
        results['direct_tv'].append(total_variation(mu_direct, mu_source))
        results['direct_peak'].append(
            1.0 if np.argmax(mu_direct) == np.argmax(mu_source) else 0.0)

        if (idx + 1) % 10 == 0:
            print(f"    Evaluated {idx+1}/{len(test_cases)}", flush=True)

    return results


# -- Console output -----------------------------------------------------------

def print_results_table(id_res, topo_res, size_res):
    """Print 3x3 table (topology x tau range) for each method, then
    difficulty bins, calibration, and peak recovery."""

    # Categorize by tau range
    def _by_tau_range(res):
        short, inrange, long_ = [], [], []
        for i, tau in enumerate(res['taus']):
            if tau < 0.25:
                short.append(i)
            elif tau <= 1.5:
                inrange.append(i)
            else:
                long_.append(i)
        return short, inrange, long_

    print("\nTV Results by Topology x Tau Range:", flush=True)
    header = (f"{'':22s} {'Short tau':>10s}  {'In-range tau':>12s}  "
              f"{'Long tau':>10s}")
    print(header, flush=True)

    for label, res in [('ID', id_res), ('OOD-topo', topo_res),
                        ('OOD-size', size_res)]:
        short, inrange, long_ = _by_tau_range(res)
        for meth, key in [('FM', 'fm_tv'), ('DirectGNN', 'direct_tv'),
                          ('Backproj', 'backproj_tv'),
                          ('Laplacian', 'laplacian_tv')]:
            s_v = np.nanmean([res[key][i] for i in short]) if short else float('nan')
            ir_v = np.nanmean([res[key][i] for i in inrange]) if inrange else float('nan')
            l_v = np.nanmean([res[key][i] for i in long_]) if long_ else float('nan')
            print(f"  {label+'/'+meth:20s} {s_v:10.4f}  {ir_v:12.4f}  "
                  f"{l_v:10.4f}", flush=True)

    # Difficulty bins
    print("\nTV by Difficulty Bin:", flush=True)
    all_diff = id_res['difficulties'] + topo_res['difficulties'] + size_res['difficulties']
    all_fm_tv = id_res['fm_tv'] + topo_res['fm_tv'] + size_res['fm_tv']
    all_dir_tv = id_res['direct_tv'] + topo_res['direct_tv'] + size_res['direct_tv']
    all_back_tv = id_res['backproj_tv'] + topo_res['backproj_tv'] + size_res['backproj_tv']

    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    print(f"  {'Difficulty':>12s}  {'FM':>8s}  {'DirectGNN':>10s}  {'Backproj':>10s}",
          flush=True)
    for lo, hi in bins:
        fm_vals = [v for d, v in zip(all_diff, all_fm_tv) if lo <= d < hi]
        dir_vals = [v for d, v in zip(all_diff, all_dir_tv) if lo <= d < hi]
        bp_vals = [v for d, v in zip(all_diff, all_back_tv) if lo <= d < hi]
        if fm_vals:
            print(f"  {lo:.1f}-{hi:.1f}        {np.mean(fm_vals):8.4f}  "
                  f"{np.mean(dir_vals):10.4f}  {np.mean(bp_vals):10.4f}",
                  flush=True)

    # Calibration and peak recovery
    print(f"\nCalibration (Pearson r between posterior std and |error|):", flush=True)
    for label, res in [('ID', id_res), ('OOD-topo', topo_res),
                        ('OOD-size', size_res)]:
        cal = np.nanmean(res['fm_calibration_r'])
        div = np.nanmean(res['fm_diversity'])
        print(f"  {label:12s}  calibration r={cal:.4f}, diversity={div:.4f}",
              flush=True)

    print(f"\nPeak Recovery:", flush=True)
    for label, res in [('ID', id_res), ('OOD-topo', topo_res),
                        ('OOD-size', size_res)]:
        fm_peak = np.nanmean(res['fm_peak'])
        dir_peak = np.nanmean(res['direct_peak'])
        bp_peak = np.nanmean(res['backproj_peak'])
        lap_peak = np.nanmean(res['laplacian_peak'])
        print(f"  {label:12s}  FM={fm_peak*100:.0f}%, "
              f"DirectGNN={dir_peak*100:.0f}%, "
              f"Backproj={bp_peak*100:.0f}%, Laplacian={lap_peak*100:.0f}%",
              flush=True)


# -- Plotting -----------------------------------------------------------------

def plot_results(losses, id_res, topo_res, size_res, out_path):
    """Create the 2x3 results figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel A: Training loss
    ax = axes[0, 0]
    if losses:
        ax.plot(losses, 'b-', alpha=0.7, linewidth=0.8)
        ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(A) Training Loss')
    ax.grid(True, alpha=0.3)

    # Panel B: TV vs difficulty scatter
    ax = axes[0, 1]
    all_diff = id_res['difficulties'] + topo_res['difficulties'] + size_res['difficulties']
    all_fm = id_res['fm_tv'] + topo_res['fm_tv'] + size_res['fm_tv']
    all_dir = id_res['direct_tv'] + topo_res['direct_tv'] + size_res['direct_tv']
    all_bp = id_res['backproj_tv'] + topo_res['backproj_tv'] + size_res['backproj_tv']
    ax.scatter(all_diff, all_fm, alpha=0.4, s=12, label='FM', c='tab:blue')
    ax.scatter(all_diff, all_dir, alpha=0.4, s=12, label='DirectGNN',
               c='tab:red', marker='s')
    ax.scatter(all_diff, all_bp, alpha=0.4, s=12, label='Backprojection',
               c='tab:orange', marker='^')
    lims = [0, max(max(all_fm + [0.01]), max(all_bp + [0.01])) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('Difficulty (TV: source vs obs)')
    ax.set_ylabel('Recovery TV')
    ax.set_title('(B) TV vs Difficulty')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel C: TV by tau range (grouped bars)
    ax = axes[0, 2]
    tau_labels = ['Short', 'In-range', 'Long']
    methods = ['FM', 'DirectGNN', 'Backproj', 'Laplacian']
    all_taus = id_res['taus'] + topo_res['taus'] + size_res['taus']
    all_fm_tv = id_res['fm_tv'] + topo_res['fm_tv'] + size_res['fm_tv']
    all_dir_tv = id_res['direct_tv'] + topo_res['direct_tv'] + size_res['direct_tv']
    all_bp_tv = id_res['backproj_tv'] + topo_res['backproj_tv'] + size_res['backproj_tv']
    all_lap_tv = id_res['laplacian_tv'] + topo_res['laplacian_tv'] + size_res['laplacian_tv']

    def _tau_bin(tau):
        if tau < 0.25:
            return 0
        elif tau <= 1.5:
            return 1
        else:
            return 2

    method_tvs = [all_fm_tv, all_dir_tv, all_bp_tv, all_lap_tv]
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green']
    x = np.arange(len(tau_labels))
    width = 0.2
    for i, (meth, tvs, color) in enumerate(zip(methods, method_tvs, colors)):
        means = []
        for b in range(3):
            vals = [v for t, v in zip(all_taus, tvs) if _tau_bin(t) == b]
            means.append(np.nanmean(vals) if vals else 0.0)
        ax.bar(x + i * width, means, width, label=meth, alpha=0.8, color=color)
    ax.set_xticks(x + width)
    ax.set_xticklabels(tau_labels)
    ax.set_ylabel('TV Distance')
    ax.set_title('(C) TV by Tau Range')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel D: TV by topology (grouped bars: ID/OOD-topo/OOD-size)
    ax = axes[1, 0]
    splits = ['ID', 'OOD-topo', 'OOD-size']
    all_results = [id_res, topo_res, size_res]
    x = np.arange(len(splits))
    width = 0.2
    for i, (meth, key, color) in enumerate(zip(
            methods, ['fm_tv', 'direct_tv', 'backproj_tv', 'laplacian_tv'], colors)):
        vals = [np.nanmean(r[key]) for r in all_results]
        ax.bar(x + i * width, vals, width, label=meth, alpha=0.8, color=color)
    ax.set_xticks(x + width)
    ax.set_xticklabels(splits)
    ax.set_ylabel('TV Distance')
    ax.set_title('(D) TV by Topology')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel E: TV vs graph size scatter
    ax = axes[1, 1]
    all_sizes = id_res['sizes'] + topo_res['sizes'] + size_res['sizes']
    ax.scatter(all_sizes, all_fm_tv, alpha=0.4, s=12, label='FM', c='tab:blue')
    ax.scatter(all_sizes, all_dir_tv, alpha=0.4, s=12, label='DirectGNN',
               c='tab:red', marker='s')
    ax.scatter(all_sizes, all_bp_tv, alpha=0.4, s=12, label='Backprojection',
               c='tab:orange', marker='^')
    ax.set_xlabel('Graph Size (N)')
    ax.set_ylabel('TV Distance')
    ax.set_title('(E) TV vs Graph Size')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel F: Calibration scatter (posterior std vs |error|)
    ax = axes[1, 2]
    for label, res, color in [
        ('ID', id_res, 'tab:blue'),
        ('OOD-topo', topo_res, 'tab:red'),
        ('OOD-size', size_res, 'tab:orange'),
    ]:
        all_std = np.concatenate(res['fm_posterior_std']) if res['fm_posterior_std'] else np.array([])
        all_err = np.concatenate(res['fm_posterior_err']) if res['fm_posterior_err'] else np.array([])
        if len(all_std) > 0:
            n_pts = min(500, len(all_std))
            idx_sub = np.random.default_rng(0).choice(
                len(all_std), n_pts, replace=False)
            ax.scatter(all_std[idx_sub], all_err[idx_sub], s=3, alpha=0.3,
                       c=color, label=label)
    all_cal = (id_res['fm_calibration_r'] + topo_res['fm_calibration_r']
               + size_res['fm_calibration_r'])
    mean_cal = np.nanmean(all_cal)
    ax.set_xlabel('Posterior Std')
    ax.set_ylabel('|Posterior Mean - True Source|')
    ax.set_title(f'(F) Calibration (r={mean_cal:.3f})')
    ax.legend(fontsize=6, markerscale=3)
    ax.grid(True, alpha=0.3)

    # Inset: diversity box plot
    inset = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
    div_data = [id_res['fm_diversity'], topo_res['fm_diversity'],
                size_res['fm_diversity']]
    bp = inset.boxplot(div_data, labels=['ID', 'Topo', 'Size'],
                       widths=0.5, patch_artist=True)
    for patch, c in zip(bp['boxes'],
                        ['tab:blue', 'tab:red', 'tab:orange']):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    inset.set_ylabel('Diversity', fontsize=6)
    inset.set_title('Posterior Diversity', fontsize=6)
    inset.tick_params(labelsize=5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def plot_reconstruction_gallery(model, test_cases, out_path, device,
                                posterior_k=20, dirichlet_alpha=1.0,
                                start_mode='uniform'):
    """6 rows (diverse test cases) x 4 cols (source, observation, FM, backprojection)."""
    rng = np.random.default_rng(42)

    # Pick 6 diverse cases
    n_show = min(6, len(test_cases))
    indices = np.linspace(0, len(test_cases) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(n_show, 4, figsize=(14, n_show * 3))
    if n_show == 1:
        axes = axes[None, :]

    col_titles = ['True Source', 'Observation', 'FM Reconstruction', 'Backprojection']

    model.eval()

    for row, case_idx in enumerate(indices):
        case = test_cases[case_idx]
        N = case['N']
        R = case['R']
        pos = case['positions']
        mu_source = case['mu_source']
        obs = case['obs']
        tau = case['tau']
        edge_index = case['edge_index']
        difficulty = case['difficulty']

        node_ctx = obs[:, None].astype(np.float32)
        global_ctx = np.array([tau], dtype=np.float32)

        # FM posterior mean
        mu_starts = np.array([
            sample_dirichlet_start(N, obs, dirichlet_alpha, start_mode, rng)
            for _ in range(posterior_k)
        ])
        try:
            fm_samples = sample_posterior_film(
                model, mu_starts, node_ctx, global_ctx, edge_index,
                n_steps=100, device=device)
            fm_samples = np.clip(fm_samples, 0, None)
            sums = fm_samples.sum(axis=1, keepdims=True) + 1e-15
            fm_samples = (fm_samples / sums).astype(np.float32)
            fm_mean = fm_samples.mean(axis=0)
            fm_mean /= fm_mean.sum() + 1e-15
        except Exception:
            fm_mean = np.ones(N, dtype=np.float32) / N

        mu_back = baseline_backprojection(obs)

        # Shared vmin/vmax
        vmax = max(mu_source.max(), obs.max(), fm_mean.max(), mu_back.max())
        node_size = max(10, min(40, 1500 // max(N, 1)))

        distributions = [mu_source, obs, fm_mean, mu_back]
        for col, (mu, title) in enumerate(zip(distributions, col_titles)):
            ax = axes[row, col]
            draw_graph_with_dist(ax, R, pos, mu=mu,
                                 title=(title if row == 0 else ''),
                                 cmap='hot', node_size=node_size,
                                 vmin=0, vmax=vmax)
            if col == 0:
                ax.set_ylabel(f'{case["name"]}\ntau={tau:.1f}, '
                              f'diff={difficulty:.2f}', fontsize=7)

    fig.suptitle('Source Recovery: Gallery', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def plot_tau_generalization(model, test_cases, out_path, device,
                            posterior_k=20, dirichlet_alpha=1.0,
                            start_mode='uniform'):
    """One OOD graph, fixed source, tau = [0.1, 0.5, 1.0, 2.0, 3.0].
    3 rows: observation / FM reconstruction / true source (repeated)."""
    tau_values = [0.1, 0.5, 1.0, 2.0, 3.0]
    rng = np.random.default_rng(999)

    # Pick one OOD test case for its graph
    case = test_cases[0]
    N = case['N']
    R = case['R']
    pos = case['positions']
    edge_index = case['edge_index']

    # Generate a fixed source
    mu_source = generate_source_distribution(N, pos, rng, R=R)

    n_cols = len(tau_values)
    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 3, 9))

    row_labels = ['Observation', 'FM Reconstruction', 'True Source']
    model.eval()

    for col, tau in enumerate(tau_values):
        obs = generate_observation(mu_source, R, tau)
        node_ctx = obs[:, None].astype(np.float32)
        global_ctx = np.array([tau], dtype=np.float32)

        # FM posterior mean
        mu_starts = np.array([
            sample_dirichlet_start(N, obs, dirichlet_alpha, start_mode, rng)
            for _ in range(posterior_k)
        ])
        try:
            fm_samples = sample_posterior_film(
                model, mu_starts, node_ctx, global_ctx, edge_index,
                n_steps=100, device=device)
            fm_samples = np.clip(fm_samples, 0, None)
            sums = fm_samples.sum(axis=1, keepdims=True) + 1e-15
            fm_samples = (fm_samples / sums).astype(np.float32)
            fm_mean = fm_samples.mean(axis=0)
            fm_mean /= fm_mean.sum() + 1e-15
        except Exception:
            fm_mean = np.ones(N, dtype=np.float32) / N

        vmax = max(mu_source.max(), obs.max(), fm_mean.max())
        node_size = max(10, min(40, 1500 // max(N, 1)))

        distributions = [obs, fm_mean, mu_source]
        for row_idx, (mu, label) in enumerate(zip(distributions, row_labels)):
            ax = axes[row_idx, col]
            tv_val = total_variation(mu, mu_source) if row_idx < 2 else 0.0
            title_str = f'tau={tau:.1f}'
            if row_idx < 2:
                title_str += f'\nTV={tv_val:.3f}'
            draw_graph_with_dist(ax, R, pos, mu=mu,
                                 title=(title_str if row_idx == 0 else
                                        (f'TV={tv_val:.3f}' if row_idx == 1 else '')),
                                 cmap='hot', node_size=node_size,
                                 vmin=0, vmax=vmax)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)

    fig.suptitle(f'Tau Generalization: {case["name"]} (N={N})',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Ex18: Diffusion Source Recovery')
    parser.add_argument('--tau-train-range', type=float, nargs=2,
                        default=[0.3, 1.5])
    parser.add_argument('--n-pairs-per-graph', type=int, default=20)
    parser.add_argument('--n-starts-per-pair', type=int, default=5)
    parser.add_argument('--n-samples', type=int, default=20000)
    parser.add_argument('--n-epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--posterior-k', type=int, default=20)
    parser.add_argument('--dirichlet-alpha', type=float, default=1.0)
    parser.add_argument('--start-mode', type=str, default='uniform',
                        choices=['uniform', 'obs_centered'],
                        help='Dirichlet start: uniform or obs-centered')
    parser.add_argument('--loss-type', type=str, default='rate_kl')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--direct-epochs', type=int, default=500)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint-suffix', type=str, default='',
                        help='Suffix for checkpoint filenames (e.g. "_75pairs_60k")')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    device = args.device or get_device()
    print(f"Device: {device}", flush=True)

    # -- Generate graphs --
    print("\n=== Experiment 18: Diffusion Source Recovery ===", flush=True)

    print("\nGenerating training graphs...", flush=True)
    train_graphs = generate_training_graphs(seed=42)
    print(f"  {len(train_graphs)} training graphs", flush=True)

    print("Generating test graphs (topology)...", flush=True)
    test_topo = generate_test_graphs_topology(seed=9000)
    print(f"  {len(test_topo)} OOD-topo graphs", flush=True)

    print("Generating test graphs (size)...", flush=True)
    test_size = generate_test_graphs_size(seed=9500)
    print(f"  {len(test_size)} OOD-size graphs", flush=True)

    # -- Build dataset --
    print("\nBuilding diffusion source recovery dataset...", flush=True)
    dataset = DiffusionSourceDataset(
        train_graphs,
        tau_range=tuple(args.tau_train_range),
        n_pairs_per_graph=args.n_pairs_per_graph,
        n_starts_per_pair=args.n_starts_per_pair,
        dirichlet_alpha=args.dirichlet_alpha,
        start_mode=args.start_mode,
        n_samples=args.n_samples,
        seed=42,
    )

    # -- Checkpointing --
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    sfx = args.checkpoint_suffix
    ckpt_path = os.path.join(ckpt_dir, f'ex18_fm_model{sfx}.pt')

    # -- FM Model --
    print("\n--- Training FM model ---", flush=True)
    model = FiLMConditionalGNNRateMatrixPredictor(
        node_context_dim=1,  # [obs(a)]
        global_dim=1,        # [tau]
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    )

    if args.resume and os.path.exists(ckpt_path):
        print(f"  Resuming from {ckpt_path}", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        losses = []
    else:
        result = train_film_conditional(
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

    model.to(device)

    # -- DirectGNN baseline --
    print("\n--- Training DirectGNN baseline ---", flush=True)
    direct_model = DirectGNNPredictor(
        context_dim=2,  # [obs(a), tau_broadcast]
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    )

    # Build training pairs: context=[obs, tau_broadcast], target=mu_source
    direct_train_pairs = []
    for p in dataset.all_pairs:
        N = p['N']
        tau_broadcast = np.full(N, p['tau'], dtype=np.float32)
        ctx = np.stack([p['obs'], tau_broadcast], axis=1)  # (N, 2)
        direct_train_pairs.append((ctx, p['mu_source'], p['edge_index']))

    direct_ckpt = os.path.join(ckpt_dir, f'ex18_direct_model{sfx}.pt')
    if args.resume and os.path.exists(direct_ckpt):
        print(f"  Resuming from {direct_ckpt}", flush=True)
        direct_model.load_state_dict(
            torch.load(direct_ckpt, map_location='cpu'))
    else:
        train_direct_gnn(
            direct_model, direct_train_pairs,
            n_epochs=args.direct_epochs,
            lr=args.lr,
            device=device,
            seed=42,
            ema_decay=0.999,
        )
        torch.save(direct_model.state_dict(), direct_ckpt)
        print(f"  Saved DirectGNN to {direct_ckpt}", flush=True)

    direct_model.to(device)

    # -- Generate test cases --
    print("\nGenerating test cases...", flush=True)

    id_cases = generate_test_cases(
        train_graphs, TAU_IN_RANGE + TAU_SHORT + TAU_LONG,
        n_per_graph=5, seed=7001)
    print(f"  ID: {len(id_cases)} test cases", flush=True)

    topo_cases = generate_test_cases(
        test_topo, TAU_IN_RANGE + TAU_SHORT + TAU_LONG,
        n_per_graph=5, seed=7002)
    print(f"  OOD-topo: {len(topo_cases)} test cases", flush=True)

    size_cases = generate_test_cases(
        test_size, TAU_IN_RANGE + TAU_SHORT + TAU_LONG,
        n_per_graph=5, seed=7003)
    print(f"  OOD-size: {len(size_cases)} test cases", flush=True)

    # -- Evaluate --
    print("\nEvaluating ID...", flush=True)
    id_results = evaluate_source_recovery(
        model, id_cases, device,
        posterior_k=args.posterior_k, dirichlet_alpha=args.dirichlet_alpha,
        start_mode=args.start_mode, direct_model=direct_model)

    print("Evaluating OOD-topo...", flush=True)
    topo_results = evaluate_source_recovery(
        model, topo_cases, device,
        posterior_k=args.posterior_k, dirichlet_alpha=args.dirichlet_alpha,
        start_mode=args.start_mode, direct_model=direct_model)

    print("Evaluating OOD-size...", flush=True)
    size_results = evaluate_source_recovery(
        model, size_cases, device,
        posterior_k=args.posterior_k, dirichlet_alpha=args.dirichlet_alpha,
        start_mode=args.start_mode, direct_model=direct_model)

    # -- Print results --
    print_results_table(id_results, topo_results, size_results)

    # -- Plots --
    out_dir = os.path.dirname(os.path.abspath(__file__))
    print("\nGenerating plots...", flush=True)

    plot_results(
        losses, id_results, topo_results, size_results,
        os.path.join(out_dir, 'ex18_results.png'),
    )

    # Reconstruction gallery from OOD-topo cases
    plot_reconstruction_gallery(
        model, topo_cases + size_cases,
        os.path.join(out_dir, 'ex18_reconstruction_gallery.png'),
        device, posterior_k=args.posterior_k,
        dirichlet_alpha=args.dirichlet_alpha,
        start_mode=args.start_mode,
    )

    # Tau generalization figure from OOD-topo cases
    plot_tau_generalization(
        model, topo_cases,
        os.path.join(out_dir, 'ex18_tau_generalization.png'),
        device, posterior_k=args.posterior_k,
        dirichlet_alpha=args.dirichlet_alpha,
        start_mode=args.start_mode,
    )

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
