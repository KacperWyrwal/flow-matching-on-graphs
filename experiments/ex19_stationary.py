"""
Experiment 19: Stationary Distribution Prediction for Asymmetric Markov Chains.

Predicts the stationary distribution pi of asymmetric Markov chains on graphs.
Flow matching framing: transport uniform (1/N) -> pi.
Uses FlexibleConditionalGNNRateMatrixPredictor with edge-aware message passing:
context_dim=0 (no per-node context), edge_dim=1 (R_ab per directed edge),
hidden_dim=64, n_layers=6.
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
from scipy.stats import spearmanr

from graph_ot_fm import (
    GraphStructure,
    GeodesicCache,
    total_variation,
)
from graph_ot_fm.ot_solver import compute_ot_coupling
from graph_ot_fm.flow import marginal_distribution_fast, marginal_rate_matrix_fast

from meta_fm import (
    FlexibleConditionalGNNRateMatrixPredictor,
    DirectGNNPredictor,
    EdgeAwareDirectGNNPredictor,
    EMA,
    train_flexible_conditional,
    train_direct_gnn,
    get_device,
)
from meta_fm.model import rate_matrix_to_edge_index

from experiments.ex17_ot_generalization import (
    generate_training_graphs,
    generate_test_graphs_topology,
    generate_test_graphs_size,
    draw_graph_with_dist,
    sample_ot_transport,
    compute_exact_interpolation,
)


# -- Asymmetric Rate Matrices ------------------------------------------------

def make_asymmetric_rate_matrix(R_base, asymmetry=1.0, rng=None):
    """Create an asymmetric rate matrix from a symmetric base.

    For each edge (a,b) in the upper triangle of R_base:
        w = rng.uniform(-asymmetry, asymmetry)
        R_asym[a,b] = R_base[a,b] * exp(+w)
        R_asym[b,a] = R_base[a,b] * exp(-w)
    Diagonal = -row_sum.

    Returns: (R_asym, edge_weights) where edge_weights maps (a,b)->w.
    """
    if rng is None:
        rng = np.random.default_rng()
    N = R_base.shape[0]
    R_asym = np.zeros((N, N))
    edge_weights = {}
    for a in range(N):
        for b in range(a + 1, N):
            if abs(R_base[a, b]) > 1e-10:
                w = rng.uniform(-asymmetry, asymmetry)
                R_asym[a, b] = R_base[a, b] * np.exp(+w)
                R_asym[b, a] = R_base[a, b] * np.exp(-w)
                edge_weights[(a, b)] = w
    np.fill_diagonal(R_asym, -R_asym.sum(axis=1))
    return R_asym, edge_weights


def compute_stationary(R):
    """Solve pi R = 0 subject to sum(pi) = 1.

    Uses R^T with last row replaced by ones, b = [0,...,0,1].
    """
    N = R.shape[0]
    A = R.T.copy()
    A[-1, :] = 1.0
    b = np.zeros(N)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.clip(pi, 0, None)
    pi = pi / (pi.sum() + 1e-15)
    return pi.astype(np.float32)


def build_node_asymmetry_features(R):
    """Build per-node asymmetry features from rate matrix R.

    For each node: in_strength, out_strength, net_flow, log_ratio.
    Returns: (N, 4) float32 array.
    """
    N = R.shape[0]
    R_off = R.copy()
    np.fill_diagonal(R_off, 0)
    out_strength = R_off.sum(axis=1)   # row sum (outgoing)
    in_strength = R_off.sum(axis=0)    # col sum (incoming)
    net_flow = in_strength - out_strength
    log_ratio = np.log((in_strength + 1e-8) / (out_strength + 1e-8))
    features = np.stack([in_strength, out_strength, net_flow, log_ratio],
                        axis=1).astype(np.float32)
    return features


# -- Graph Instance Generation -----------------------------------------------

def build_edge_features(R, edge_index):
    """Build per-edge features: just the raw rate R_ab.

    Args:
        R: (N, N) rate matrix
        edge_index: (2, E) LongTensor
    Returns:
        edge_feat: (E, 1) FloatTensor
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    feats = R[src, dst].astype(np.float32)
    return torch.tensor(feats[:, None], dtype=torch.float32)


def build_graph_instances(base_graphs, asymmetry_levels, seed=42):
    """Create asymmetric graph instances from base graphs.

    Returns list of (name, R_asym, R_base, pos, asym) 5-tuples.
    """
    rng = np.random.default_rng(seed)
    instances = []
    for name, R_base, pos in base_graphs:
        for asym in asymmetry_levels:
            R_asym, _ = make_asymmetric_rate_matrix(R_base, asymmetry=asym, rng=rng)
            instances.append((f'{name}_a{asym}', R_asym, R_base, pos, asym))
    return instances


# -- Dataset ------------------------------------------------------------------

class StationaryDistDataset(torch.utils.data.Dataset):
    """Training data for stationary distribution prediction.

    Supports two modes:
    - 'uniform': single start from 1/N (classic OT coupling)
    - 'dirichlet': multiple Dirichlet-random starts per graph for
      epistemic uncertainty quantification at test time

    Returns 7-tuple for train_flexible_conditional:
        (mu_tau, tau, node_ctx, u_tilde, edge_index, edge_feat, N)

    node_ctx is (N, 0) — empty.  All asymmetry info is in edge_feat (E, 1).
    """

    def __init__(self, graph_instances, mode='dirichlet',
                 dirichlet_alpha=1.0, n_starts_per_graph=5,
                 transport_graph='undirected',
                 n_samples=20000, seed=42):
        """
        graph_instances: list of (name, R_asym, R_base, pos, asym) 5-tuples.
        transport_graph: 'undirected' (use R_base for OT) or 'directed' (use R_asym)
        """
        rng = np.random.default_rng(seed)
        all_items = []
        self.all_pairs = []

        for name, R_asym, R_base, pos, asymmetry in graph_instances:
            N = R_asym.shape[0]

            # Compute stationary distribution
            pi = compute_stationary(R_asym)

            # OT transport on undirected or directed graph
            R_transport = R_base if transport_graph == 'undirected' else R_asym
            graph_struct = GraphStructure(R_transport)
            geo_cache = GeodesicCache(graph_struct)

            # Edge index from base graph (all edges in both directions)
            edge_index = rate_matrix_to_edge_index(R_base)

            # Edge features from asymmetric R
            edge_feat = build_edge_features(R_asym, edge_index)  # (E, 1)

            # Empty node context (context_dim=0)
            node_ctx = np.zeros((N, 0), dtype=np.float32)

            self.all_pairs.append({
                'name': name,
                'N': N,
                'R_asym': R_asym,
                'R_base': R_base,
                'positions': pos,
                'edge_index': edge_index,
                'edge_feat': edge_feat,
                'pi': pi,
                'node_ctx': node_ctx,
                'asymmetry': asymmetry,
            })

            # Multiple starts for epistemic uncertainty
            n_starts = n_starts_per_graph if mode == 'dirichlet' else 1
            n_per = max(1, n_samples // (len(graph_instances) * n_starts))

            for _ in range(n_starts):
                if mode == 'dirichlet':
                    mu_start = rng.dirichlet(
                        np.full(N, dirichlet_alpha)).astype(np.float32)
                else:
                    mu_start = (np.ones(N) / N).astype(np.float32)

                coupling = compute_ot_coupling(mu_start, pi, graph_struct=graph_struct)
                geo_cache.precompute_for_coupling(coupling)

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
                        edge_index,
                        edge_feat,
                        N,
                    ))

        idx = rng.permutation(len(all_items))
        self.samples = [all_items[i] for i in idx[:n_samples]]
        print(f"  Dataset: {len(self.samples)} samples from "
              f"{len(self.all_pairs)} instances ({mode} mode, "
              f"{n_starts_per_graph if mode == 'dirichlet' else 1} starts/graph, "
              f"transport={transport_graph})", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# -- Baselines ----------------------------------------------------------------

def baseline_degree_proportional(R):
    """Estimate pi proportional to in-strength (column sums of off-diag R)."""
    R_off = R.copy()
    np.fill_diagonal(R_off, 0)
    in_strength = R_off.sum(axis=0)
    pi_hat = in_strength / (in_strength.sum() + 1e-15)
    return pi_hat.astype(np.float32)


def baseline_power_iteration(R, n_steps=1000, dt=0.01):
    """Estimate pi via power iteration: pi = pi @ expm(dt * R)."""
    N = R.shape[0]
    pi = np.ones(N) / N
    P = expm(dt * R)
    for _ in range(n_steps):
        pi = pi @ P
        pi = np.clip(pi, 0, None)
        pi /= pi.sum() + 1e-15
    return pi.astype(np.float32)


# -- Edge-Aware ODE Sampler ---------------------------------------------------

def sample_ot_transport_edge(model, mu_start, node_ctx, edge_index,
                             edge_feat, n_steps=100, device=None):
    """ODE integration for FlexibleConditionalGNNRateMatrixPredictor with
    optional edge features.  Returns final distribution as numpy array."""
    if device is None:
        device = get_device()
    model.eval()
    N = len(mu_start)
    dt = 0.999 / n_steps

    ctx = torch.tensor(node_ctx, dtype=torch.float32, device=device)
    ei = edge_index.to(device)
    ef = edge_feat.to(device) if edge_feat is not None else None
    mu = mu_start.copy().astype(float)

    with torch.no_grad():
        for step in range(n_steps):
            t = step * dt
            mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
            t_t = torch.tensor([t], dtype=torch.float32, device=device)

            R_pred = model.forward_single(mu_t, t_t, ctx, ei, edge_feat=ef)
            R_np = R_pred.cpu().numpy() / (1.0 - t + 1e-10)

            dp = mu @ R_np
            mu = mu + dt * dp
            mu = np.clip(mu, 0.0, None)
            s = mu.sum()
            if s > 1e-15:
                mu /= s

    return mu


# -- Evaluation ---------------------------------------------------------------

def evaluate_stationary(model, instances, device, direct_model=None,
                        posterior_k=20, dirichlet_alpha=1.0):
    """Evaluate FM model and baselines on stationary distribution prediction.

    For the FM model, runs K Dirichlet-start ODE integrations to produce
    posterior samples.  Metrics are computed on the posterior mean.

    instances: list of (name, R_asym, R_base, pos, asym) 5-tuples.
    Returns dict with per-instance metrics.
    """
    results = {
        'fm_tv': [], 'fm_kl': [], 'fm_spearman': [], 'fm_topk': [],
        'direct_tv': [], 'direct_kl': [], 'direct_spearman': [], 'direct_topk': [],
        'degree_tv': [], 'degree_kl': [], 'degree_spearman': [], 'degree_topk': [],
        'power_tv': [], 'power_kl': [], 'power_spearman': [], 'power_topk': [],
        'names': [], 'sizes': [], 'asymmetries': [],
        # Posterior / calibration metrics (FM only)
        'fm_posterior_std': [],
        'fm_posterior_err': [],
        'fm_calibration_r': [],
        'fm_diversity': [],
    }

    model.eval()
    if direct_model is not None:
        direct_model.eval()

    rng = np.random.default_rng(12345)

    for idx, (name, R_asym, R_base, pos, asymmetry) in enumerate(instances):
        N = R_asym.shape[0]
        pi_true = compute_stationary(R_asym)
        edge_index = rate_matrix_to_edge_index(R_base)
        edge_feat = build_edge_features(R_asym, edge_index)
        node_ctx = np.zeros((N, 0), dtype=np.float32)
        k = min(5, N)

        results['names'].append(name)
        results['sizes'].append(N)
        results['asymmetries'].append(asymmetry)

        # Helper to compute metrics
        def _metrics(pi_hat, prefix):
            tv = total_variation(pi_hat, pi_true)
            results[f'{prefix}_tv'].append(tv)
            kl = float(np.sum(
                pi_true * np.log((pi_true + 1e-10) / (pi_hat + 1e-10))))
            results[f'{prefix}_kl'].append(kl)
            rho, _ = spearmanr(pi_true, pi_hat)
            results[f'{prefix}_spearman'].append(
                rho if not np.isnan(rho) else 0.0)
            true_topk = set(np.argsort(pi_true)[-k:])
            pred_topk = set(np.argsort(pi_hat)[-k:])
            results[f'{prefix}_topk'].append(len(true_topk & pred_topk) / k)

        # FM: K posterior samples from Dirichlet starts
        fm_samples = []
        for _ in range(posterior_k):
            mu_start = rng.dirichlet(
                np.full(N, dirichlet_alpha)).astype(np.float32)
            try:
                pi_s = sample_ot_transport_edge(
                    model, mu_start, node_ctx, edge_index, edge_feat,
                    n_steps=100, device=device)
                pi_s = np.clip(pi_s, 0, None).astype(np.float32)
                pi_s /= pi_s.sum() + 1e-15
            except Exception:
                pi_s = np.ones(N, dtype=np.float32) / N
            fm_samples.append(pi_s)

        fm_samples = np.array(fm_samples)  # (K, N)
        pi_fm_mean = fm_samples.mean(axis=0)
        pi_fm_mean /= pi_fm_mean.sum() + 1e-15
        pi_fm_std = fm_samples.std(axis=0)   # (N,)
        pi_fm_err = np.abs(pi_fm_mean - pi_true)  # (N,)

        _metrics(pi_fm_mean, 'fm')

        # Calibration: Pearson r between posterior std and |error|
        from scipy.stats import pearsonr
        if pi_fm_std.std() > 1e-12 and pi_fm_err.std() > 1e-12:
            cal_r, _ = pearsonr(pi_fm_std, pi_fm_err)
            cal_r = cal_r if not np.isnan(cal_r) else 0.0
        else:
            cal_r = 0.0
        results['fm_calibration_r'].append(cal_r)
        results['fm_posterior_std'].append(pi_fm_std)
        results['fm_posterior_err'].append(pi_fm_err)

        # Diversity: mean pairwise TV among K samples
        pair_tvs = []
        for i in range(min(posterior_k, 10)):
            for j in range(i + 1, min(posterior_k, 10)):
                pair_tvs.append(total_variation(fm_samples[i], fm_samples[j]))
        results['fm_diversity'].append(
            np.mean(pair_tvs) if pair_tvs else 0.0)

        # DirectGNN prediction (EdgeAwareDirectGNNPredictor)
        if direct_model is not None:
            try:
                ei_dev = edge_index.to(device)
                ef_dev = edge_feat.to(device)
                with torch.no_grad():
                    pi_direct = direct_model(ei_dev, ef_dev, N).cpu().numpy()
            except Exception as e:
                print(f"  DirectGNN failed on {name}: {e}", flush=True)
                pi_direct = np.ones(N, dtype=np.float32) / N
        else:
            pi_direct = np.ones(N, dtype=np.float32) / N
        _metrics(pi_direct, 'direct')

        # Degree-proportional baseline
        pi_deg = baseline_degree_proportional(R_asym)
        _metrics(pi_deg, 'degree')

        # Power iteration baseline
        try:
            pi_pow = baseline_power_iteration(R_asym)
        except Exception:
            pi_pow = np.ones(N, dtype=np.float32) / N
        _metrics(pi_pow, 'power')

        if (idx + 1) % 10 == 0:
            print(f"    Evaluated {idx+1}/{len(instances)}", flush=True)

    return results


# -- Console Output -----------------------------------------------------------

def print_results_table(id_res, topo_res, size_res, asym_res):
    """Print formatted TV results table."""
    print("\nTV Results:", flush=True)
    header = (f"{'':22s} {'ID':>10s}  {'OOD-topo':>10s}  "
              f"{'OOD-size':>10s}  {'OOD-asym':>10s}")
    print(header, flush=True)

    for meth, key in [('FM', 'fm_tv'),
                      ('DirectGNN', 'direct_tv'),
                      ('Degree-prop', 'degree_tv'),
                      ('Power iteration', 'power_tv')]:
        id_v = np.nanmean(id_res[key])
        topo_v = np.nanmean(topo_res[key])
        size_v = np.nanmean(size_res[key])
        asym_v = np.nanmean(asym_res[key])
        print(f"  {meth:20s} {id_v:10.4f}  {topo_v:10.4f}  "
              f"{size_v:10.4f}  {asym_v:10.4f}", flush=True)

    # TV by asymmetry level
    all_asym = (id_res['asymmetries'] + topo_res['asymmetries']
                + size_res['asymmetries'] + asym_res['asymmetries'])
    all_fm_tv = (id_res['fm_tv'] + topo_res['fm_tv']
                 + size_res['fm_tv'] + asym_res['fm_tv'])
    all_deg_tv = (id_res['degree_tv'] + topo_res['degree_tv']
                  + size_res['degree_tv'] + asym_res['degree_tv'])
    all_dir_tv = (id_res['direct_tv'] + topo_res['direct_tv']
                  + size_res['direct_tv'] + asym_res['direct_tv'])

    asym_levels_all = sorted(set(all_asym))
    print("\nTV by asymmetry level:", flush=True)
    print(f"  {'asym':>8s}  {'FM':>8s}  {'DirectGNN':>10s}  "
          f"{'Degree-prop':>12s}", flush=True)
    for a in asym_levels_all:
        fm_vals = [v for aa, v in zip(all_asym, all_fm_tv) if aa == a]
        deg_vals = [v for aa, v in zip(all_asym, all_deg_tv) if aa == a]
        dir_vals = [v for aa, v in zip(all_asym, all_dir_tv) if aa == a]
        if fm_vals:
            print(f"  {a:8.2f}  {np.mean(fm_vals):8.4f}  "
                  f"{np.mean(dir_vals):10.4f}  {np.mean(deg_vals):12.4f}",
                  flush=True)

    # Rank correlation summary
    all_fm_rho = (id_res['fm_spearman'] + topo_res['fm_spearman']
                  + size_res['fm_spearman'] + asym_res['fm_spearman'])
    all_dir_rho = (id_res['direct_spearman'] + topo_res['direct_spearman']
                   + size_res['direct_spearman'] + asym_res['direct_spearman'])
    all_deg_rho = (id_res['degree_spearman'] + topo_res['degree_spearman']
                   + size_res['degree_spearman'] + asym_res['degree_spearman'])
    print(f"\nSpearman rank correlation (mean):", flush=True)
    print(f"  FM: {np.nanmean(all_fm_rho):.4f}, "
          f"DirectGNN: {np.nanmean(all_dir_rho):.4f}, "
          f"Degree-prop: {np.nanmean(all_deg_rho):.4f}", flush=True)

    # Calibration and diversity
    print(f"\nCalibration (Pearson r between posterior std and |error|):", flush=True)
    for label, res in [('ID', id_res), ('OOD-topo', topo_res),
                        ('OOD-size', size_res), ('OOD-asym', asym_res)]:
        cal = np.nanmean(res['fm_calibration_r'])
        div = np.nanmean(res['fm_diversity'])
        print(f"  {label:12s}  calibration r={cal:.4f}, diversity={div:.4f}",
              flush=True)


# -- Plotting -----------------------------------------------------------------

def plot_results_stationary(losses, id_res, topo_res, size_res, asym_res,
                            instances_topo, instances_size,
                            out_path, device, model, direct_model):
    """Create the 2x3 results figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel A: Training loss (semilogy)
    ax = axes[0, 0]
    if losses:
        ax.plot(losses, 'b-', alpha=0.7, linewidth=0.8)
        ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('(A) Training Loss')
    ax.grid(True, alpha=0.3)

    # Panel B: TV bars grouped by test condition
    ax = axes[0, 1]
    methods = ['FM', 'DirectGNN', 'Degree-prop']
    splits = ['ID', 'OOD-topo', 'OOD-size', 'OOD-asym']
    all_results = [id_res, topo_res, size_res, asym_res]
    x = np.arange(len(splits))
    width = 0.25
    for i, (meth, key) in enumerate(zip(methods,
                                        ['fm_tv', 'direct_tv', 'degree_tv'])):
        vals = [np.nanmean(r[key]) for r in all_results]
        ax.bar(x + i * width, vals, width, label=meth, alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(splits)
    ax.set_ylabel('TV Distance')
    ax.set_title('(B) TV by Test Condition')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: TV vs asymmetry level
    ax = axes[0, 2]
    all_asym = (id_res['asymmetries'] + topo_res['asymmetries']
                + size_res['asymmetries'] + asym_res['asymmetries'])
    all_fm_tv = (id_res['fm_tv'] + topo_res['fm_tv']
                 + size_res['fm_tv'] + asym_res['fm_tv'])
    all_dir_tv = (id_res['direct_tv'] + topo_res['direct_tv']
                  + size_res['direct_tv'] + asym_res['direct_tv'])
    all_deg_tv = (id_res['degree_tv'] + topo_res['degree_tv']
                  + size_res['degree_tv'] + asym_res['degree_tv'])

    asym_levels_all = sorted(set(all_asym))
    for label, all_tv, color, marker in [
        ('FM', all_fm_tv, 'tab:blue', 'o'),
        ('DirectGNN', all_dir_tv, 'tab:orange', '^'),
        ('Degree-prop', all_deg_tv, 'tab:green', 's'),
    ]:
        means = []
        stds = []
        for a in asym_levels_all:
            vals = [v for aa, v in zip(all_asym, all_tv) if aa == a]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0.0)
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(asym_levels_all, means, f'{marker}-', color=color, label=label,
                markersize=5)
        ax.fill_between(asym_levels_all, means - stds, means + stds,
                        alpha=0.15, color=color)
    ax.set_xlabel('Asymmetry Level')
    ax.set_ylabel('TV Distance')
    ax.set_title('(C) TV vs Asymmetry')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel D: Prediction gallery -- 3 OOD graphs x 4 cols
    ax = axes[1, 0]
    ax.axis('off')
    # Pick up to 3 diverse OOD instances
    gallery_instances = []
    seen_base = set()
    for inst in (instances_topo + instances_size):
        base = inst[0].rsplit('_a', 1)[0]
        if base not in seen_base and len(gallery_instances) < 3:
            gallery_instances.append(inst)
            seen_base.add(base)

    if gallery_instances:
        n_gallery = len(gallery_instances)
        # Create inset axes in Panel D region
        bbox = ax.get_position()
        for g_idx, (name, R_asym, R_base, pos, asymmetry) in enumerate(gallery_instances):
            N = R_asym.shape[0]
            pi_true = compute_stationary(R_asym)
            mu_uniform = np.ones(N, dtype=np.float32) / N
            edge_index = rate_matrix_to_edge_index(R_base)
            edge_feat = build_edge_features(R_asym, edge_index)
            node_ctx = np.zeros((N, 0), dtype=np.float32)

            # Posterior mean from 5 Dirichlet starts
            gallery_rng = np.random.default_rng(g_idx * 100)
            fm_samps = []
            for _ in range(5):
                mu_s = gallery_rng.dirichlet(np.ones(N)).astype(np.float32)
                try:
                    pi_s = sample_ot_transport_edge(
                        model, mu_s, node_ctx, edge_index, edge_feat,
                        n_steps=100, device=device)
                    pi_s = np.clip(pi_s, 0, None).astype(np.float32)
                    pi_s /= pi_s.sum() + 1e-15
                    fm_samps.append(pi_s)
                except Exception:
                    pass
            if fm_samps:
                pi_fm = np.mean(fm_samps, axis=0)
                pi_fm /= pi_fm.sum() + 1e-15
            else:
                pi_fm = mu_uniform.copy()

            if direct_model is not None:
                try:
                    ei_dev = edge_index.to(device)
                    ef_dev = edge_feat.to(device)
                    with torch.no_grad():
                        pi_direct = direct_model(
                            ei_dev, ef_dev, N).cpu().numpy()
                except Exception:
                    pi_direct = mu_uniform.copy()
            else:
                pi_direct = mu_uniform.copy()

            pi_deg = baseline_degree_proportional(R_asym)

            preds = [
                (pi_true, 'True pi'),
                (pi_fm, 'FM'),
                (pi_direct, 'DirectGNN'),
                (pi_deg, 'Degree-prop'),
            ]
            for c_idx, (mu, label) in enumerate(preds):
                inset_ax = fig.add_axes([
                    bbox.x0 + c_idx * bbox.width / 4,
                    bbox.y0 + (n_gallery - 1 - g_idx) * bbox.height / n_gallery,
                    bbox.width / 4 - 0.005,
                    bbox.height / n_gallery - 0.005,
                ])
                node_size = max(8, min(30, 800 // max(N, 1)))
                draw_graph_with_dist(inset_ax, R_base, pos, mu=mu,
                                     title=(label if g_idx == 0 else ''),
                                     cmap='hot', node_size=node_size)
                if c_idx == 0:
                    inset_ax.set_ylabel(name[:20], fontsize=5)

    # Panel E: TV vs graph size scatter
    ax = axes[1, 1]
    all_sizes = (id_res['sizes'] + topo_res['sizes']
                 + size_res['sizes'] + asym_res['sizes'])
    all_fm = (id_res['fm_tv'] + topo_res['fm_tv']
              + size_res['fm_tv'] + asym_res['fm_tv'])
    all_dir = (id_res['direct_tv'] + topo_res['direct_tv']
               + size_res['direct_tv'] + asym_res['direct_tv'])
    all_deg = (id_res['degree_tv'] + topo_res['degree_tv']
               + size_res['degree_tv'] + asym_res['degree_tv'])
    ax.scatter(all_sizes, all_fm, alpha=0.5, s=15, label='FM', c='tab:blue')
    ax.scatter(all_sizes, all_dir, alpha=0.5, s=15, label='DirectGNN',
               c='tab:orange', marker='^')
    ax.scatter(all_sizes, all_deg, alpha=0.5, s=15, label='Degree-prop',
               c='tab:green', marker='s')
    ax.set_xlabel('Graph Size (N)')
    ax.set_ylabel('TV Distance')
    ax.set_title('(E) TV vs Graph Size')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel F: Calibration scatter + diversity box plot
    ax = axes[1, 2]
    # Calibration: scatter posterior std vs |error| across all test cases
    colors_map = {'ID': 'tab:blue', 'OOD-topo': 'tab:red',
                  'OOD-size': 'tab:orange', 'OOD-asym': 'tab:purple'}
    for label, res, color in [
        ('ID', id_res, 'tab:blue'),
        ('OOD-topo', topo_res, 'tab:red'),
        ('OOD-size', size_res, 'tab:orange'),
        ('OOD-asym', asym_res, 'tab:purple'),
    ]:
        # Concatenate all per-node stds and errors
        all_std = np.concatenate(res['fm_posterior_std']) if res['fm_posterior_std'] else np.array([])
        all_err = np.concatenate(res['fm_posterior_err']) if res['fm_posterior_err'] else np.array([])
        if len(all_std) > 0:
            # Subsample for readability
            n_pts = min(500, len(all_std))
            idx_sub = np.random.default_rng(0).choice(len(all_std), n_pts, replace=False)
            ax.scatter(all_std[idx_sub], all_err[idx_sub], s=3, alpha=0.3,
                       c=color, label=label)
    # Overall Pearson r
    all_cal = (id_res['fm_calibration_r'] + topo_res['fm_calibration_r']
               + size_res['fm_calibration_r'] + asym_res['fm_calibration_r'])
    mean_cal = np.nanmean(all_cal)
    ax.set_xlabel('Posterior Std')
    ax.set_ylabel('|Posterior Mean - True pi|')
    ax.set_title(f'(F) Calibration (r={mean_cal:.3f})')
    ax.legend(fontsize=6, markerscale=3)
    ax.grid(True, alpha=0.3)

    # Inset: diversity box plot
    inset = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
    div_data = [id_res['fm_diversity'], topo_res['fm_diversity'],
                size_res['fm_diversity'], asym_res['fm_diversity']]
    bp = inset.boxplot(div_data, labels=['ID', 'Topo', 'Size', 'Asym'],
                       widths=0.5, patch_artist=True)
    for patch, c in zip(bp['boxes'],
                        ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple']):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    inset.set_ylabel('Diversity', fontsize=6)
    inset.set_title('Posterior Diversity', fontsize=6)
    inset.tick_params(labelsize=5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Ex19: Stationary Distribution Prediction')
    parser.add_argument('--asymmetry-levels', type=float, nargs='+',
                        default=[0.5, 1.0, 2.0])
    parser.add_argument('--test-asymmetry', type=float, nargs='+',
                        default=[0.25, 3.0])
    parser.add_argument('--n-samples', type=int, default=20000)
    parser.add_argument('--n-epochs', type=int, default=1000)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--loss-type', type=str, default='rate_kl')
    parser.add_argument('--mode', type=str, default='dirichlet',
                        choices=['uniform', 'dirichlet'])
    parser.add_argument('--dirichlet-alpha', type=float, default=1.0)
    parser.add_argument('--n-starts-per-graph', type=int, default=5,
                        help='Dirichlet starts per graph during training')
    parser.add_argument('--posterior-k', type=int, default=20,
                        help='Number of posterior samples at test time')
    parser.add_argument('--transport-graph', type=str, default='undirected',
                        choices=['directed', 'undirected'],
                        help='Use directed or undirected graph for OT transport')
    parser.add_argument('--edge-dim', type=int, default=1,
                        help='Dimension of edge features (0 to disable)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--direct-epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()

    device = args.device or get_device()
    print(f"Device: {device}", flush=True)

    # -- Generate base graphs --
    print("\n=== Experiment 19: Stationary Distribution Prediction ===",
          flush=True)

    print("\nGenerating base graphs...", flush=True)
    train_base = generate_training_graphs(seed=42)
    print(f"  {len(train_base)} training base graphs", flush=True)

    test_topo_base = generate_test_graphs_topology(seed=9000)
    print(f"  {len(test_topo_base)} OOD-topo base graphs", flush=True)

    test_size_base = generate_test_graphs_size(seed=9500)
    print(f"  {len(test_size_base)} OOD-size base graphs", flush=True)

    # -- Build asymmetric instances --
    print("\nBuilding asymmetric instances...", flush=True)
    train_instances = build_graph_instances(
        train_base, args.asymmetry_levels, seed=42)
    print(f"  Train: {len(train_instances)} instances", flush=True)

    test_topo_instances = build_graph_instances(
        test_topo_base, args.asymmetry_levels, seed=43)
    print(f"  OOD-topo: {len(test_topo_instances)} instances", flush=True)

    test_size_instances = build_graph_instances(
        test_size_base, args.asymmetry_levels, seed=44)
    print(f"  OOD-size: {len(test_size_instances)} instances", flush=True)

    test_asym_instances = build_graph_instances(
        train_base, args.test_asymmetry, seed=45)
    print(f"  OOD-asym: {len(test_asym_instances)} instances", flush=True)

    # -- Build dataset --
    print("\nBuilding stationary distribution dataset...", flush=True)
    dataset = StationaryDistDataset(
        train_instances,
        mode=args.mode,
        dirichlet_alpha=args.dirichlet_alpha,
        n_starts_per_graph=args.n_starts_per_graph,
        transport_graph=args.transport_graph,
        n_samples=args.n_samples,
        seed=42,
    )

    # -- Checkpointing --
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'ex19_fm_model.pt')

    # -- FM Model (edge-aware, no per-node context) --
    print("\n--- Training FM model ---", flush=True)
    model = FlexibleConditionalGNNRateMatrixPredictor(
        context_dim=0,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        edge_dim=args.edge_dim,
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

    # -- Edge-aware DirectGNN baseline --
    print("\n--- Training EdgeAwareDirectGNN baseline ---", flush=True)
    direct_model = EdgeAwareDirectGNNPredictor(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        edge_dim=args.edge_dim,
    )

    direct_ckpt = os.path.join(ckpt_dir, 'ex19_direct_model.pt')
    if args.resume and os.path.exists(direct_ckpt):
        print(f"  Resuming from {direct_ckpt}", flush=True)
        direct_model.load_state_dict(
            torch.load(direct_ckpt, map_location='cpu'))
    else:
        # Train EdgeAwareDirectGNN on (graph, pi) pairs
        direct_model.to(device)
        direct_optimizer = torch.optim.Adam(direct_model.parameters(), lr=args.lr)
        direct_ema = EMA(direct_model, decay=0.999)
        for epoch in range(args.direct_epochs):
            ep_loss = 0.0
            idx_perm = np.random.permutation(len(dataset.all_pairs))
            for i in idx_perm:
                p = dataset.all_pairs[i]
                pi_t = torch.tensor(p['pi'], dtype=torch.float32, device=device)
                ei = p['edge_index'].to(device)
                ef = p['edge_feat'].to(device)
                N = p['N']
                pi_pred = direct_model(ei, ef, N)
                loss = (pi_t * (pi_t.clamp(min=1e-10).log()
                                - pi_pred.clamp(min=1e-10).log())).sum()
                direct_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(direct_model.parameters(), 1.0)
                direct_optimizer.step()
                direct_ema.update(direct_model)
                ep_loss += loss.item()
            if (epoch + 1) % 50 == 0:
                print(f"  DirectGNN Epoch {epoch+1}/{args.direct_epochs} | "
                      f"Loss: {ep_loss/len(dataset.all_pairs):.6f}", flush=True)
        direct_ema.apply(direct_model)
        torch.save(direct_model.state_dict(), direct_ckpt)
        print(f"  Saved DirectGNN to {direct_ckpt}", flush=True)

    # -- Evaluate --
    model.to(device)
    direct_model.to(device)

    print("\nEvaluating ID...", flush=True)
    id_results = evaluate_stationary(
        model, train_instances, device, direct_model=direct_model,
        posterior_k=args.posterior_k, dirichlet_alpha=args.dirichlet_alpha)

    print("Evaluating OOD-topo...", flush=True)
    topo_results = evaluate_stationary(
        model, test_topo_instances, device, direct_model=direct_model,
        posterior_k=args.posterior_k, dirichlet_alpha=args.dirichlet_alpha)

    print("Evaluating OOD-size...", flush=True)
    size_results = evaluate_stationary(
        model, test_size_instances, device, direct_model=direct_model,
        posterior_k=args.posterior_k, dirichlet_alpha=args.dirichlet_alpha)

    print("Evaluating OOD-asym...", flush=True)
    asym_results = evaluate_stationary(
        model, test_asym_instances, device, direct_model=direct_model,
        posterior_k=args.posterior_k, dirichlet_alpha=args.dirichlet_alpha)

    # -- Print results --
    print_results_table(id_results, topo_results, size_results, asym_results)

    # -- Plots --
    out_dir = os.path.dirname(os.path.abspath(__file__))
    print("\nGenerating plots...", flush=True)
    plot_results_stationary(
        losses, id_results, topo_results, size_results, asym_results,
        test_topo_instances, test_size_instances,
        os.path.join(out_dir, 'ex19_results.png'),
        device, model, direct_model,
    )

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
