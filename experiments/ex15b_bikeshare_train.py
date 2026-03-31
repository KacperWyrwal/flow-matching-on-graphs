"""
Experiment 15b: Citi Bike NYC — Model Training and Evaluation

Two tasks:
  interpolation  — given sparse station observations, reconstruct full distribution
  forecasting    — given current hour distribution, predict next hour

Baselines:
  Interpolation: k-NN fill, harmonic extension, historical mean
  Forecasting:   persistence (μ_t = μ_{t+1}), historical mean, linear extrapolation

Run: uv run experiments/ex15b_bikeshare_train.py --task both
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
from scipy.spatial.distance import cdist

from graph_ot_fm import (
    GraphStructure,
    compute_ot_coupling,
    GeodesicCache,
    marginal_distribution_fast,
    marginal_rate_matrix_fast,
    total_variation,
)
from meta_fm import (
    FiLMConditionalGNNRateMatrixPredictor,
    train_film_conditional,
    sample_trajectory_film,
    get_device,
    EMA,
)
from meta_fm.model import rate_matrix_to_edge_index

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
CKPT_DIR = os.path.join(ROOT, 'checkpoints')
DATA_DIR = os.path.join(ROOT, 'data')


# ── Time encoding ─────────────────────────────────────────────────────────────

def time_features(hour, dow):
    """Cyclical encoding of hour-of-day and day-of-week → (4,) vector."""
    return np.array([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow  /  7),
        np.cos(2 * np.pi * dow  /  7),
    ], dtype=np.float32)


# ── Context building ──────────────────────────────────────────────────────────

def build_interp_context(dist, obs_mask, positions_norm):
    """
    Build context for interpolation task.

    obs_mask: (N,) bool — which stations are observed
    dist:     (N,) true distribution (only obs_mask entries are used)

    Returns:
        node_context: (N, 3) — [obs_val * is_obs, is_obs, pos_dist_to_nearest_obs]
        obs_vals:     (N,) — observed values, 0 for masked
        mean_obs:     float — mean of observed values
    """
    N = len(dist)
    obs_vals   = np.where(obs_mask, dist, 0.0).astype(np.float32)
    is_obs     = obs_mask.astype(np.float32)
    mean_obs   = float(obs_vals[obs_mask].mean()) if obs_mask.any() else 0.0

    # Distance to nearest observed station (normalised by max)
    if obs_mask.any():
        obs_pos = positions_norm[obs_mask]
        dists   = cdist(positions_norm, obs_pos).min(axis=1)
        dists   = (dists / (dists.max() + 1e-8)).astype(np.float32)
    else:
        dists = np.zeros(N, dtype=np.float32)

    node_context = np.stack([obs_vals * is_obs, is_obs, dists], axis=-1)
    return node_context, obs_vals, mean_obs


def build_interp_global(hour, dow, mean_obs):
    """Global FiLM context for interpolation: time features + mean observation."""
    tf = time_features(hour, dow)
    return np.concatenate([tf, [mean_obs]]).astype(np.float32)   # (5,)


def build_forecast_context(dist, positions_norm):
    """
    Build context for forecasting task.

    node_context: (N, 2) — [current distribution value, 1.0]
    """
    node_context = np.stack([dist.astype(np.float32),
                              np.ones(len(dist), dtype=np.float32)], axis=-1)
    return node_context


def build_forecast_global(hour, dow):
    """Global FiLM context for forecasting: time features only → (4,)."""
    return time_features(hour, dow)


# ── Baselines ─────────────────────────────────────────────────────────────────

def baseline_knn_interp(dist, obs_mask, adj, k=3):
    """
    Fill each unobserved station with the weighted average of its observed
    k-hop neighbors in the trip graph.
    """
    N    = len(dist)
    pred = dist.copy()
    unobs = np.where(~obs_mask)[0]

    # Adjacency-based k-hop reachability
    reach = (adj > 0).astype(float)
    reach_k = reach.copy()
    for _ in range(k - 1):
        reach_k = reach_k @ reach
    reach_k = np.clip(reach_k, 0, 1)

    for i in unobs:
        nbr_weights = reach_k[i] * obs_mask.astype(float)
        s = nbr_weights.sum()
        if s > 1e-12:
            pred[i] = (nbr_weights * dist).sum() / s
        else:
            # Fallback: use mean of observed
            pred[i] = dist[obs_mask].mean() if obs_mask.any() else 1.0 / N

    pred = np.clip(pred, 0, None)
    s = pred.sum()
    return pred / s if s > 1e-12 else np.ones(N) / N


def baseline_harmonic(dist, obs_mask, adj):
    """
    Harmonic extension: solve Laplacian equations for unobserved nodes.
    Minimizes sum of squared differences between neighbors.
    """
    N    = len(dist)
    obs  = np.where(obs_mask)[0]
    unob = np.where(~obs_mask)[0]
    n_u  = len(unob)

    if n_u == 0:
        return dist.copy()
    if len(obs) == 0:
        return np.ones(N) / N

    # Degree matrix and Laplacian for unobserved nodes
    deg   = adj.sum(axis=1)
    # L[i, j] = -adj[i,j] for i != j; L[i,i] = deg[i]
    # For unobserved nodes: L_uu * f_u = -L_uo * f_o
    idx_map = {j: k for k, j in enumerate(unob)}

    # Build L_uu (n_u × n_u) and b = -L_uo * f_o (n_u,)
    L_uu = np.zeros((n_u, n_u))
    b    = np.zeros(n_u)
    for ki, i in enumerate(unob):
        L_uu[ki, ki] = deg[i]
        for j in range(N):
            if adj[i, j] > 0:
                if j in idx_map:
                    L_uu[ki, idx_map[j]] -= adj[i, j]
                elif obs_mask[j]:
                    b[ki] += adj[i, j] * dist[j]

    # Regularise and solve
    L_uu += 1e-8 * np.eye(n_u)
    try:
        f_u = np.linalg.solve(L_uu, b)
    except np.linalg.LinAlgError:
        f_u = np.linalg.lstsq(L_uu, b, rcond=None)[0]

    pred = dist.copy()
    for ki, i in enumerate(unob):
        pred[i] = float(f_u[ki])
    pred = np.clip(pred, 0, None)
    s = pred.sum()
    return pred / s if s > 1e-12 else np.ones(N) / N


def baseline_hist_mean(hist_mean, hour, dow):
    """Return historical mean distribution for this (hour, dow)."""
    return hist_mean[hour, dow].copy()


def baseline_persistence(dist_t):
    """Predict μ_{t+1} = μ_t."""
    return dist_t.copy()


def baseline_linear_extrap(dist_t, dist_t_prev):
    """Predict μ_{t+1} = 2*μ_t - μ_{t-1} (linear extrapolation)."""
    pred = 2 * dist_t - dist_t_prev
    pred = np.clip(pred, 0, None)
    s = pred.sum()
    return pred / s if s > 1e-12 else np.ones(len(dist_t)) / len(dist_t)


# ── Interpolation dataset ─────────────────────────────────────────────────────

class BikeInterpolationDataset(torch.utils.data.Dataset):
    """
    For each training snapshot, randomly mask 1-obs_frac of stations.
    Flow: Dirichlet → true distribution, conditioned on masked observations.
    """

    def __init__(self, R, train_dists, train_hours, train_dows,
                 positions_norm, obs_frac=0.3,
                 n_starts_per_snap=3, n_samples=20000, seed=42):
        rng          = np.random.default_rng(seed)
        N            = R.shape[0]
        graph_struct = GraphStructure(R)
        cache        = GeodesicCache(graph_struct)
        self._edge_index = rate_matrix_to_edge_index(R)

        triples = []
        for dist, hour, dow in zip(train_dists, train_hours, train_dows):
            for _ in range(n_starts_per_snap):
                # Random observation mask
                n_obs   = max(1, int(round(N * obs_frac)))
                obs_idx = rng.choice(N, size=n_obs, replace=False)
                obs_mask = np.zeros(N, dtype=bool)
                obs_mask[obs_idx] = True

                node_ctx, _, mean_obs = build_interp_context(
                    dist, obs_mask, positions_norm)
                global_ctx = build_interp_global(hour, dow, mean_obs)

                mu_start = rng.dirichlet(np.ones(N))
                pi       = compute_ot_coupling(mu_start, dist, graph_struct=graph_struct)
                cache.precompute_for_coupling(pi)
                triples.append((dist, node_ctx, global_ctx, pi))

        print(f"  Precomputed {len(triples)} OT couplings")

        self.samples = []
        for _ in range(n_samples):
            dist, node_ctx, global_ctx, pi = \
                triples[int(rng.integers(len(triples)))]
            tau      = float(rng.uniform(0.0, 0.999))
            mu_tau   = marginal_distribution_fast(cache, pi, tau)
            R_target = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi, tau)
            self.samples.append((
                torch.tensor(mu_tau,     dtype=torch.float32),
                torch.tensor([tau],      dtype=torch.float32),
                torch.tensor(node_ctx,   dtype=torch.float32),
                torch.tensor(global_ctx, dtype=torch.float32),
                torch.tensor(R_target,   dtype=torch.float32),
                self._edge_index,
                N,
            ))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# ── Forecasting dataset ───────────────────────────────────────────────────────

class BikeForecastDataset(torch.utils.data.Dataset):
    """
    For each consecutive (μ_t, μ_{t+1}) pair, flow from μ_t to μ_{t+1}.
    Conditioning: time features for hour t.
    """

    def __init__(self, R, train_dists, train_hours, train_dows,
                 positions_norm, n_starts_per_pair=5,
                 n_samples=20000, seed=42):
        rng          = np.random.default_rng(seed)
        N            = R.shape[0]
        graph_struct = GraphStructure(R)
        cache        = GeodesicCache(graph_struct)
        self._edge_index = rate_matrix_to_edge_index(R)

        triples = []
        T = len(train_dists)
        for t in range(T - 1):
            dist_t   = train_dists[t]
            dist_t1  = train_dists[t + 1]
            hour, dow = int(train_hours[t]), int(train_dows[t])

            node_ctx   = build_forecast_context(dist_t, positions_norm)
            global_ctx = build_forecast_global(hour, dow)

            for _ in range(n_starts_per_pair):
                mu_start = rng.dirichlet(np.ones(N))
                pi = compute_ot_coupling(mu_start, dist_t1, graph_struct=graph_struct)
                cache.precompute_for_coupling(pi)
                triples.append((dist_t1, node_ctx, global_ctx, pi))

        print(f"  Precomputed {len(triples)} OT couplings")

        self.samples = []
        for _ in range(n_samples):
            dist_t1, node_ctx, global_ctx, pi = \
                triples[int(rng.integers(len(triples)))]
            tau      = float(rng.uniform(0.0, 0.999))
            mu_tau   = marginal_distribution_fast(cache, pi, tau)
            R_target = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi, tau)
            self.samples.append((
                torch.tensor(mu_tau,     dtype=torch.float32),
                torch.tensor([tau],      dtype=torch.float32),
                torch.tensor(node_ctx,   dtype=torch.float32),
                torch.tensor(global_ctx, dtype=torch.float32),
                torch.tensor(R_target,   dtype=torch.float32),
                self._edge_index,
                N,
            ))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# ── Evaluation: interpolation ─────────────────────────────────────────────────

def eval_interpolation(model, test_dists, test_hours, test_dows,
                       R, adj, edge_index, positions_norm, hist_mean,
                       device, obs_frac=0.3, K=20, seed=99):
    """
    Evaluate interpolation on test snapshots.
    For each snapshot, randomly mask 1-obs_frac stations.
    Returns list of result dicts.
    """
    rng  = np.random.default_rng(seed)
    N    = R.shape[0]
    results = []

    for dist, hour, dow in zip(test_dists, test_hours, test_dows):
        n_obs    = max(1, int(round(N * obs_frac)))
        obs_idx  = rng.choice(N, size=n_obs, replace=False)
        obs_mask = np.zeros(N, dtype=bool)
        obs_mask[obs_idx] = True

        node_ctx, obs_vals, mean_obs = build_interp_context(
            dist, obs_mask, positions_norm)
        global_ctx = build_interp_global(int(hour), int(dow), mean_obs)

        # Posterior: K samples from Dirichlet starts
        samples = []
        for _ in range(K):
            mu_start = rng.dirichlet(np.ones(N))
            _, traj  = sample_trajectory_film(
                model, mu_start, node_ctx, global_ctx, edge_index,
                n_steps=200, device=device)
            samples.append(traj[-1])
        mu_learned = np.mean(samples, axis=0)

        # Baselines
        mu_knn  = baseline_knn_interp(dist, obs_mask, adj)
        mu_harm = baseline_harmonic(dist, obs_mask, adj)
        mu_hist = baseline_hist_mean(hist_mean, int(hour), int(dow))

        # Oracle: use the true values at unobserved stations = upper bound
        # but only at observed stations, fill rest with hist mean
        mu_obs_only          = dist.copy()
        mu_obs_only[~obs_mask] = mu_hist[~obs_mask]
        s = mu_obs_only.sum(); mu_obs_only /= s

        results.append({
            'dist':       dist,
            'obs_mask':   obs_mask,
            'mu_learned': mu_learned,
            'mu_knn':     mu_knn,
            'mu_harm':    mu_harm,
            'mu_hist':    mu_hist,
            'hour':       int(hour),
            'dow':        int(dow),
            'tv_learned': total_variation(mu_learned, dist),
            'tv_knn':     total_variation(mu_knn,     dist),
            'tv_harm':    total_variation(mu_harm,    dist),
            'tv_hist':    total_variation(mu_hist,    dist),
            # Unobserved stations only
            'tv_unobs_learned': total_variation(
                mu_learned[~obs_mask], dist[~obs_mask]),
            'tv_unobs_knn':  total_variation(mu_knn[~obs_mask],  dist[~obs_mask]),
            'tv_unobs_harm': total_variation(mu_harm[~obs_mask], dist[~obs_mask]),
            'tv_unobs_hist': total_variation(mu_hist[~obs_mask], dist[~obs_mask]),
            # Posterior
            'posterior_samples': samples,
            'posterior_std':     np.array(samples).std(axis=0),
        })

    return results


# ── Evaluation: forecasting ───────────────────────────────────────────────────

def eval_forecasting(model, test_dists, test_hours, test_dows,
                     R, edge_index, positions_norm, hist_mean,
                     device, K=10, seed=99):
    """
    Evaluate forecasting: predict μ_{t+1} given μ_t.
    Returns list of result dicts.
    """
    rng  = np.random.default_rng(seed)
    N    = R.shape[0]
    results = []
    T = len(test_dists)

    for t in range(T - 1):
        dist_t  = test_dists[t]
        dist_t1 = test_dists[t + 1]
        hour, dow = int(test_hours[t]), int(test_dows[t])

        node_ctx   = build_forecast_context(dist_t, positions_norm)
        global_ctx = build_forecast_global(hour, dow)

        # Model prediction (K posterior samples)
        samples = []
        for _ in range(K):
            mu_start = rng.dirichlet(np.ones(N))
            _, traj  = sample_trajectory_film(
                model, mu_start, node_ctx, global_ctx, edge_index,
                n_steps=200, device=device)
            samples.append(traj[-1])
        mu_learned = np.mean(samples, axis=0)

        # Baselines
        mu_persist = baseline_persistence(dist_t)
        mu_hist    = baseline_hist_mean(hist_mean, (hour + 1) % 24, dow)
        mu_linext  = (baseline_linear_extrap(dist_t, test_dists[t - 1])
                      if t > 0 else mu_persist)

        results.append({
            'dist_t':     dist_t,
            'dist_t1':    dist_t1,
            'mu_learned': mu_learned,
            'mu_persist': mu_persist,
            'mu_hist':    mu_hist,
            'mu_linext':  mu_linext,
            'hour':       hour,
            'dow':        dow,
            'tv_learned': total_variation(mu_learned, dist_t1),
            'tv_persist': total_variation(mu_persist, dist_t1),
            'tv_hist':    total_variation(mu_hist,    dist_t1),
            'tv_linext':  total_variation(mu_linext,  dist_t1),
            'posterior_samples': samples,
            'posterior_std':     np.array(samples).std(axis=0),
        })

    return results


# ── Print summaries ───────────────────────────────────────────────────────────

def print_interp_results(results, obs_frac):
    n = len(results)
    def mr(key): return np.mean([r[key] for r in results]), np.std([r[key] for r in results])
    print(f"\nInterpolation ({obs_frac:.0%} observed, {n} test snapshots):")
    print(f"  {'Method':12s}  {'TV (all)':>10s}  {'TV (unobs)':>12s}")
    print(f"  {'-'*40}")
    for tv_k, tv_u, name in [
        ('tv_learned', 'tv_unobs_learned', 'Learned'),
        ('tv_knn',     'tv_unobs_knn',     'k-NN'),
        ('tv_harm',    'tv_unobs_harm',     'Harmonic'),
        ('tv_hist',    'tv_unobs_hist',     'Hist. mean'),
    ]:
        m, s   = mr(tv_k)
        mu, su = mr(tv_u)
        print(f"  {name:12s}  {m:.4f}±{s:.4f}  {mu:.4f}±{su:.4f}")


def print_forecast_results(results):
    n = len(results)
    def mr(key): return np.mean([r[key] for r in results]), np.std([r[key] for r in results])
    print(f"\nForecasting ({n} test pairs):")
    print(f"  {'Method':12s}  {'TV':>12s}")
    print(f"  {'-'*30}")
    for tv_k, name in [
        ('tv_learned', 'Learned'),
        ('tv_persist', 'Persistence'),
        ('tv_hist',    'Hist. mean'),
        ('tv_linext',  'LinearExtrap'),
    ]:
        m, s = mr(tv_k)
        print(f"  {name:12s}  {m:.4f}±{s:.4f}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _station_scatter(ax, positions, vals, title='', cmap='YlOrRd', vmin=0, vmax=None,
                     markers=None):
    vm = vmax if vmax is not None else vals.max()
    sc = ax.scatter(positions[:, 0], positions[:, 1], c=vals, cmap=cmap,
                    vmin=vmin, vmax=vm, s=30, edgecolors='k', linewidths=0.2, zorder=3)
    if markers is not None:
        ax.scatter(positions[markers, 0], positions[markers, 1],
                   s=80, marker='*', c='blue', zorder=5,
                   edgecolors='white', linewidths=0.4)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    return sc


def plot_interpolation_figure(results, positions, losses=None, out_path=None):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Ex15b: Bike Sharing — Sparse Station Interpolation', fontsize=11)
    gs  = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.40)

    # Panel A: training loss
    ax_A = fig.add_subplot(gs[0, 0])
    if losses is not None:
        ax_A.plot(losses, lw=1.0, alpha=0.6, color='steelblue', label='loss')
        k = max(1, len(losses) // 50)
        smooth = np.convolve(losses, np.ones(k) / k, mode='valid')
        ax_A.plot(np.arange(k - 1, len(losses)), smooth, lw=1.5, color='red', label='smoothed')
        ax_A.set_yscale('log'); ax_A.legend(fontsize=7)
    else:
        ax_A.text(0.5, 0.5, 'Loaded from\ncheckpoint',
                  transform=ax_A.transAxes, ha='center', va='center', fontsize=12)
    ax_A.set_xlabel('Epoch'); ax_A.set_ylabel('Loss')
    ax_A.set_title('A: Training Loss', fontsize=9); ax_A.grid(True, alpha=0.3)

    # Panel B: example case — scatter comparison (True / Learned / k-NN)
    ex = results[len(results) // 3]   # pick a middle example
    vm = max(ex['dist'].max(), 0.02)
    ax_B = fig.add_subplot(gs[0, 1]); ax_B.axis('off')
    ax_B.set_title('B: Example Reconstruction (axial)', fontsize=9)
    inner_B = gs[0, 1].subgridspec(3, 1, hspace=0.5)
    obs_stars = np.where(ex['obs_mask'])[0]
    for row_i, (key, lbl, col_c) in enumerate([
        ('dist',       'True',    'black'),
        ('mu_learned', 'Learned', '#2166ac'),
        ('mu_knn',     'k-NN',    '#f58231'),
    ]):
        axi = fig.add_subplot(inner_B[row_i])
        _station_scatter(axi, positions, ex[key], vmin=0, vmax=vm,
                         markers=(obs_stars if key == 'dist' else None))
        tv_str = '' if key == 'dist' else f' TV={ex[key.replace("mu_", "tv_")]:.3f}'
        axi.set_title(lbl + tv_str, fontsize=7, color=col_c)

    # Panel C: TV bar chart
    ax_C = fig.add_subplot(gs[0, 2])
    methods = [('tv_learned','Learned','#2166ac'),('tv_knn','k-NN','#f58231'),
               ('tv_harm','Harmonic','#3cb44b'),('tv_hist','Hist. mean','#e6194b')]
    means_c = [np.mean([r[k] for r in results]) for k, _, _ in methods]
    stds_c  = [np.std( [r[k] for r in results]) for k, _, _ in methods]
    labels_c = [l for _, l, _ in methods]
    colors_c = [c for _, _, c in methods]
    ax_C.bar(np.arange(len(methods)), means_c, yerr=stds_c, capsize=4,
             color=colors_c, edgecolor='k', linewidth=0.4, alpha=0.85)
    ax_C.set_xticks(np.arange(len(methods)))
    ax_C.set_xticklabels(labels_c, rotation=20, ha='right', fontsize=8)
    ax_C.set_ylabel('Total Variation'); ax_C.grid(True, alpha=0.3, axis='y')
    ax_C.set_title(f'C: TV (all stations, {len(results)} tests)', fontsize=9)

    # Panel D: TV at unobserved stations only
    ax_D = fig.add_subplot(gs[1, 0])
    methods_u = [('tv_unobs_learned','Learned','#2166ac'),
                 ('tv_unobs_knn','k-NN','#f58231'),
                 ('tv_unobs_harm','Harmonic','#3cb44b'),
                 ('tv_unobs_hist','Hist. mean','#e6194b')]
    means_u = [np.mean([r[k] for r in results]) for k, _, _ in methods_u]
    stds_u  = [np.std( [r[k] for r in results]) for k, _, _ in methods_u]
    ax_D.bar(np.arange(len(methods_u)), means_u, yerr=stds_u, capsize=4,
             color=[c for _,_,c in methods_u], edgecolor='k', linewidth=0.4, alpha=0.85)
    ax_D.set_xticks(np.arange(len(methods_u)))
    ax_D.set_xticklabels([l for _,l,_ in methods_u], rotation=20, ha='right', fontsize=8)
    ax_D.set_ylabel('Total Variation'); ax_D.grid(True, alpha=0.3, axis='y')
    ax_D.set_title('D: TV (unobserved stations only)', fontsize=9)

    # Panel E: TV by hour of day
    ax_E = fig.add_subplot(gs[1, 1])
    hours = sorted(set(r['hour'] for r in results))
    for tv_k, lbl, color in methods:
        by_h = [np.mean([r[tv_k] for r in results if r['hour'] == h]) for h in hours]
        ax_E.plot(hours, by_h, 'o-', label=lbl, color=color, lw=1.4, ms=4)
    ax_E.set_xlabel('Hour of day'); ax_E.set_ylabel('Mean TV')
    ax_E.set_title('E: TV by Hour of Day', fontsize=9)
    ax_E.legend(fontsize=7); ax_E.grid(True, alpha=0.3)

    # Panel F: Posterior uncertainty map (mean std over test set)
    ax_F = fig.add_subplot(gs[1, 2])
    mean_std = np.mean([r['posterior_std'] for r in results], axis=0)
    sc = _station_scatter(ax_F, positions, mean_std, cmap='viridis')
    plt.colorbar(sc, ax=ax_F, fraction=0.04, label='Mean posterior std')
    ax_F.set_title('F: Posterior Uncertainty (mean std)', fontsize=9)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")
    return fig


def plot_forecasting_figure(results, positions, losses=None, out_path=None):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Ex15b: Bike Sharing — 1-Hour Forecasting', fontsize=11)
    gs  = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.40)

    # Panel A: training loss
    ax_A = fig.add_subplot(gs[0, 0])
    if losses is not None:
        ax_A.plot(losses, lw=1.0, alpha=0.6, color='steelblue', label='loss')
        k = max(1, len(losses) // 50)
        smooth = np.convolve(losses, np.ones(k) / k, mode='valid')
        ax_A.plot(np.arange(k - 1, len(losses)), smooth, lw=1.5, color='red')
        ax_A.set_yscale('log'); ax_A.legend(fontsize=7)
    else:
        ax_A.text(0.5, 0.5, 'Loaded from\ncheckpoint',
                  transform=ax_A.transAxes, ha='center', va='center', fontsize=12)
    ax_A.set_xlabel('Epoch'); ax_A.set_ylabel('Loss')
    ax_A.set_title('A: Training Loss', fontsize=9); ax_A.grid(True, alpha=0.3)

    # Panel B: example case — Current / Predicted / True
    ex = results[len(results) // 4]
    vm = max(ex['dist_t'].max(), ex['dist_t1'].max(), 0.02)
    ax_B = fig.add_subplot(gs[0, 1]); ax_B.axis('off')
    ax_B.set_title('B: Forecast Example', fontsize=9)
    inner_B = gs[0, 1].subgridspec(3, 1, hspace=0.5)
    for row_i, (key, lbl, col_c) in enumerate([
        ('dist_t',     f'Current (h={ex["hour"]})',     'black'),
        ('mu_learned', 'Predicted (h+1)',  '#2166ac'),
        ('dist_t1',    'True (h+1)',       '#3cb44b'),
    ]):
        axi = fig.add_subplot(inner_B[row_i])
        _station_scatter(axi, positions, ex[key], vmin=0, vmax=vm)
        tv_str = ''
        if key == 'mu_learned':
            tv_str = f' TV={ex["tv_learned"]:.3f}'
        axi.set_title(lbl + tv_str, fontsize=7, color=col_c)

    # Panel C: TV bar chart
    ax_C = fig.add_subplot(gs[0, 2])
    methods_f = [('tv_learned','Learned','#2166ac'),
                 ('tv_persist','Persistence','#f58231'),
                 ('tv_hist','Hist. mean','#3cb44b'),
                 ('tv_linext','LinearExtrap','#e6194b')]
    means_f = [np.mean([r[k] for r in results]) for k, _, _ in methods_f]
    stds_f  = [np.std( [r[k] for r in results]) for k, _, _ in methods_f]
    ax_C.bar(np.arange(len(methods_f)), means_f, yerr=stds_f, capsize=4,
             color=[c for _,_,c in methods_f], edgecolor='k', linewidth=0.4, alpha=0.85)
    ax_C.set_xticks(np.arange(len(methods_f)))
    ax_C.set_xticklabels([l for _,l,_ in methods_f], rotation=20, ha='right', fontsize=8)
    ax_C.set_ylabel('Total Variation'); ax_C.grid(True, alpha=0.3, axis='y')
    ax_C.set_title(f'C: Forecast TV ({len(results)} test pairs)', fontsize=9)

    # Panel D: TV by hour of day
    ax_D = fig.add_subplot(gs[1, 0])
    hours = sorted(set(r['hour'] for r in results))
    for tv_k, lbl, color in methods_f:
        by_h = [np.mean([r[tv_k] for r in results if r['hour'] == h]) for h in hours]
        ax_D.plot(hours, by_h, 'o-', label=lbl, color=color, lw=1.4, ms=4)
    ax_D.set_xlabel('Hour of day'); ax_D.set_ylabel('Mean TV')
    ax_D.set_title('D: Forecast TV by Hour of Day', fontsize=9)
    ax_D.legend(fontsize=7); ax_D.grid(True, alpha=0.3)

    # Panel E: TV by day of week
    ax_E = fig.add_subplot(gs[1, 1])
    dows     = sorted(set(r['dow'] for r in results))
    dow_lbls = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    for tv_k, lbl, color in methods_f:
        by_d = [np.mean([r[tv_k] for r in results if r['dow'] == d]) for d in dows]
        ax_E.plot(dows, by_d, 'o-', label=lbl, color=color, lw=1.4, ms=4)
    ax_E.set_xticks(dows)
    ax_E.set_xticklabels([dow_lbls[d] for d in dows], fontsize=8)
    ax_E.set_ylabel('Mean TV')
    ax_E.set_title('E: Forecast TV by Day of Week', fontsize=9)
    ax_E.legend(fontsize=7); ax_E.grid(True, alpha=0.3)

    # Panel F: Posterior uncertainty (forecast)
    ax_F = fig.add_subplot(gs[1, 2])
    mean_std = np.mean([r['posterior_std'] for r in results[:100]], axis=0)
    sc = _station_scatter(ax_F, positions, mean_std, cmap='viridis')
    plt.colorbar(sc, ax=ax_F, fraction=0.04, label='Forecast posterior std')
    ax_F.set_title('F: Forecast Uncertainty (mean std)', fontsize=9)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ex15b: Citi Bike NYC — model training and evaluation')
    parser.add_argument('--task',           type=str,   default='both',
                        choices=['interpolation', 'forecasting', 'both'])
    parser.add_argument('--data-path',      type=str,   default='ex15_bikeshare_data.npz')
    parser.add_argument('--obs-fraction',   type=float, default=0.3,
                        help='Fraction of stations observed (interpolation)')
    parser.add_argument('--mode',           type=str,   default='posterior',
                        choices=['point_estimate', 'posterior'])
    parser.add_argument('--n-epochs',       type=int,   default=1000)
    parser.add_argument('--hidden-dim',     type=int,   default=128)
    parser.add_argument('--n-layers',       type=int,   default=6)
    parser.add_argument('--lr',             type=float, default=5e-4)
    parser.add_argument('--ema-decay',      type=float, default=0.999)
    parser.add_argument('--loss-type',      type=str,   default='rate_kl',
                        choices=['rate_kl', 'mse'])
    parser.add_argument('--n-samples',      type=int,   default=20000)
    parser.add_argument('--n-starts',       type=int,   default=3,
                        help='Dirichlet starts per training snapshot')
    parser.add_argument('--posterior-k',    type=int,   default=20,
                        help='Posterior samples at test time')
    parser.add_argument('--n-test-snaps',   type=int,   default=200,
                        help='Number of test snapshots to evaluate')
    parser.add_argument('--seed',           type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(args.seed)
    device = get_device()

    # ── Load data ──────────────────────────────────────────────────────────────
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(ROOT, data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data not found at {data_path}.\n"
            "Run: uv run experiments/ex15a_bikeshare_setup.py")

    print(f"=== Experiment 15b: Citi Bike NYC Model Training ===\n")
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)

    R               = data['R']
    adj             = data['adj']
    positions       = data['positions']    # (N, 2) metres
    train_dists     = data['train_dists']  # (T_train, N)
    train_hours     = data['train_hours']  # (T_train,)
    train_dows      = data['train_dows']   # (T_train,)
    test_dists      = data['test_dists']   # (T_test, N)
    test_hours      = data['test_hours']
    test_dows       = data['test_dows']
    hist_mean       = data['hist_mean']    # (24, 7, N)

    N = R.shape[0]
    edge_index = rate_matrix_to_edge_index(R)

    # Normalise positions for context (zero mean, unit std)
    pos_mean = positions.mean(axis=0)
    pos_std  = positions.std(axis=0) + 1e-8
    positions_norm = (positions - pos_mean) / pos_std

    # Subsample test set
    n_test = min(args.n_test_snaps, len(test_dists))
    idx    = np.linspace(0, len(test_dists) - 1, n_test, dtype=int)
    test_dists_sub  = test_dists[idx]
    test_hours_sub  = test_hours[idx]
    test_dows_sub   = test_dows[idx]

    print(f"  Stations: {N}, Train: {len(train_dists)}, Test: {n_test}")
    print(f"  Task: {args.task}")

    # ── Helper: build/load model ───────────────────────────────────────────────
    def _ckpt(tag):
        return os.path.join(CKPT_DIR,
            f'meta_model_ex15b_{tag}_{args.n_epochs}ep'
            f'_h{args.hidden_dim}_l{args.n_layers}.pt')

    def _train_or_load(tag, dataset, global_dim, node_dim):
        mdl = FiLMConditionalGNNRateMatrixPredictor(
            node_context_dim=node_dim,
            global_dim=global_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers)
        ck  = _ckpt(tag)
        lss = None
        if os.path.exists(ck):
            mdl.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
            print(f"  Loaded from {ck}")
        else:
            print(f"  Training ({args.n_epochs} epochs, {len(dataset)} samples)...")
            hist = train_film_conditional(
                mdl, dataset,
                n_epochs=args.n_epochs, batch_size=256, lr=args.lr,
                device=device, loss_weighting='uniform',
                loss_type=args.loss_type, ema_decay=args.ema_decay)
            lss = hist['losses']
            print(f"  Loss: {lss[0]:.6f} → {lss[-1]:.6f}")
            torch.save(mdl.state_dict(), ck)
            print(f"  Saved to {ck}")
        mdl = mdl.to(device)
        mdl.eval()
        return mdl, lss

    # ── Task 1: Interpolation ─────────────────────────────────────────────────
    if args.task in ('interpolation', 'both'):
        print(f"\n--- Task 1: Spatial Interpolation ({args.obs_fraction:.0%} observed) ---")
        print(f"Building BikeInterpolationDataset ({args.n_samples} samples)...")
        dataset_interp = BikeInterpolationDataset(
            R, train_dists, train_hours, train_dows,
            positions_norm,
            obs_frac=args.obs_fraction,
            n_starts_per_snap=args.n_starts,
            n_samples=args.n_samples,
            seed=args.seed)

        # global_dim = 5 (4 time features + 1 mean obs)
        # node_dim = 3 (obs_val, is_obs, dist_to_nearest_obs)
        model_interp, losses_interp = _train_or_load(
            'interp', dataset_interp, global_dim=5, node_dim=3)

        print(f"\nEvaluating interpolation on {n_test} test snapshots "
              f"(K={args.posterior_k} posterior samples)...")
        results_interp = eval_interpolation(
            model_interp, test_dists_sub, test_hours_sub, test_dows_sub,
            R, adj, edge_index, positions_norm, hist_mean,
            device, obs_frac=args.obs_fraction,
            K=args.posterior_k, seed=99)

        print_interp_results(results_interp, args.obs_fraction)

        plot_interpolation_figure(
            results_interp, positions,
            losses=losses_interp,
            out_path=os.path.join(HERE, 'ex15b_interpolation.png'))

    # ── Task 2: Forecasting ───────────────────────────────────────────────────
    if args.task in ('forecasting', 'both'):
        print(f"\n--- Task 2: 1-Hour Forecasting ---")
        print(f"Building BikeForecastDataset ({args.n_samples} samples)...")
        dataset_forecast = BikeForecastDataset(
            R, train_dists, train_hours, train_dows,
            positions_norm,
            n_starts_per_pair=args.n_starts,
            n_samples=args.n_samples,
            seed=args.seed)

        # global_dim = 4 (time features), node_dim = 2 (current dist + 1)
        model_forecast, losses_forecast = _train_or_load(
            'forecast', dataset_forecast, global_dim=4, node_dim=2)

        n_forecast_test = min(n_test, len(test_dists) - 1)
        test_dists_fc   = test_dists[:n_forecast_test + 1]
        test_hours_fc   = test_hours[:n_forecast_test + 1]
        test_dows_fc    = test_dows[:n_forecast_test + 1]

        print(f"\nEvaluating forecasting on {n_forecast_test} test pairs "
              f"(K={args.posterior_k // 2} posterior samples)...")
        results_forecast = eval_forecasting(
            model_forecast, test_dists_fc, test_hours_fc, test_dows_fc,
            R, edge_index, positions_norm, hist_mean,
            device, K=max(1, args.posterior_k // 2), seed=99)

        print_forecast_results(results_forecast)

        plot_forecasting_figure(
            results_forecast, positions,
            losses=losses_forecast,
            out_path=os.path.join(HERE, 'ex15b_forecasting.png'))

    print("\nDone.")


if __name__ == '__main__':
    main()
