"""
Experiment 15b: Bike Sharing Spatial Interpolation

Given bike counts at 30% of stations, reconstruct the full distribution.
Posterior sampling provides uncertainty over unobserved stations.

Run: uv run experiments/ex15b_bikeshare_interpolation.py [--obs-fraction 0.3]
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
)
from meta_fm.sample import sample_posterior_film
from meta_fm.model import rate_matrix_to_edge_index

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
CKPT_DIR = os.path.join(ROOT, 'checkpoints')
DATA_DIR = os.path.join(ROOT, 'data')


# ── Observation model ─────────────────────────────────────────────────────────

def mask_distribution(mu, obs_fraction=0.3, rng=None):
    """
    Randomly observe a fraction of stations.
    Returns (mu_obs, mask) where mu_obs has unobserved entries zeroed
    and renormalized over observed stations only.
    """
    N     = len(mu)
    n_obs = max(1, int(N * obs_fraction))
    obs_idx = rng.choice(N, size=n_obs, replace=False)
    mask = np.zeros(N, dtype=np.float32)
    mask[obs_idx] = 1.0
    mu_obs = mu * mask
    s = mu_obs.sum()
    if s > 1e-12:
        mu_obs = mu_obs / s
    return mu_obs, mask


def build_bike_context(mu_obs, mask, N):
    """
    Per-node:  [observed_value * mask, mask]  — (N, 2)
    Global:    raw observation vector (zeros for unobserved)  — (N,)
    """
    node_context = np.stack([mu_obs * mask, mask], axis=-1).astype(np.float32)
    global_cond  = (mu_obs * mask).astype(np.float32)
    return node_context, global_cond


# ── Baselines ─────────────────────────────────────────────────────────────────

def baseline_laplacian(mu_obs, mask, R):
    """
    Harmonic extension: fix observed values, solve Dirichlet problem
    for unobserved using the graph Laplacian L = -R.
    """
    N        = len(mu_obs)
    obs_idx  = np.where(mask == 1)[0]
    unob_idx = np.where(mask == 0)[0]

    if len(unob_idx) == 0:
        result = mu_obs.copy()
        s = result.sum()
        return result / s if s > 1e-12 else np.ones(N) / N
    if len(obs_idx) == 0:
        return np.ones(N) / N

    L    = -R.copy()                            # positive semi-definite Laplacian
    L_II = L[np.ix_(unob_idx, unob_idx)]
    L_IB = L[np.ix_(unob_idx, obs_idx)]
    u_B  = mu_obs[obs_idx]

    # Regularise for numerical stability
    L_II_reg = L_II + 1e-8 * np.eye(len(unob_idx))
    try:
        u_I = np.linalg.solve(L_II_reg, -L_IB @ u_B)
    except np.linalg.LinAlgError:
        u_I = np.linalg.lstsq(L_II_reg, -L_IB @ u_B, rcond=None)[0]

    result = mu_obs.copy()
    result[unob_idx] = u_I
    result = np.clip(result, 0, None)
    s = result.sum()
    return result / s if s > 1e-12 else np.ones(N) / N


def baseline_knn(mu_obs, mask, positions, k=3):
    """
    Fill each unobserved station with the mean of its k geographically
    nearest observed neighbours.
    """
    N        = len(mu_obs)
    obs_idx  = np.where(mask == 1)[0]
    unob_idx = np.where(mask == 0)[0]

    result = mu_obs.copy()
    for i in unob_idx:
        if len(obs_idx) == 0:
            result[i] = 1.0 / N
            continue
        dists   = np.linalg.norm(positions[obs_idx] - positions[i], axis=1)
        nearest = obs_idx[np.argsort(dists)[:k]]
        result[i] = mu_obs[nearest].mean()

    result = np.clip(result, 0, None)
    s = result.sum()
    return result / s if s > 1e-12 else np.ones(N) / N


def baseline_historical_mean(hour, dow, train_dists, train_hours, train_dows):
    """Average distribution at same hour + day-of-week from training data."""
    mask_hd = (train_hours == hour) & (train_dows == dow)
    if mask_hd.sum() > 0:
        return train_dists[mask_hd].mean(axis=0)
    mask_h = train_hours == hour
    if mask_h.sum() > 0:
        return train_dists[mask_h].mean(axis=0)
    return train_dists.mean(axis=0)


def baseline_hist_laplacian(mu_obs, mask, hist_mean):
    """
    Hybrid: use historical mean as prior, override with observed values,
    then renormalise.  Simple but strong — combines temporal knowledge
    with current observations.
    """
    result = hist_mean.copy().astype(np.float32)
    obs_idx = np.where(mask == 1)[0]
    result[obs_idx] = mu_obs[obs_idx]
    result = np.clip(result, 0, None)
    s = result.sum()
    return result / s if s > 1e-12 else np.ones(len(result)) / len(result)


# ── Dataset ───────────────────────────────────────────────────────────────────

class BikeInterpolationDataset(torch.utils.data.Dataset):
    """
    Training data for bike sharing interpolation.

    For each hourly snapshot, randomly mask 1-obs_fraction of stations.
    Flow: Dirichlet (or uniform) → true distribution, conditioned on
    the masked observation.

    Returns 7-tuple for train_film_conditional:
        (mu_tau, tau, node_context, global_cond, R_target, edge_index, N)
    """

    def __init__(self, R, distributions, obs_fraction=0.3,
                 mode='dirichlet', dirichlet_alpha=1.0,
                 n_starts_per_dist=5, n_samples=15000, seed=42):
        rng          = np.random.default_rng(seed)
        N            = R.shape[0]
        graph_struct = GraphStructure(R)
        cache        = GeodesicCache(graph_struct)
        self._edge_index = rate_matrix_to_edge_index(R)

        all_triples = []
        for mu_clean in distributions:
            mu_obs, mask = mask_distribution(mu_clean, obs_fraction, rng)
            node_ctx, global_ctx = build_bike_context(mu_obs, mask, N)

            n_starts = n_starts_per_dist if mode == 'dirichlet' else 1
            for _ in range(n_starts):
                if mode == 'dirichlet':
                    mu_start = rng.dirichlet(np.full(N, dirichlet_alpha))
                else:
                    mu_start = np.ones(N) / N
                pi = compute_ot_coupling(mu_start, mu_clean, graph_struct=graph_struct)
                cache.precompute_for_coupling(pi)
                all_triples.append((mu_clean, node_ctx, global_ctx, pi))

        print(f"  Precomputed {len(all_triples)} OT couplings "
              f"({len(distributions)} snapshots × {n_starts_per_dist} starts)")

        self.N = N
        self.samples = []
        for _ in range(n_samples):
            mu_clean, node_ctx, global_ctx, pi = \
                all_triples[int(rng.integers(len(all_triples)))]
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


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_interpolation(model, edge_index, test_dists, test_hours, test_dows,
                       R, positions, train_dists, train_hours, train_dows,
                       hist_mean, device, obs_fraction=0.3, K=20, seed=99):
    """
    Evaluate on held-out snapshots. Returns list of result dicts.
    """
    rng = np.random.default_rng(seed)
    N   = R.shape[0]
    results = []

    for mu_clean, hour, dow in zip(test_dists, test_hours, test_dows):
        hour, dow = int(hour), int(dow)
        mu_obs, mask = mask_distribution(mu_clean, obs_fraction, rng)
        node_ctx, global_ctx = build_bike_context(mu_obs, mask, N)
        obs_idx  = np.where(mask == 1)[0]
        unob_idx = np.where(mask == 0)[0]

        # Model: K Dirichlet starts → posterior mean (batched)
        mu_starts = rng.dirichlet(np.ones(N), size=K)   # (K, N)
        samples = list(sample_posterior_film(
            model, mu_starts, node_ctx, global_ctx, edge_index,
            n_steps=100, device=device))
        mu_learned = np.mean(samples, axis=0)
        post_std   = np.std(samples, axis=0)

        # Baselines
        hist = baseline_historical_mean(hour, dow, train_dists, train_hours, train_dows)
        mu_lap  = baseline_laplacian(mu_obs, mask, R)
        mu_knn  = baseline_knn(mu_obs, mask, positions)
        mu_hist = hist / hist.sum()
        mu_hl   = baseline_hist_laplacian(mu_obs, mask, mu_hist)

        # Full TV
        tv_l    = total_variation(mu_learned, mu_clean)
        tv_lap  = total_variation(mu_lap,     mu_clean)
        tv_knn  = total_variation(mu_knn,     mu_clean)
        tv_hist = total_variation(mu_hist,    mu_clean)
        tv_hl   = total_variation(mu_hl,      mu_clean)

        # Unobserved-only TV (unnormalised subset)
        def tv_sub(a, b, idx):
            return 0.5 * np.abs(a[idx] - b[idx]).sum()

        tvu_l    = tv_sub(mu_learned, mu_clean, unob_idx)
        tvu_lap  = tv_sub(mu_lap,     mu_clean, unob_idx)
        tvu_knn  = tv_sub(mu_knn,     mu_clean, unob_idx)
        tvu_hist = tv_sub(mu_hist,    mu_clean, unob_idx)
        tvu_hl   = tv_sub(mu_hl,      mu_clean, unob_idx)

        # Calibration: corr(posterior_std, |error|)
        err    = np.abs(mu_learned - mu_clean)
        calib_r = float(np.corrcoef(post_std, err)[0, 1]) \
            if post_std.std() > 1e-8 else 0.0

        # Diversity: mean pairwise TV
        divs = [total_variation(samples[i], samples[j])
                for i in range(K) for j in range(i + 1, K)]
        diversity = float(np.mean(divs))

        results.append({
            'mu_clean':   mu_clean,
            'mu_obs':     mu_obs,
            'mask':       mask,
            'mu_learned': mu_learned,
            'mu_lap':     mu_lap,
            'mu_knn':     mu_knn,
            'mu_hist':    mu_hist,
            'mu_hl':      mu_hl,
            'samples':    samples,
            'post_std':   post_std,
            'hour':       hour,
            'dow':        dow,
            'tv_l':       tv_l,   'tv_lap':  tv_lap,  'tv_knn':  tv_knn,
            'tv_hist':    tv_hist, 'tv_hl':   tv_hl,
            'tvu_l':      tvu_l,  'tvu_lap': tvu_lap, 'tvu_knn': tvu_knn,
            'tvu_hist':   tvu_hist,'tvu_hl':  tvu_hl,
            'calib_r':    calib_r,
            'diversity':  diversity,
        })

    return results


def eval_obs_fraction_sweep(model, edge_index, test_dists, test_hours, test_dows,
                             R, positions, train_dists, train_hours, train_dows,
                             hist_mean, device,
                             fracs=(0.1, 0.2, 0.3, 0.5), seed=77):
    """Run evaluation at multiple observation fractions."""
    sweep = {}
    for frac in fracs:
        res = eval_interpolation(
            model, edge_index, test_dists, test_hours, test_dows,
            R, positions, train_dists, train_hours, train_dows,
            hist_mean, device, obs_fraction=frac, K=10, seed=seed)
        sweep[frac] = {
            'tv_l':    np.mean([r['tv_l']    for r in res]),
            'tv_lap':  np.mean([r['tv_lap']  for r in res]),
            'tv_knn':  np.mean([r['tv_knn']  for r in res]),
            'tv_hist': np.mean([r['tv_hist'] for r in res]),
            'tv_hl':   np.mean([r['tv_hl']   for r in res]),
        }
    return sweep


# ── Console output ────────────────────────────────────────────────────────────

def print_results(results, obs_fraction, sweep=None):
    n = len(results)
    def mr(key): return np.mean([r[key] for r in results]), np.std([r[key] for r in results])

    print(f"\nFull TV ({obs_fraction:.0%} observed, {n} test cases):")
    print(f"  {'Method':24s}  {'Mean':>8s} ± {'Std':>8s}")
    for key, name in [('tv_l','Learned (posterior mean)'),
                       ('tv_lap','Laplacian'),
                       ('tv_knn','k-NN'),
                       ('tv_hist','Historical mean'),
                       ('tv_hl','Hist + observed')]:
        m, s = mr(key)
        print(f"  {name:24s}  {m:.4f}   ± {s:.4f}")

    print(f"\nUnobserved TV:")
    for key, name in [('tvu_l','Learned'),('tvu_lap','Laplacian'),
                       ('tvu_knn','k-NN'),('tvu_hist','Hist. mean'),('tvu_hl','Hist + obs')]:
        m, s = mr(key)
        print(f"  {name:24s}  {m:.4f}   ± {s:.4f}")

    calib = np.mean([r['calib_r']  for r in results])
    div   = np.mean([r['diversity'] for r in results])
    print(f"\nCalibration: r = {calib:.3f}")
    print(f"Diversity:   {div:.4f}")

    # By hour bucket
    buckets = {'Night(0-5)': range(0,6), 'Morning(6-11)': range(6,12),
               'Afternoon(12-17)': range(12,18), 'Evening(18-23)': range(18,24)}
    print(f"\nBy time of day:")
    print(f"  {'Bucket':20s}  {'Learned':>8s}  {'Laplacian':>10s}  {'Hist+obs':>9s}")
    for bname, hrs in buckets.items():
        sub = [r for r in results if r['hour'] in hrs]
        if not sub: continue
        tl = np.mean([r['tv_l']  for r in sub])
        tla = np.mean([r['tv_lap'] for r in sub])
        th = np.mean([r['tv_hl']  for r in sub])
        print(f"  {bname:20s}  {tl:.4f}    {tla:.4f}      {th:.4f}")

    # Weekday vs weekend
    print(f"\nWeekday vs Weekend:")
    for label, dows_sel in [('Weekday', range(5)), ('Weekend', range(5,7))]:
        sub = [r for r in results if r['dow'] in dows_sel]
        if not sub: continue
        tl = np.mean([r['tv_l']  for r in sub])
        tla = np.mean([r['tv_lap'] for r in sub])
        th = np.mean([r['tv_hl']  for r in sub])
        print(f"  {label:10s}  Learned={tl:.4f}  Laplacian={tla:.4f}  Hist+obs={th:.4f}")

    if sweep:
        print(f"\nBy observation fraction:")
        print(f"  {'Frac':>5s}  {'Learned':>8s}  {'Laplacian':>10s}  "
              f"{'k-NN':>8s}  {'Hist+obs':>9s}")
        for frac, tvs in sorted(sweep.items()):
            print(f"  {frac:>5.0%}  {tvs['tv_l']:.4f}    {tvs['tv_lap']:.4f}      "
                  f"{tvs['tv_knn']:.4f}   {tvs['tv_hl']:.4f}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def _map(ax, positions, vals, title='', cmap='YlOrRd', vmin=0, vmax=None,
         obs_mask=None, s=60):
    vm  = vmax if vmax is not None else max(vals.max(), 1e-6)
    sc  = ax.scatter(positions[:, 0], positions[:, 1],
                     c=vals, cmap=cmap, vmin=vmin, vmax=vm, s=s,
                     edgecolors='k', linewidths=0.3, zorder=3)
    if obs_mask is not None:
        # highlight observed stations with a white ring
        obs_idx = np.where(obs_mask)[0]
        ax.scatter(positions[obs_idx, 0], positions[obs_idx, 1],
                   s=s * 2.2, facecolors='none', edgecolors='white',
                   linewidths=1.5, zorder=4)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=8)
    return sc


def plot_main_figure(results, positions, losses=None, sweep=None, out_path=None):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Ex15b: Citi Bike Spatial Interpolation (30% observed)', fontsize=11)
    gs  = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.40)

    # Panel A: training loss
    ax_A = fig.add_subplot(gs[0, 0])
    if losses is not None:
        ax_A.plot(losses, lw=1.0, alpha=0.6, color='steelblue', label='loss')
        k = max(1, len(losses) // 50)
        smooth = np.convolve(losses, np.ones(k) / k, mode='valid')
        ax_A.plot(np.arange(k-1, len(losses)), smooth, lw=1.5, color='red', label='smooth')
        ax_A.set_yscale('log'); ax_A.legend(fontsize=7)
    else:
        ax_A.text(0.5, 0.5, 'Loaded from\ncheckpoint',
                  transform=ax_A.transAxes, ha='center', va='center', fontsize=12)
    ax_A.set_xlabel('Epoch'); ax_A.set_ylabel('Loss')
    ax_A.set_title('A: Training Loss', fontsize=9); ax_A.grid(True, alpha=0.3)

    # Panel B: example interpolation (True / Observed / Learned / Laplacian)
    ex   = results[len(results) // 3]
    vm_b = max(ex['mu_clean'].max(), 0.02)
    ax_B = fig.add_subplot(gs[0, 1]); ax_B.axis('off')
    ax_B.set_title('B: Interpolation Example', fontsize=9, pad=4)
    inner_B = gs[0, 1].subgridspec(2, 2, hspace=0.5, wspace=0.4)
    n_obs_pct = int(ex['mask'].mean() * 100)
    for ri, ci, (key, lbl, col_c) in zip(
        [0,0,1,1], [0,1,0,1],
        [('mu_clean',  f'True',               'black'),
         ('mu_obs',    f'Observed ({n_obs_pct}%)', 'gray'),
         ('mu_learned', f'Learned TV={ex["tv_l"]:.3f}', '#2166ac'),
         ('mu_lap',    f'Laplacian TV={ex["tv_lap"]:.3f}', '#f58231')]):
        axi = fig.add_subplot(inner_B[ri, ci])
        obs_m = ex['mask'].astype(bool) if key in ('mu_clean', 'mu_obs') else None
        _map(axi, positions, ex[key], vmin=0, vmax=vm_b, obs_mask=obs_m, s=35)
        axi.set_title(lbl, fontsize=7, color=col_c)

    # Panel C: Full TV bar chart
    ax_C = fig.add_subplot(gs[0, 2])
    methods = [('tv_l','Learned','#2166ac'),('tv_lap','Laplacian','#f58231'),
               ('tv_knn','k-NN','#3cb44b'),('tv_hist','Hist.mean','#e6194b'),
               ('tv_hl','Hist+obs','#911eb4')]
    means_c = [np.mean([r[k] for r in results]) for k,_,_ in methods]
    stds_c  = [np.std( [r[k] for r in results]) for k,_,_ in methods]
    ax_C.bar(np.arange(len(methods)), means_c, yerr=stds_c, capsize=4,
             color=[c for _,_,c in methods], edgecolor='k', linewidth=0.4, alpha=0.85)
    ax_C.set_xticks(np.arange(len(methods)))
    ax_C.set_xticklabels([l for _,l,_ in methods], rotation=25, ha='right', fontsize=8)
    ax_C.set_ylabel('Total Variation'); ax_C.grid(True, alpha=0.3, axis='y')
    ax_C.set_title(f'C: Full TV ({len(results)} tests)', fontsize=9)

    # Panel D: Unobserved-only TV
    ax_D = fig.add_subplot(gs[1, 0])
    methods_u = [('tvu_l','Learned','#2166ac'),('tvu_lap','Laplacian','#f58231'),
                 ('tvu_knn','k-NN','#3cb44b'),('tvu_hist','Hist.mean','#e6194b'),
                 ('tvu_hl','Hist+obs','#911eb4')]
    means_u = [np.mean([r[k] for r in results]) for k,_,_ in methods_u]
    stds_u  = [np.std( [r[k] for r in results]) for k,_,_ in methods_u]
    ax_D.bar(np.arange(len(methods_u)), means_u, yerr=stds_u, capsize=4,
             color=[c for _,_,c in methods_u], edgecolor='k', linewidth=0.4, alpha=0.85)
    ax_D.set_xticks(np.arange(len(methods_u)))
    ax_D.set_xticklabels([l for _,l,_ in methods_u], rotation=25, ha='right', fontsize=8)
    ax_D.set_ylabel('Unobserved TV'); ax_D.grid(True, alpha=0.3, axis='y')
    ax_D.set_title('D: Unobserved Stations TV', fontsize=9)

    # Panel E: TV vs obs fraction
    ax_E = fig.add_subplot(gs[1, 1])
    if sweep:
        fracs = sorted(sweep.keys())
        for tv_k, lbl, color in [('tv_l','Learned','#2166ac'),('tv_lap','Laplacian','#f58231'),
                                   ('tv_knn','k-NN','#3cb44b'),('tv_hl','Hist+obs','#911eb4')]:
            ax_E.plot([f * 100 for f in fracs],
                      [sweep[f][tv_k] for f in fracs],
                      'o-', label=lbl, color=color, lw=1.5, ms=5)
        ax_E.set_xlabel('Observed stations (%)')
        ax_E.set_ylabel('Mean TV')
        ax_E.set_title('E: TV vs Observation Fraction', fontsize=9)
        ax_E.legend(fontsize=7); ax_E.grid(True, alpha=0.3)
    else:
        ax_E.text(0.5, 0.5, 'Run with\n--obs-sweep',
                  transform=ax_E.transAxes, ha='center', va='center', fontsize=12)
        ax_E.set_title('E: TV vs Obs Fraction', fontsize=9)

    # Panel F: Calibration scatter
    ax_F = fig.add_subplot(gs[1, 2])
    all_std = np.concatenate([r['post_std']                          for r in results])
    all_err = np.concatenate([np.abs(r['mu_learned'] - r['mu_clean']) for r in results])
    ax_F.scatter(all_std, all_err, alpha=0.02, s=3, c='steelblue', rasterized=True)
    r_val = float(np.mean([r['calib_r'] for r in results]))
    ax_F.set_xlabel('Posterior std'); ax_F.set_ylabel('|Predicted − True|')
    ax_F.set_title(f'F: Calibration (mean r = {r_val:.3f})', fontsize=9)
    ax_F.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")
    return fig


def plot_posterior_figure(results, positions, out_path=None):
    """The 'aha' figure: True / Posterior mean / Uncertainty / 4 samples."""
    ex    = results[len(results) // 4]
    K     = len(ex['samples'])
    n_sp  = min(4, K)
    vm    = max(ex['mu_clean'].max(), 0.02)

    ncols = 3 + n_sp
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 4.5))
    fig.suptitle(
        f'Ex15b: Posterior Sampling — Station map (h={ex["hour"]}, '
        f'dow={ex["dow"]}, TV={ex["tv_l"]:.3f})', fontsize=10)

    # True distribution
    ax = axes[0]
    sc = _map(ax, positions, ex['mu_clean'], vmin=0, vmax=vm,
              obs_mask=ex['mask'].astype(bool), s=55)
    ax.set_title('True distribution\n(★ = observed)', fontsize=8)
    plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)

    # Posterior mean
    ax = axes[1]
    sc = _map(ax, positions, ex['mu_learned'], vmin=0, vmax=vm, s=55)
    ax.set_title(f'Posterior mean\nTV={ex["tv_l"]:.3f}', fontsize=8)
    plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)

    # Posterior uncertainty
    ax = axes[2]
    sc = _map(ax, positions, ex['post_std'], cmap='viridis',
              vmin=0, vmax=ex['post_std'].max(), s=55)
    ax.set_title('Posterior std\n(uncertainty)', fontsize=8)
    plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)

    # Individual posterior samples
    for k in range(n_sp):
        ax  = axes[3 + k]
        sc  = _map(ax, positions, ex['samples'][k], vmin=0, vmax=vm, s=55)
        ax.set_title(f'Sample {k+1}', fontsize=8)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ex15b: Citi Bike spatial interpolation')
    parser.add_argument('--data-path',       type=str,   default='data/ex15_bikeshare_data.npz')
    parser.add_argument('--obs-fraction',    type=float, default=0.3)
    parser.add_argument('--mode',            type=str,   default='posterior',
                        choices=['point_estimate', 'posterior'])
    parser.add_argument('--dirichlet-alpha', type=float, default=1.0)
    parser.add_argument('--n-epochs',        type=int,   default=1000)
    parser.add_argument('--hidden-dim',      type=int,   default=64)
    parser.add_argument('--n-layers',        type=int,   default=4)
    parser.add_argument('--lr',              type=float, default=5e-4)
    parser.add_argument('--ema-decay',       type=float, default=0.999)
    parser.add_argument('--loss-type',       type=str,   default='rate_kl',
                        choices=['rate_kl', 'mse'])
    parser.add_argument('--n-starts',        type=int,   default=5,
                        help='Dirichlet starts per training snapshot')
    parser.add_argument('--n-samples',       type=int,   default=15000)
    parser.add_argument('--posterior-k',     type=int,   default=20)
    parser.add_argument('--n-test',          type=int,   default=None,
                        help='Cap number of test snapshots (default: all)')
    parser.add_argument('--obs-sweep',       action='store_true',
                        help='Sweep obs fractions 0.1/0.2/0.3/0.5 at test time')
    parser.add_argument('--seed',            type=int,   default=42)
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
            f"Data not found: {data_path}\n"
            "Run: uv run experiments/ex15a_bikeshare_setup.py")

    print(f"=== Experiment 15b: Bike Sharing Interpolation ===\n")
    data = np.load(data_path, allow_pickle=True)

    R           = data['R']
    positions   = data['positions']
    train_dists = data['train_dists']
    train_hours = data['train_hours']
    train_dows  = data['train_dows']
    test_dists  = data['test_dists']
    test_hours  = data['test_hours']
    test_dows   = data['test_dows']
    hist_mean   = data['hist_mean']     # (24, 7, N)

    N = R.shape[0]
    edge_index = rate_matrix_to_edge_index(R)

    n_test = args.n_test or len(test_dists)
    n_test = min(n_test, len(test_dists))
    test_dists  = test_dists[:n_test]
    test_hours  = test_hours[:n_test]
    test_dows   = test_dows[:n_test]

    print(f"Stations: {N}, Edges: {int((R != 0).sum() // 2)}")
    print(f"Train: {len(train_dists)} snapshots, Test: {n_test} snapshots")
    print(f"Obs fraction: {args.obs_fraction:.0%}, Mode: {args.mode}")

    # ── Build dataset ──────────────────────────────────────────────────────────
    ckpt = os.path.join(CKPT_DIR,
        f'meta_model_ex15b_interp_{args.n_epochs}ep'
        f'_h{args.hidden_dim}_l{args.n_layers}.pt')

    model = FiLMConditionalGNNRateMatrixPredictor(
        node_context_dim=2,   # [observed_value * mask, mask]
        global_dim=N,         # raw observation vector (60-dim)
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers)

    losses = None
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
        print(f"\nLoaded checkpoint from {ckpt}")
    else:
        print(f"\nBuilding BikeInterpolationDataset ({args.n_samples} samples)...")
        n_starts = args.n_starts if args.mode == 'posterior' else 1
        dataset = BikeInterpolationDataset(
            R, train_dists,
            obs_fraction=args.obs_fraction,
            mode='dirichlet' if args.mode == 'posterior' else 'uniform',
            dirichlet_alpha=args.dirichlet_alpha,
            n_starts_per_dist=n_starts,
            n_samples=args.n_samples,
            seed=args.seed)
        print(f"  Dataset: {len(dataset)} samples")

        print(f"Training ({args.n_epochs} epochs, lr={args.lr}, "
              f"hidden={args.hidden_dim}, layers={args.n_layers})...")
        history = train_film_conditional(
            model, dataset,
            n_epochs=args.n_epochs, batch_size=256, lr=args.lr,
            device=device, loss_weighting='uniform',
            loss_type=args.loss_type, ema_decay=args.ema_decay)
        losses = history['losses']
        print(f"  Loss: {losses[0]:.6f} → {losses[-1]:.6f}")
        torch.save(model.state_dict(), ckpt)
        print(f"  Saved to {ckpt}")

    model = model.to(device)
    model.eval()

    # ── Evaluate ───────────────────────────────────────────────────────────────
    K = args.posterior_k if args.mode == 'posterior' else 1
    print(f"\nEvaluating on {n_test} test snapshots (K={K} samples)...")
    results = eval_interpolation(
        model, edge_index, test_dists, test_hours, test_dows,
        R, positions, train_dists, train_hours, train_dows,
        hist_mean, device,
        obs_fraction=args.obs_fraction, K=K, seed=99)

    # ── Obs fraction sweep ────────────────────────────────────────────────────
    sweep = None
    if args.obs_sweep:
        print("\nRunning observation fraction sweep (0.1, 0.2, 0.3, 0.5)...")
        sweep = eval_obs_fraction_sweep(
            model, edge_index, test_dists, test_hours, test_dows,
            R, positions, train_dists, train_hours, train_dows,
            hist_mean, device)

    print_results(results, args.obs_fraction, sweep=sweep)

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_main_figure(
        results, positions, losses=losses, sweep=sweep,
        out_path=os.path.join(HERE, 'ex15b_interpolation.png'))

    if args.mode == 'posterior':
        plot_posterior_figure(
            results, positions,
            out_path=os.path.join(HERE, 'ex15b_posterior.png'))

    print("\nDone.")


if __name__ == '__main__':
    main()
