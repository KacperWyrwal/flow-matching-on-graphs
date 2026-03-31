"""
Experiment 14b: EEG Model Training & Evaluation

Loads precomputed graph + leadfield from ex14_eeg_data.npz (no MNE needed).
Trains FiLM-conditioned flow matching model to reconstruct cortical source
distributions from EEG observations.

Modes:
  point_estimate  -- uniform start → source, FiLM-conditioned on EEG + tau
  posterior       -- Dirichlet start → source, for uncertainty quantification
  both            -- train and evaluate both

Run: uv run experiments/ex14b_eeg_train.py [--mode both] [--n-epochs 1000]
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
from sklearn.linear_model import Lasso

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
    DirectGNNPredictor,
    train_film_conditional,
    train_direct_gnn,
    sample_trajectory_film,
    get_device,
    EMA,
)
from meta_fm.model import rate_matrix_to_edge_index

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT_DIR = os.path.join(ROOT, 'checkpoints')


# ── Source model ──────────────────────────────────────────────────────────────

def make_cortical_source(n_parcels, n_peaks, rng):
    peak_parcels = rng.choice(n_parcels, size=n_peaks, replace=False).tolist()
    weights = rng.dirichlet(np.full(n_peaks, 2.0))
    dist = np.ones(n_parcels) * 0.2 / n_parcels
    for p, w in zip(peak_parcels, weights):
        dist[p] += 0.8 * w
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist, peak_parcels


# ── Forward model helpers ─────────────────────────────────────────────────────

def generate_eeg_training_data(R, A, n_dists=300, tau_range=(0.20, 0.60),
                                snr_db=20, seed=42):
    """
    Generate training pairs: (mu_source, y, tau_diff).
    Returns list of dicts with keys: mu_source, y, tau_diff, peak_parcels, n_peaks.
    """
    rng = np.random.default_rng(seed)
    N = R.shape[0]
    pairs = []
    per_count = n_dists // 3
    for n_peaks in [1, 2, 3]:
        for _ in range(per_count):
            mu_source, peak_parcels = make_cortical_source(N, n_peaks, rng)
            tau_diff = float(rng.uniform(*tau_range))
            mu_diffused = mu_source @ expm(tau_diff * R)
            y_clean = A @ mu_diffused
            if snr_db is not None and snr_db < 100:
                signal_power = np.mean(y_clean ** 2)
                noise_power = signal_power / (10 ** (snr_db / 10))
                y = y_clean + rng.normal(0, np.sqrt(noise_power), len(y_clean))
            else:
                y = y_clean
            pairs.append({
                'mu_source':   mu_source,
                'y':           y,
                'tau_diff':    tau_diff,
                'peak_parcels': peak_parcels,
                'n_peaks':     n_peaks,
            })
    return pairs


# ── Electrode → parcel mapping ────────────────────────────────────────────────

def compute_electrode_parcels(A):
    """
    Map each electrode to the parcel it sees most strongly (argmax per row).
    Returns (M,) int array.
    """
    return np.argmax(A, axis=1)


def build_eeg_context_spatial(y, electrode_parcels, N, tau_diff):
    """
    Build per-node and global context for EEG.

    electrode_parcels: (M,) int — maps electrode m → nearest parcel index.

    Returns:
        node_context: (N, 2) — [mean |y| for electrodes assigned here, is_assigned]
        global_cond:  (M+1,) — [y, tau_diff]
    """
    parcel_val   = np.zeros(N)
    parcel_count = np.zeros(N, dtype=int)
    for m, p in enumerate(electrode_parcels):
        parcel_val[p]   += abs(y[m])
        parcel_count[p] += 1
    is_assigned = (parcel_count > 0).astype(float)
    parcel_val[is_assigned > 0] /= parcel_count[is_assigned > 0]
    node_context = np.stack([parcel_val * is_assigned, is_assigned], axis=-1)
    global_cond  = np.concatenate([y, [tau_diff]])
    return node_context, global_cond


# ── Baselines ─────────────────────────────────────────────────────────────────

def baseline_backproj_eeg(y, A):
    mu = A.T @ y
    mu = np.clip(mu, 0, None)
    s = mu.sum()
    return mu / s if s > 1e-12 else np.ones(A.shape[1]) / A.shape[1]


def baseline_mne_eeg(y, A, lam=1e-3):
    M = A.shape[0]
    mu = A.T @ np.linalg.solve(A @ A.T + lam * np.eye(M), y)
    mu = np.clip(mu, 0, None)
    s = mu.sum()
    return mu / s if s > 1e-12 else np.ones(A.shape[1]) / A.shape[1]


def baseline_sloreta(y, A, lam=1e-3):
    M, N = A.shape
    C = A @ A.T + lam * np.eye(M)
    W = A.T @ np.linalg.solve(C, np.eye(M))
    mu_raw = W @ y
    R_diag = np.einsum('ij,ji->i', W, A)
    R_diag = np.maximum(R_diag, 1e-12)
    mu = np.abs(mu_raw) / np.sqrt(R_diag)
    mu = np.clip(mu, 0, None)
    s = mu.sum()
    return mu / s if s > 1e-12 else np.ones(N) / N


def baseline_lasso_eeg(y, A, alpha=0.01):
    model = Lasso(alpha=alpha, positive=True, max_iter=5000, tol=1e-4)
    model.fit(A, y)
    mu = np.clip(model.coef_, 0, None)
    s = mu.sum()
    return mu / s if s > 1e-12 else np.ones(A.shape[1]) / A.shape[1]


def tune_baselines_eeg(val_pairs, A):
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1]
    alphas  = [1e-4, 1e-3, 1e-2, 1e-1]

    best_lam, best_lam_tv = lambdas[0], float('inf')
    for lam in lambdas:
        tvs = [total_variation(baseline_mne_eeg(p['y'], A, lam), p['mu_source'])
               for p in val_pairs]
        if np.mean(tvs) < best_lam_tv:
            best_lam_tv = np.mean(tvs)
            best_lam = lam

    best_alpha, best_alpha_tv = alphas[0], float('inf')
    for alpha in alphas:
        tvs = [total_variation(baseline_lasso_eeg(p['y'], A, alpha), p['mu_source'])
               for p in val_pairs]
        if np.mean(tvs) < best_alpha_tv:
            best_alpha_tv = np.mean(tvs)
            best_alpha = alpha

    return best_lam, best_alpha


# ── Metrics ───────────────────────────────────────────────────────────────────

def peak_recovery_topk(recovered, true_peaks):
    k = len(true_peaks)
    top_k = set(np.argsort(recovered)[-k:].tolist())
    return len(top_k & set(true_peaks)) / k


def hemi_accuracy(recovered, true_peaks, parcel_hemis):
    """
    Fraction of true peaks whose hemisphere is represented in top-k predictions.
    parcel_hemis: (N,) int array, 0=LH, 1=RH.
    """
    k = len(true_peaks)
    top_k = np.argsort(recovered)[-k:]
    true_hemis  = set(parcel_hemis[p] for p in true_peaks)
    pred_hemis  = set(parcel_hemis[p] for p in top_k)
    return len(true_hemis & pred_hemis) / len(true_hemis)


def net_accuracy(recovered, true_peaks, network_assignments):
    """
    Fraction of true peaks whose network is represented in top-k predictions.
    network_assignments: (N,) int array.
    """
    k = len(true_peaks)
    top_k = np.argsort(recovered)[-k:]
    true_nets = set(network_assignments[p] for p in true_peaks)
    pred_nets  = set(network_assignments[p] for p in top_k)
    return len(true_nets & pred_nets) / len(true_nets)


# ── Datasets ──────────────────────────────────────────────────────────────────

class EEGPointEstimateDataset(torch.utils.data.Dataset):
    """
    Flow: uniform → source, conditioned on EEG + tau_diff (FiLM).
    One OT coupling per training pair (uniform start).
    """

    def __init__(self, R, A, training_pairs, electrode_parcels,
                 n_samples=15000, seed=42):
        rng = np.random.default_rng(seed)
        N = R.shape[0]
        graph_struct = GraphStructure(R)
        cache        = GeodesicCache(graph_struct)
        edge_index   = rate_matrix_to_edge_index(R)
        mu_start     = np.ones(N) / N

        pairs = []
        for pair in training_pairs:
            node_ctx, global_ctx = build_eeg_context_spatial(
                pair['y'], electrode_parcels, N, pair['tau_diff'])
            pi = compute_ot_coupling(mu_start, pair['mu_source'], graph_struct=graph_struct)
            cache.precompute_for_coupling(pi)
            pairs.append((pair['mu_source'], node_ctx, global_ctx, pi))

        self.samples = []
        for _ in range(n_samples):
            mu_source, node_ctx, global_ctx, pi = \
                pairs[int(rng.integers(len(pairs)))]
            tau = float(rng.uniform(0.0, 0.999))
            mu_tau   = marginal_distribution_fast(cache, pi, tau)
            R_target = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi, tau)
            self.samples.append((
                torch.tensor(mu_tau,     dtype=torch.float32),
                torch.tensor([tau],      dtype=torch.float32),
                torch.tensor(node_ctx,   dtype=torch.float32),
                torch.tensor(global_ctx, dtype=torch.float32),
                torch.tensor(R_target,   dtype=torch.float32),
                edge_index,
                N,
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class EEGPosteriorDataset(torch.utils.data.Dataset):
    """
    Flow: Dirichlet(1,...,1) → source, conditioned on EEG + tau_diff (FiLM).
    Multiple Dirichlet starts per source pair for posterior diversity.
    """

    def __init__(self, R, A, training_pairs, electrode_parcels,
                 n_starts_per_pair=10, n_samples=15000, seed=42):
        rng = np.random.default_rng(seed)
        N = R.shape[0]
        graph_struct = GraphStructure(R)
        cache        = GeodesicCache(graph_struct)
        edge_index   = rate_matrix_to_edge_index(R)

        triples = []
        for pair in training_pairs:
            node_ctx, global_ctx = build_eeg_context_spatial(
                pair['y'], electrode_parcels, N, pair['tau_diff'])
            for _ in range(n_starts_per_pair):
                mu_start = rng.dirichlet(np.ones(N))
                pi = compute_ot_coupling(mu_start, pair['mu_source'], graph_struct=graph_struct)
                cache.precompute_for_coupling(pi)
                triples.append((pair['mu_source'], node_ctx, global_ctx, pi))

        self.samples = []
        for _ in range(n_samples):
            mu_source, node_ctx, global_ctx, pi = \
                triples[int(rng.integers(len(triples)))]
            tau = float(rng.uniform(0.0, 0.999))
            mu_tau   = marginal_distribution_fast(cache, pi, tau)
            R_target = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi, tau)
            self.samples.append((
                torch.tensor(mu_tau,     dtype=torch.float32),
                torch.tensor([tau],      dtype=torch.float32),
                torch.tensor(node_ctx,   dtype=torch.float32),
                torch.tensor(global_ctx, dtype=torch.float32),
                torch.tensor(R_target,   dtype=torch.float32),
                edge_index,
                N,
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_evaluation(model, R, A, edge_index, electrode_parcels,
                   parcel_hemis, network_assignments,
                   best_lam, best_alpha, device, seed=99,
                   snr_db=20.0, mode='point_estimate', K=20):
    """
    Run structured evaluation: 3 tau × 3 n_peaks × 10 = 90 test cases.
    Returns list of result dicts.
    """
    rng = np.random.default_rng(seed)
    N = R.shape[0]
    TAU_DIFFS   = [0.20, 0.40, 0.60]
    N_PEAKS_ALL = [1, 2, 3]
    N_PER_CELL  = 10
    results = []

    for td in TAU_DIFFS:
        for n_peaks in N_PEAKS_ALL:
            for _ in range(N_PER_CELL):
                mu_source, peak_parcels = make_cortical_source(N, n_peaks, rng)
                mu_diffused = mu_source @ expm(td * R)
                y_clean = A @ mu_diffused
                if snr_db < 100:
                    sp = np.mean(y_clean ** 2)
                    np_ = sp / (10 ** (snr_db / 10))
                    y = y_clean + rng.normal(0, np.sqrt(np_), len(y_clean))
                else:
                    y = y_clean

                node_ctx, global_ctx = build_eeg_context_spatial(
                    y, electrode_parcels, N, td)

                if mode == 'point_estimate':
                    mu_start = np.ones(N) / N
                    _, traj = sample_trajectory_film(
                        model, mu_start, node_ctx, global_ctx, edge_index,
                        n_steps=200, device=device)
                    mu_learned = traj[-1]
                    posterior_samples = None
                else:
                    samples = []
                    for _ in range(K):
                        mu_start = rng.dirichlet(np.ones(N))
                        _, traj = sample_trajectory_film(
                            model, mu_start, node_ctx, global_ctx, edge_index,
                            n_steps=200, device=device)
                        samples.append(traj[-1])
                    posterior_samples = samples
                    mu_learned = np.mean(samples, axis=0)

                mu_mne     = baseline_mne_eeg(y, A, lam=best_lam)
                mu_sloreta = baseline_sloreta(y, A, lam=best_lam)
                mu_lasso   = baseline_lasso_eeg(y, A, alpha=best_alpha)
                mu_bp      = baseline_backproj_eeg(y, A)

                r = {
                    'mu_source':    mu_source,
                    'mu_learned':   mu_learned,
                    'mu_mne':       mu_mne,
                    'mu_sloreta':   mu_sloreta,
                    'mu_lasso':     mu_lasso,
                    'mu_bp':        mu_bp,
                    'y':            y,
                    'tau_diff':     td,
                    'n_peaks':      n_peaks,
                    'peak_parcels': peak_parcels,
                    'tv_learned':   total_variation(mu_learned,  mu_source),
                    'tv_mne':       total_variation(mu_mne,      mu_source),
                    'tv_sloreta':   total_variation(mu_sloreta,  mu_source),
                    'tv_lasso':     total_variation(mu_lasso,    mu_source),
                    'tv_bp':        total_variation(mu_bp,       mu_source),
                    'pk_learned':   peak_recovery_topk(mu_learned,  peak_parcels),
                    'pk_mne':       peak_recovery_topk(mu_mne,      peak_parcels),
                    'pk_sloreta':   peak_recovery_topk(mu_sloreta,  peak_parcels),
                    'pk_lasso':     peak_recovery_topk(mu_lasso,    peak_parcels),
                    'pk_bp':        peak_recovery_topk(mu_bp,       peak_parcels),
                    'hemi_learned': hemi_accuracy(mu_learned,  peak_parcels, parcel_hemis),
                    'hemi_mne':     hemi_accuracy(mu_mne,      peak_parcels, parcel_hemis),
                    'hemi_sloreta': hemi_accuracy(mu_sloreta,  peak_parcels, parcel_hemis),
                    'hemi_lasso':   hemi_accuracy(mu_lasso,    peak_parcels, parcel_hemis),
                    'hemi_bp':      hemi_accuracy(mu_bp,       peak_parcels, parcel_hemis),
                    'net_learned':  net_accuracy(mu_learned,   peak_parcels, network_assignments),
                    'net_mne':      net_accuracy(mu_mne,       peak_parcels, network_assignments),
                    'net_sloreta':  net_accuracy(mu_sloreta,   peak_parcels, network_assignments),
                    'net_lasso':    net_accuracy(mu_lasso,     peak_parcels, network_assignments),
                    'net_bp':       net_accuracy(mu_bp,        peak_parcels, network_assignments),
                }

                if posterior_samples is not None:
                    post_arr = np.array(posterior_samples)
                    post_mean = post_arr.mean(axis=0)
                    post_std  = post_arr.std(axis=0)
                    err = np.abs(post_mean - mu_source)
                    r['posterior_samples'] = posterior_samples
                    r['posterior_mean']    = post_mean
                    r['posterior_std']     = post_std
                    r['post_tv']    = total_variation(post_mean, mu_source)
                    # Calibration: per-parcel correlation of std vs error
                    if post_std.std() > 1e-8:
                        r['calib_r'] = float(np.corrcoef(post_std, err)[0, 1])
                    else:
                        r['calib_r'] = 0.0
                    # Diversity: mean pairwise TV
                    divs = [total_variation(posterior_samples[i], posterior_samples[j])
                            for i in range(K) for j in range(i+1, K)]
                    r['diversity'] = float(np.mean(divs))

                results.append(r)

    return results


def run_noise_sweep(model, R, A, edge_index, electrode_parcels,
                    parcel_hemis, network_assignments,
                    best_lam, best_alpha, device, snr_levels=None,
                    seed=77, K=10):
    """Evaluate at multiple SNR levels, return dict snr → mean metrics."""
    if snr_levels is None:
        snr_levels = [float('inf'), 20, 10, 5]
    N = R.shape[0]
    rng = np.random.default_rng(seed)
    # Generate 30 fixed source cases, apply different noise levels
    cases = []
    for _ in range(30):
        n_peaks = int(rng.integers(1, 4))
        mu_src, peaks = make_cortical_source(N, n_peaks, rng)
        td = float(rng.uniform(0.05, 0.20))
        mu_diff = mu_src @ expm(td * R)
        y_clean = A @ mu_diff
        cases.append((mu_src, peaks, td, y_clean))

    sweep = {}
    for snr in snr_levels:
        tvs = {'learned': [], 'sloreta': [], 'mne': [], 'lasso': [], 'bp': []}
        for mu_src, peaks, td, y_clean in cases:
            if snr < 100:
                sp = np.mean(y_clean ** 2)
                np_ = sp / (10 ** (snr / 10))
                y = y_clean + rng.normal(0, np.sqrt(np_), len(y_clean))
            else:
                y = y_clean
            node_ctx, global_ctx = build_eeg_context_spatial(
                y, electrode_parcels, N, td)
            mu_start = np.ones(N) / N
            _, traj = sample_trajectory_film(
                model, mu_start, node_ctx, global_ctx, edge_index,
                n_steps=100, device=device)
            mu_learned = traj[-1]
            tvs['learned'].append(total_variation(mu_learned, mu_src))
            tvs['sloreta'].append(total_variation(baseline_sloreta(y, A, best_lam), mu_src))
            tvs['mne'].append(total_variation(baseline_mne_eeg(y, A, best_lam), mu_src))
            tvs['lasso'].append(total_variation(baseline_lasso_eeg(y, A, best_alpha), mu_src))
            tvs['bp'].append(total_variation(baseline_backproj_eeg(y, A), mu_src))
        sweep[snr] = {k: float(np.mean(v)) for k, v in tvs.items()}
    return sweep


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    'Learned':    '#2166ac',
    'sLORETA':    '#4363d8',
    'MNE':        '#3cb44b',
    'LASSO':      '#f58231',
    'Backproj':   '#e6194b',
}


def _parcel_scatter(ax, parcel_centroids, vals, title='', cmap='hot',
                    vmin=0, vmax=None):
    """Scatter plot of parcel centroids coloured by value (axial view)."""
    x = parcel_centroids[:, 0] * 1e3
    y = parcel_centroids[:, 1] * 1e3
    vm = vmax if vmax is not None else vals.max()
    sc = ax.scatter(x, y, c=vals, cmap=cmap, vmin=vmin, vmax=vm,
                    s=55, edgecolors='k', linewidths=0.3)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)
    return sc


def plot_main_figure(results, parcel_centroids, network_names,
                     losses=None, sweep=None, mode='point_estimate',
                     out_path=None):
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f'Experiment 14b: EEG Source Reconstruction (mode={mode})\n'
        f'Flow Matching on 100-parcel Schaefer Cortical Graph, 64-channel EEG',
        fontsize=11)
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.45)

    # Panel A: Training loss
    ax_A = fig.add_subplot(gs[0, 0])
    if losses is not None:
        ax_A.plot(losses, color='steelblue', lw=1.0, alpha=0.6, label='loss')
        # EMA smoothed overlay
        if len(losses) > 20:
            k = max(1, len(losses) // 50)
            smooth = np.convolve(losses, np.ones(k)/k, mode='valid')
            ax_A.plot(np.arange(k-1, len(losses)), smooth,
                      color='red', lw=1.5, label='smoothed')
        ax_A.set_yscale('log')
        ax_A.legend(fontsize=7)
        ax_A.set_xlabel('Epoch')
        ax_A.set_ylabel('Loss')
        ax_A.set_title('A: Training Loss', fontsize=9)
    else:
        ax_A.text(0.5, 0.5, 'Loaded from\ncheckpoint',
                  transform=ax_A.transAxes, ha='center', va='center', fontsize=12)
        ax_A.set_title('A: Training Loss', fontsize=9)
    ax_A.grid(True, alpha=0.3)

    # Panel B: Brain surface scatter (one representative test case)
    # Pick a 1-peak case at tau=1.0
    example = next(
        (r for r in results if r['n_peaks'] == 1 and abs(r['tau_diff'] - 1.0) < 0.01),
        results[0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_B.axis('off')
    ax_B.set_title('B: Brain Surface Reconstruction (axial)', fontsize=9, pad=4)

    inner_B = gs[0, 1].subgridspec(3, 1, hspace=0.6)
    vm_B = max(example['mu_source'].max(), 0.05)
    for row_i, (key, label, col_c) in enumerate([
        ('mu_source',  'True source',  'black'),
        ('mu_learned', 'Learned',      '#2166ac'),
        ('mu_sloreta', 'sLORETA',      '#4363d8'),
    ]):
        ax_bb = fig.add_subplot(inner_B[row_i])
        sc = _parcel_scatter(ax_bb, parcel_centroids, example[key],
                             vmin=0, vmax=vm_B, cmap='hot')
        peaks = example['peak_parcels']
        ax_bb.scatter(parcel_centroids[peaks, 0] * 1e3,
                      parcel_centroids[peaks, 1] * 1e3,
                      s=120, marker='*', c='cyan', zorder=5,
                      edgecolors='k', linewidths=0.5)
        ax_bb.set_ylabel(label, fontsize=7, rotation=0, labelpad=38, va='center',
                         color=col_c)
        if row_i == 0:
            tv_str = ''
        else:
            tv_key = key.replace('mu_', 'tv_')
            tv_str = f' (TV={example[tv_key]:.3f})'
        ax_bb.set_title(label + tv_str, fontsize=7, color=col_c)

    # Panel C: TV comparison bar chart
    ax_C = fig.add_subplot(gs[0, 2])
    methods_tv = [
        ('tv_learned',  'Learned',  '#2166ac'),
        ('tv_sloreta',  'sLORETA',  '#4363d8'),
        ('tv_mne',      'MNE',      '#3cb44b'),
        ('tv_lasso',    'LASSO',    '#f58231'),
        ('tv_bp',       'Backproj', '#e6194b'),
    ]
    means_tv = [np.mean([r[k] for r in results]) for k, _, _ in methods_tv]
    stds_tv  = [np.std( [r[k] for r in results]) for k, _, _ in methods_tv]
    labels_tv = [l for _, l, _ in methods_tv]
    colors_tv = [c for _, _, c in methods_tv]
    x_tv = np.arange(len(methods_tv))
    ax_C.bar(x_tv, means_tv, yerr=stds_tv, capsize=4, color=colors_tv,
             edgecolor='k', linewidth=0.4, alpha=0.85)
    ax_C.set_xticks(x_tv)
    ax_C.set_xticklabels(labels_tv, rotation=20, ha='right', fontsize=8)
    ax_C.set_ylabel('Total Variation')
    ax_C.set_title('C: TV Comparison (90 test cases)', fontsize=9)
    ax_C.grid(True, alpha=0.3, axis='y')

    # Panel D: Network accuracy bar chart
    ax_D = fig.add_subplot(gs[1, 0])
    methods_net = [
        ('net_learned', 'Learned',  '#2166ac'),
        ('net_sloreta', 'sLORETA',  '#4363d8'),
        ('net_mne',     'MNE',      '#3cb44b'),
        ('net_lasso',   'LASSO',    '#f58231'),
        ('net_bp',      'Backproj', '#e6194b'),
    ]
    means_net = [np.mean([r[k] for r in results]) * 100 for k, _, _ in methods_net]
    labels_net = [l for _, l, _ in methods_net]
    colors_net = [c for _, _, c in methods_net]
    x_net = np.arange(len(methods_net))
    ax_D.bar(x_net, means_net, color=colors_net,
             edgecolor='k', linewidth=0.4, alpha=0.85)
    ax_D.set_xticks(x_net)
    ax_D.set_xticklabels(labels_net, rotation=20, ha='right', fontsize=8)
    ax_D.set_ylabel('Network accuracy (%)')
    ax_D.set_ylim(0, 115)
    ax_D.set_title('D: Network Accuracy (correct of 7)', fontsize=9)
    ax_D.grid(True, alpha=0.3, axis='y')

    # Panel E: Calibration scatter (posterior) or TV by tau (point estimate)
    ax_E = fig.add_subplot(gs[1, 1])
    if mode in ('posterior', 'both') and 'calib_r' in results[0]:
        all_std = np.concatenate([r['posterior_std'] for r in results])
        all_err = np.concatenate([
            np.abs(r['posterior_mean'] - r['mu_source']) for r in results])
        ax_E.scatter(all_std, all_err, alpha=0.01, s=2, c='steelblue')
        r_val = float(np.mean([r.get('calib_r', 0) for r in results]))
        ax_E.set_xlabel('Posterior std')
        ax_E.set_ylabel('|Posterior mean − true|')
        ax_E.set_title(f'E: Calibration (mean r = {r_val:.3f})', fontsize=9)
        ax_E.grid(True, alpha=0.3)
    else:
        # TV by tau_diff
        taus = sorted(set(r['tau_diff'] for r in results))
        for key, label, color in methods_tv:
            tv_by_tau = [np.mean([r[key] for r in results if abs(r['tau_diff']-td) < 0.01])
                         for td in taus]
            ax_E.plot(taus, tv_by_tau, 'o-', label=label, color=color, lw=1.5)
        ax_E.set_xlabel('τ_diff')
        ax_E.set_ylabel('Mean TV')
        ax_E.set_title('E: TV vs Diffusion Time', fontsize=9)
        ax_E.legend(fontsize=7)
        ax_E.grid(True, alpha=0.3)

    # Panel F: Noise robustness
    ax_F = fig.add_subplot(gs[1, 2])
    if sweep is not None:
        snr_vals = sorted(sweep.keys(), reverse=True)
        snr_labels = [f'{s} dB' if s < 100 else '∞ dB' for s in snr_vals]
        for key, label, color in [
            ('learned', 'Learned', '#2166ac'),
            ('sloreta', 'sLORETA', '#4363d8'),
            ('mne',     'MNE',     '#3cb44b'),
            ('lasso',   'LASSO',   '#f58231'),
            ('bp',      'Backproj','#e6194b'),
        ]:
            tv_vals = [sweep[s][key] for s in snr_vals]
            ax_F.plot(range(len(snr_vals)), tv_vals, 'o-',
                      label=label, color=color, lw=1.5)
        ax_F.set_xticks(range(len(snr_vals)))
        ax_F.set_xticklabels(snr_labels, fontsize=8)
        ax_F.set_ylabel('Mean TV')
        ax_F.set_title('F: TV vs SNR', fontsize=9)
        ax_F.legend(fontsize=7)
        ax_F.grid(True, alpha=0.3)
    else:
        # TV by n_peaks
        n_peaks_all = sorted(set(r['n_peaks'] for r in results))
        for key, label, color in methods_tv:
            tv_by_np = [np.mean([r[key] for r in results if r['n_peaks'] == np_])
                        for np_ in n_peaks_all]
            ax_F.plot(n_peaks_all, tv_by_np, 'o-', label=label, color=color, lw=1.5)
        ax_F.set_xlabel('Number of peaks')
        ax_F.set_ylabel('Mean TV')
        ax_F.set_title('F: TV vs Source Complexity', fontsize=9)
        ax_F.legend(fontsize=7)
        ax_F.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")
    return fig


def plot_posterior_figure(results, parcel_centroids, network_names,
                          out_path=None):
    """Supplementary figure: posterior analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Experiment 14b: Posterior Analysis', fontsize=11)

    # Select 3 representative cases (1-peak, 2-peak, 3-peak at tau=1.0)
    cases = [next((r for r in results if r['n_peaks'] == k and
                   abs(r['tau_diff'] - 1.0) < 0.01), results[k-1])
             for k in [1, 2, 3]]

    # Row 0: posterior mean for each case
    for col, r in enumerate(cases):
        ax = axes[0, col]
        vm = max(r['mu_source'].max(), 0.05)
        _parcel_scatter(ax, parcel_centroids, r['posterior_mean'],
                        vmin=0, vmax=vm, cmap='hot')
        peaks = r['peak_parcels']
        ax.scatter(parcel_centroids[peaks, 0] * 1e3,
                   parcel_centroids[peaks, 1] * 1e3,
                   s=120, marker='*', c='cyan', zorder=5,
                   edgecolors='k', linewidths=0.5)
        ax.set_title(f'{r["n_peaks"]}-peak, τ={r["tau_diff"]:.1f}\n'
                     f'TV={r["post_tv"]:.3f}, calib r={r.get("calib_r",0):.2f}',
                     fontsize=8)
        if col == 0:
            ax.set_ylabel('Posterior mean', fontsize=8)

    # Row 1: posterior std for each case
    for col, r in enumerate(cases):
        ax = axes[1, col]
        _parcel_scatter(ax, parcel_centroids, r['posterior_std'],
                        vmin=0, cmap='viridis')
        ax.set_title(f'Diversity={r.get("diversity", 0):.3f}', fontsize=8)
        if col == 0:
            ax.set_ylabel('Posterior std', fontsize=8)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")
    return fig


def print_results(results, label, mode='point_estimate'):
    def mr(key):
        vals = [r[key] for r in results]
        return np.mean(vals), np.std(vals)

    methods = [
        ('tv_learned',   'pk_learned',   'hemi_learned', 'net_learned',  'Learned'),
        ('tv_sloreta',   'pk_sloreta',   'hemi_sloreta', 'net_sloreta',  'sLORETA'),
        ('tv_mne',       'pk_mne',       'hemi_mne',     'net_mne',      'MNE'),
        ('tv_lasso',     'pk_lasso',     'hemi_lasso',   'net_lasso',    'LASSO'),
        ('tv_bp',        'pk_bp',        'hemi_bp',      'net_bp',       'Backproj'),
    ]

    print(f"\n{label} results ({len(results)} test cases):")
    print(f"  {'Method':12s} {'TV':>12s}  {'Peak%':>6s}  {'Hemi%':>6s}  {'Net%':>6s}")
    print(f"  {'-'*50}")
    for tv_k, pk_k, hemi_k, net_k, name in methods:
        tv_m, tv_s = mr(tv_k)
        pk_m = np.mean([r[pk_k] for r in results]) * 100
        hemi_m = np.mean([r[hemi_k] for r in results]) * 100
        net_m  = np.mean([r[net_k]  for r in results]) * 100
        print(f"  {name:12s}: {tv_m:.4f}±{tv_s:.4f}  {pk_m:5.0f}%  {hemi_m:5.0f}%  {net_m:5.0f}%")

    if mode in ('posterior', 'both') and 'calib_r' in results[0]:
        calib_r = np.mean([r['calib_r'] for r in results])
        div_m   = np.mean([r['diversity'] for r in results])
        post_tv = np.mean([r['post_tv'] for r in results])
        print(f"\n  Posterior:")
        print(f"    Mean TV (posterior mean): {post_tv:.4f}")
        print(f"    Calibration r:            {calib_r:.3f}")
        print(f"    Diversity (mean pairwise TV): {div_m:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='point_estimate',
                        choices=['point_estimate', 'posterior', 'both'])
    parser.add_argument('--data-path', type=str, default='ex14_eeg_data.npz')
    parser.add_argument('--n-epochs',        type=int,   default=1000)
    parser.add_argument('--hidden-dim',      type=int,   default=128)
    parser.add_argument('--n-layers',        type=int,   default=6)
    parser.add_argument('--lr',              type=float, default=5e-4)
    parser.add_argument('--ema-decay',       type=float, default=0.999)
    parser.add_argument('--loss-type',       type=str,   default='rate_kl',
                        choices=['rate_kl', 'mse'],
                        help='rate_kl: principled path-measure KL; mse: fallback')
    parser.add_argument('--n-train-dists',   type=int,   default=300)
    parser.add_argument('--n-samples',       type=int,   default=15000)
    parser.add_argument('--snr-db',          type=float, default=20.0)
    parser.add_argument('--posterior-k',     type=int,   default=20)
    parser.add_argument('--n-starts-per-pair', type=int, default=10)
    parser.add_argument('--train-direct-gnn', action='store_true')
    parser.add_argument('--noise-sweep',     action='store_true',
                        help='Evaluate at multiple SNR levels')
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(42)
    device = get_device()

    # ── Load data ─────────────────────────────────────────────────────────────
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(ROOT, data_path)
    print(f"=== Experiment 14b: EEG Model Training ===\n")
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    R    = data['R']
    A    = data['A']
    adj  = data['adj']
    parcel_centroids   = data['parcel_centroids']
    parcel_names       = data['parcel_names']
    network_assignments = data['network_assignments'].astype(int)
    network_names      = data['network_names']
    N = R.shape[0]
    n_channels = A.shape[0]

    # Hemisphere assignment from parcel names
    parcel_hemis = np.array(
        [0 if '_LH_' in str(n) else 1 for n in parcel_names], dtype=int)

    edge_index = rate_matrix_to_edge_index(R)
    print(f"Graph: {N} parcels, {edge_index.shape[1]//2} edges, {n_channels} EEG channels")
    print(f"Mode: {args.mode}")

    # Electrode → parcel assignment
    electrode_parcels = compute_electrode_parcels(A)
    n_covered = len(set(electrode_parcels.tolist()))
    print(f"Electrode coverage: {n_covered}/{N} parcels assigned at least one electrode")

    global_dim = n_channels + 1  # raw EEG vector + tau_diff

    # ── Generate training data ─────────────────────────────────────────────────
    print(f"\nGenerating {args.n_train_dists} training distributions (SNR={args.snr_db} dB)...")
    training_pairs = generate_eeg_training_data(
        R, A,
        n_dists=args.n_train_dists,
        tau_range=(0.20, 0.60),   # diagnostics: LASSO TV>0.65 at τ>0.20; equiv to cube τ∈[0.5,1.0]
        snr_db=args.snr_db,
        seed=42)

    # Baseline tuning on first 20 training cases
    print("Tuning baselines...")
    best_lam, best_alpha = tune_baselines_eeg(training_pairs[:20], A)
    print(f"  Best MNE lambda: {best_lam}, LASSO alpha: {best_alpha}")

    # ── Helper to build / load a model ────────────────────────────────────────
    def _ckpt_path(suffix):
        return os.path.join(
            CKPT_DIR,
            f'meta_model_ex14b_{suffix}_{args.n_epochs}ep'
            f'_h{args.hidden_dim}_l{args.n_layers}.pt')

    def _make_model():
        return FiLMConditionalGNNRateMatrixPredictor(
            node_context_dim=2,
            global_dim=global_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers)

    def _train_or_load(tag, dataset, ckpt):
        model = _make_model()
        losses_out = None
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location='cpu',
                                              weights_only=True))
            print(f"  Loaded checkpoint from {ckpt}")
        else:
            print(f"  Training ({args.n_epochs} epochs)...")
            history = train_film_conditional(
                model, dataset,
                n_epochs=args.n_epochs,
                batch_size=256,
                lr=args.lr,
                device=device,
                loss_weighting='uniform',
                loss_type=args.loss_type,
                ema_decay=args.ema_decay)
            losses_out = history['losses']
            print(f"  Initial loss: {losses_out[0]:.6f}, "
                  f"final loss: {losses_out[-1]:.6f}")
            torch.save(model.state_dict(), ckpt)
            print(f"  Checkpoint saved to {ckpt}")
        model = model.to(device)
        model.eval()
        return model, losses_out

    results_point = None
    results_post  = None
    losses_point  = None
    losses_post   = None
    model_point   = None
    model_post    = None

    # ── Point estimate mode ───────────────────────────────────────────────────
    if args.mode in ('point_estimate', 'both'):
        print(f"\n--- Mode: Point Estimate ---")
        print(f"Building EEGPointEstimateDataset ({args.n_samples} samples)...")
        dataset_point = EEGPointEstimateDataset(
            R, A, training_pairs, electrode_parcels,
            n_samples=args.n_samples, seed=42)
        print(f"  Dataset: {len(dataset_point)} samples")

        model_point, losses_point = _train_or_load(
            'point', dataset_point, _ckpt_path('point'))

        print(f"\nEvaluating (90 test cases)...")
        results_point = run_evaluation(
            model_point, R, A, edge_index, electrode_parcels,
            parcel_hemis, network_assignments,
            best_lam, best_alpha, device,
            seed=99, snr_db=args.snr_db, mode='point_estimate')

        print_results(results_point, 'Point estimate', mode='point_estimate')

        sweep = None
        if args.noise_sweep:
            print("\nRunning noise sweep...")
            sweep = run_noise_sweep(
                model_point, R, A, edge_index, electrode_parcels,
                parcel_hemis, network_assignments, best_lam, best_alpha,
                device, seed=77)
            for snr, tvs in sorted(sweep.items(), reverse=True):
                snr_str = f'{snr:.0f} dB' if snr < 100 else '∞ dB'
                print(f"  SNR={snr_str}: learned={tvs['learned']:.4f}, "
                      f"sloreta={tvs['sloreta']:.4f}, mne={tvs['mne']:.4f}")

        plot_main_figure(
            results_point, parcel_centroids, network_names,
            losses=losses_point, sweep=sweep,
            mode='point_estimate',
            out_path=os.path.join(HERE, 'ex14b_point_estimate.png'))

    # ── Posterior mode ────────────────────────────────────────────────────────
    if args.mode in ('posterior', 'both'):
        print(f"\n--- Mode: Posterior Sampling ---")
        print(f"Building EEGPosteriorDataset "
              f"({args.n_train_dists} pairs × {args.n_starts_per_pair} starts, "
              f"{args.n_samples} samples)...")
        dataset_post = EEGPosteriorDataset(
            R, A, training_pairs, electrode_parcels,
            n_starts_per_pair=args.n_starts_per_pair,
            n_samples=args.n_samples, seed=42)
        print(f"  Dataset: {len(dataset_post)} samples")

        model_post, losses_post = _train_or_load(
            'posterior', dataset_post, _ckpt_path('posterior'))

        print(f"\nEvaluating with K={args.posterior_k} posterior samples...")
        results_post = run_evaluation(
            model_post, R, A, edge_index, electrode_parcels,
            parcel_hemis, network_assignments,
            best_lam, best_alpha, device,
            seed=99, snr_db=args.snr_db,
            mode='posterior', K=args.posterior_k)

        print_results(results_post, 'Posterior', mode='posterior')

        plot_main_figure(
            results_post, parcel_centroids, network_names,
            losses=losses_post, mode='posterior',
            out_path=os.path.join(HERE, 'ex14b_posterior.png'))

        plot_posterior_figure(
            results_post, parcel_centroids, network_names,
            out_path=os.path.join(HERE, 'ex14b_posterior_analysis.png'))

    # ── Direct GNN baseline ───────────────────────────────────────────────────
    if args.train_direct_gnn:
        print(f"\n--- Direct GNN Baseline ---")
        ckpt_direct = _ckpt_path('direct')
        direct_model = DirectGNNPredictor(
            context_dim=2, hidden_dim=args.hidden_dim, n_layers=args.n_layers)

        direct_losses = None
        if os.path.exists(ckpt_direct):
            direct_model.load_state_dict(
                torch.load(ckpt_direct, map_location='cpu', weights_only=True))
            print(f"  Loaded from {ckpt_direct}")
        else:
            train_pairs_direct = []
            for pair in training_pairs:
                mu_bp = baseline_backproj_eeg(pair['y'], A)
                ctx   = np.stack([mu_bp, np.full(N, pair['tau_diff'])], axis=-1)
                train_pairs_direct.append((ctx, pair['mu_source'], edge_index))
            print(f"  Training DirectGNN ({args.n_epochs} epochs)...")
            hist = train_direct_gnn(
                direct_model, train_pairs_direct,
                n_epochs=args.n_epochs, lr=args.lr,
                device=device, seed=0)
            direct_losses = hist['losses']
            print(f"  Initial KL={direct_losses[0]:.6f}, final={direct_losses[-1]:.6f}")
            torch.save(direct_model.state_dict(), ckpt_direct)
            print(f"  Saved to {ckpt_direct}")

        direct_model = direct_model.to(device)
        direct_model.eval()

    print("\nDone.")


if __name__ == '__main__':
    main()
