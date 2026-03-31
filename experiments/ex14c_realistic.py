"""
Experiment 14c: Realistic Simulated EEG with MNE

Uses MNE-simulated sources (Gaussian spatial spread + evoked time course) instead
of synthetic peaked distributions. No artificial diffusion step — EEG is generated
directly from source activation through the vertex-level leadfield.

Two-stage pipeline:
  Stage 1: MNE simulation  → ex14c_sims_*.npz  (cached)
  Stage 2: Flow matching   → train on cached (source, EEG) pairs

Run: uv run experiments/ex14c_realistic.py [--mode posterior] [--n-epochs 1000]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
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
    train_film_conditional,
    sample_trajectory_film,
    get_device,
    EMA,
)
from meta_fm.model import rate_matrix_to_edge_index

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT_DIR = os.path.join(ROOT, 'checkpoints')
DATA_DIR  = os.path.join(ROOT, 'data')

_PARC = 'Schaefer2018_100Parcels_7Networks_order'


# ── MNE setup helpers (only needed for simulation stage) ──────────────────────

def _setup_mne(subjects_dir, n_sensors=64, recompute_fwd=False):
    """
    Return (src, labels, fwd, info).
    Loads from cache in DATA_DIR when possible.
    """
    import mne

    src_path = os.path.join(DATA_DIR, 'ex14a_src.fif')
    fwd_path = os.path.join(DATA_DIR, 'ex14a_fwd.fif')
    bem_path = os.path.join(DATA_DIR, 'ex14a_bem.fif')

    # Source space
    if not recompute_fwd and os.path.exists(src_path):
        print(f"  Loading cached source space from {src_path}")
        src = mne.read_source_spaces(src_path, verbose=False)
    else:
        print("  Setting up oct6 source space...")
        src = mne.setup_source_space(
            'fsaverage', spacing='oct6',
            subjects_dir=subjects_dir, add_dist=False, verbose=False)
        mne.write_source_spaces(src_path, src, overwrite=True, verbose=False)

    # Labels
    labels = mne.read_labels_from_annot(
        'fsaverage', parc=_PARC,
        subjects_dir=subjects_dir, verbose=False)
    _SCHAEFER_7 = {'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default'}

    def _label_network(lbl):
        for net in _SCHAEFER_7:
            if net in lbl.name:
                return net
        return None

    labels = [l for l in labels if _label_network(l) is not None]

    # EEG info
    montage   = mne.channels.make_standard_montage('standard_1020')
    ch_names  = montage.ch_names[:n_sensors]
    info = mne.create_info(ch_names=ch_names, sfreq=256,
                           ch_types='eeg', verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        info.set_montage(montage, on_missing='ignore', verbose=False)

    # Forward solution
    if not recompute_fwd and os.path.exists(fwd_path):
        print(f"  Loading cached forward solution from {fwd_path}")
        fwd = mne.read_forward_solution(fwd_path, verbose=False)
        fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose=False)
    else:
        if not recompute_fwd and os.path.exists(bem_path):
            bem_sol = mne.read_bem_solution(bem_path)
        else:
            print("  Computing BEM model...")
            bem_model = mne.make_bem_model(
                'fsaverage', subjects_dir=subjects_dir,
                conductivity=(0.3, 0.006, 0.3))
            bem_sol = mne.make_bem_solution(bem_model)
            mne.write_bem_solution(bem_path, bem_sol, overwrite=True)
        print("  Computing forward solution...")
        fwd = mne.make_forward_solution(
            info, trans='fsaverage', src=src, bem=bem_sol,
            eeg=True, meg=False, mindist=5.0, verbose=False)
        fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose=False)
        mne.write_forward_solution(fwd_path, fwd, overwrite=True)

    return src, labels, fwd, info


# ── Simulation helpers ────────────────────────────────────────────────────────

def generate_realistic_source(labels, src, n_active=None, rng=None,
                               spatial_extent=10.0, sfreq=256, duration=0.3):
    """
    Generate one realistic source activation using MNE.

    Returns:
        stc: SourceEstimate with realistic activation
        active_labels: list of activated labels
        active_indices: int array of activated parcel indices (0-based into `labels`)
        peak_time: float, seconds
    """
    import mne

    if rng is None:
        rng = np.random.default_rng()
    if n_active is None:
        n_active = int(rng.integers(1, 4))

    n_parcels = len(labels)
    active_indices = rng.choice(n_parcels, size=n_active, replace=False)
    active_labels  = [labels[i] for i in active_indices]

    n_times   = int(sfreq * duration)
    times     = np.arange(n_times) / sfreq
    peak_time = float(rng.uniform(0.05, duration - 0.05))

    stc_data_lh = np.zeros((len(src[0]['vertno']), n_times))
    stc_data_rh = np.zeros((len(src[1]['vertno']), n_times))

    for label in active_labels:
        hemi_idx  = 0 if label.hemi == 'lh' else 1
        src_verts = src[hemi_idx]['vertno']

        label_mask       = np.isin(src_verts, label.vertices)
        label_src_indices = np.where(label_mask)[0]
        if len(label_src_indices) == 0:
            continue

        # Temporal envelope: Gaussian
        amplitude  = float(rng.uniform(5e-9, 50e-9))   # 5–50 nAm
        sigma_t    = float(rng.uniform(0.02, 0.08))
        time_course = amplitude * np.exp(
            -0.5 * ((times - peak_time) / sigma_t) ** 2)

        # Spatial envelope: Gaussian falloff from label centroid
        center_pos  = src[hemi_idx]['rr'][src_verts[label_src_indices]].mean(axis=0)
        all_pos     = src[hemi_idx]['rr'][src_verts]
        distances   = np.linalg.norm(all_pos - center_pos, axis=1)
        sigma_s     = spatial_extent / 1000.0           # mm → metres
        spatial_pat = np.exp(-0.5 * (distances / sigma_s) ** 2)

        if hemi_idx == 0:
            stc_data_lh += np.outer(spatial_pat, time_course)
        else:
            stc_data_rh += np.outer(spatial_pat, time_course)

    stc = mne.SourceEstimate(
        np.vstack([stc_data_lh, stc_data_rh]),
        vertices=[src[0]['vertno'], src[1]['vertno']],
        tmin=0, tstep=1.0 / sfreq)

    return stc, active_labels, active_indices, peak_time


def parcellate_stc(stc, labels, src):
    """Average |source| within each parcel → (n_parcels, n_times)."""
    n_lh    = len(src[0]['vertno'])
    n_times = stc.data.shape[1]
    parcel_tc = np.zeros((len(labels), n_times))

    for i, label in enumerate(labels):
        hemi_idx  = 0 if label.hemi == 'lh' else 1
        src_verts = src[hemi_idx]['vertno']
        label_mask = np.isin(src_verts, label.vertices)
        offset     = 0 if hemi_idx == 0 else n_lh
        indices    = np.where(label_mask)[0] + offset
        if len(indices) > 0:
            parcel_tc[i] = np.abs(stc.data[indices]).mean(axis=0)

    return parcel_tc


def stc_to_distribution(parcel_tc, time_idx):
    """Normalise parcel activity at `time_idx` to a probability distribution."""
    values = np.clip(parcel_tc[:, time_idx], 0, None)
    s = values.sum()
    return values / s if s > 1e-12 else np.ones(len(values)) / len(values)


def simulate_eeg_from_stc(stc, fwd, snr_db=20, rng=None):
    """Project STC through full vertex-level leadfield, add noise."""
    if rng is None:
        rng = np.random.default_rng()
    leadfield = fwd['sol']['data']          # (n_channels, n_dipoles)
    eeg_clean = leadfield @ stc.data        # (n_channels, n_times)
    if snr_db is not None and snr_db < 200:
        sig_pwr   = np.mean(eeg_clean ** 2)
        nse_pwr   = sig_pwr / (10 ** (snr_db / 10))
        eeg_clean = eeg_clean + rng.normal(0, np.sqrt(nse_pwr), eeg_clean.shape)
    return eeg_clean


def generate_training_data(labels, src, fwd, n_simulations=1000,
                            spatial_extent=10.0, snr_db=20, sfreq=256, seed=42):
    """
    Generate (source_distribution, EEG) pairs from MNE simulations.

    Returns list of dicts: mu_source, y, active_parcels, n_active, peak_time
    """
    rng   = np.random.default_rng(seed)
    pairs = []

    for i in range(n_simulations):
        stc, _, active_indices, peak_time = generate_realistic_source(
            labels, src, rng=rng, spatial_extent=spatial_extent, sfreq=sfreq)

        eeg       = simulate_eeg_from_stc(stc, fwd, snr_db=snr_db, rng=rng)
        parcel_tc = parcellate_stc(stc, labels, src)

        peak_sample = min(int(peak_time * sfreq), parcel_tc.shape[1] - 1)
        mu_source   = stc_to_distribution(parcel_tc, peak_sample)
        y           = eeg[:, peak_sample]

        pairs.append({
            'mu_source':     mu_source,
            'y':             y,
            'active_parcels': active_indices.tolist(),
            'n_active':      len(active_indices),
            'peak_time':     peak_time,
        })

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_simulations} simulations")

    return pairs


# ── Simulation cache ──────────────────────────────────────────────────────────

def save_simulation_cache(path, train_pairs, test_pairs, metadata):
    np.savez(
        path,
        train_mu_sources   = np.array([p['mu_source']     for p in train_pairs]),
        train_ys           = np.array([p['y']             for p in train_pairs]),
        train_active_parcels=np.array([p['active_parcels'] for p in train_pairs],
                                       dtype=object),
        train_n_active     = np.array([p['n_active']      for p in train_pairs]),
        test_mu_sources    = np.array([p['mu_source']     for p in test_pairs]),
        test_ys            = np.array([p['y']             for p in test_pairs]),
        test_active_parcels= np.array([p['active_parcels'] for p in test_pairs],
                                       dtype=object),
        test_n_active      = np.array([p['n_active']      for p in test_pairs]),
        **metadata,
    )


def _to_list(x):
    """Convert numpy array or list to plain list."""
    if hasattr(x, 'tolist'):
        return x.tolist()
    return list(x)


def load_simulation_cache(path):
    data = np.load(path, allow_pickle=True)
    train_pairs = [
        {'mu_source':      data['train_mu_sources'][i],
         'y':              data['train_ys'][i],
         'active_parcels': _to_list(data['train_active_parcels'][i]),
         'n_active':       int(data['train_n_active'][i])}
        for i in range(len(data['train_mu_sources']))
    ]
    test_pairs = [
        {'mu_source':      data['test_mu_sources'][i],
         'y':              data['test_ys'][i],
         'active_parcels': _to_list(data['test_active_parcels'][i]),
         'n_active':       int(data['test_n_active'][i])}
        for i in range(len(data['test_mu_sources']))
    ]
    return train_pairs, test_pairs


# ── Context building (no tau_diff) ────────────────────────────────────────────

def compute_electrode_parcels(A):
    """Map each electrode to its strongest parcel (argmax per row)."""
    return np.argmax(np.abs(A), axis=1)


def build_eeg_context_spatial(y, electrode_parcels, N):
    """
    Per-node context [mean |y| for electrodes mapped here, is_sensor],
    plus raw EEG as global conditioning.

    Returns:
        node_context: (N, 2)
        global_cond:  (n_channels,)  — raw EEG, no tau_diff
    """
    parcel_val   = np.zeros(N)
    parcel_count = np.zeros(N, dtype=int)
    for m, p in enumerate(electrode_parcels):
        parcel_val[p]   += abs(y[m])
        parcel_count[p] += 1
    is_assigned = (parcel_count > 0).astype(float)
    parcel_val[is_assigned > 0] /= parcel_count[is_assigned > 0]
    node_context = np.stack([parcel_val * is_assigned, is_assigned], axis=-1)
    return node_context, y.copy()


# ── Baselines ─────────────────────────────────────────────────────────────────

def baseline_backproj_eeg(y, A):
    mu = A.T @ y
    mu = np.clip(mu, 0, None)
    s  = mu.sum()
    return mu / s if s > 1e-12 else np.ones(A.shape[1]) / A.shape[1]


def baseline_mne_eeg(y, A, lam=1e-3):
    M  = A.shape[0]
    mu = A.T @ np.linalg.solve(A @ A.T + lam * np.eye(M), y)
    mu = np.clip(mu, 0, None)
    s  = mu.sum()
    return mu / s if s > 1e-12 else np.ones(A.shape[1]) / A.shape[1]


def baseline_sloreta(y, A, lam=1e-3):
    M, N = A.shape
    C    = A @ A.T + lam * np.eye(M)
    W    = A.T @ np.linalg.solve(C, np.eye(M))
    mu_raw = W @ y
    R_diag = np.einsum('ij,ji->i', W, A)
    R_diag = np.maximum(R_diag, 1e-12)
    mu     = np.abs(mu_raw) / np.sqrt(R_diag)
    mu     = np.clip(mu, 0, None)
    s      = mu.sum()
    return mu / s if s > 1e-12 else np.ones(N) / N


def baseline_lasso_eeg(y, A, alpha=0.01):
    clf = Lasso(alpha=alpha, positive=True, max_iter=5000, tol=1e-4)
    clf.fit(A, y)
    mu = np.clip(clf.coef_, 0, None)
    s  = mu.sum()
    return mu / s if s > 1e-12 else np.ones(A.shape[1]) / A.shape[1]


def tune_baselines_eeg(val_pairs, A):
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1]
    alphas  = [1e-4, 1e-3, 1e-2, 1e-1]

    best_lam, best_tv = lambdas[0], float('inf')
    for lam in lambdas:
        tv = np.mean([total_variation(baseline_mne_eeg(p['y'], A, lam), p['mu_source'])
                      for p in val_pairs])
        if tv < best_tv:
            best_tv, best_lam = tv, lam

    best_alpha, best_tv = alphas[0], float('inf')
    for alpha in alphas:
        tv = np.mean([total_variation(baseline_lasso_eeg(p['y'], A, alpha), p['mu_source'])
                      for p in val_pairs])
        if tv < best_tv:
            best_tv, best_alpha = tv, alpha

    return best_lam, best_alpha


# ── Metrics ───────────────────────────────────────────────────────────────────

def peak_recovery_topk(recovered, true_peaks):
    k     = max(1, len(true_peaks))
    top_k = set(np.argsort(recovered)[-k:].tolist())
    return len(top_k & set(true_peaks)) / k


def hemi_accuracy(recovered, true_peaks, parcel_hemis):
    k          = max(1, len(true_peaks))
    top_k      = np.argsort(recovered)[-k:]
    true_hemis = set(parcel_hemis[p] for p in true_peaks)
    pred_hemis = set(parcel_hemis[p] for p in top_k)
    return len(true_hemis & pred_hemis) / len(true_hemis)


def net_accuracy(recovered, true_peaks, network_assignments):
    k         = max(1, len(true_peaks))
    top_k     = np.argsort(recovered)[-k:]
    true_nets = set(network_assignments[p] for p in true_peaks)
    pred_nets = set(network_assignments[p] for p in top_k)
    return len(true_nets & pred_nets) / len(true_nets)


def entropy_ratio(dist):
    """H(dist) / H(uniform): 0=peaked, 1=flat."""
    N    = len(dist)
    p    = np.clip(dist, 1e-12, None)
    p    = p / p.sum()
    H    = -np.sum(p * np.log(p))
    H_u  = np.log(N)
    return H / H_u if H_u > 0 else 0.0


# ── Dataset ───────────────────────────────────────────────────────────────────

class RealisticEEGDataset(torch.utils.data.Dataset):
    """
    Flow: Dirichlet (or uniform) → source distribution.
    Conditioning: EEG via FiLM (no tau_diff).

    Returns per sample:
        mu_tau:       (N,)
        tau:          (1,)
        node_context: (N, 2)
        global_cond:  (n_channels,)
        R_target:     (N, N)
        edge_index:   (2, E)
        n_nodes:      int
    """

    def __init__(self, R, training_pairs, electrode_parcels,
                 mode='dirichlet', dirichlet_alpha=1.0,
                 n_starts_per_pair=10, n_samples=20000, seed=42):
        rng          = np.random.default_rng(seed)
        N            = R.shape[0]
        graph_struct = GraphStructure(R)
        cache        = GeodesicCache(graph_struct)
        self._edge_index = rate_matrix_to_edge_index(R)

        n_starts = n_starts_per_pair if mode == 'dirichlet' else 1

        triples = []
        for pair in training_pairs:
            mu_source = pair['mu_source']
            y         = pair['y']
            node_ctx, global_ctx = build_eeg_context_spatial(
                y, electrode_parcels, N)
            for _ in range(n_starts):
                if mode == 'dirichlet':
                    mu_start = rng.dirichlet(np.full(N, dirichlet_alpha))
                else:
                    mu_start = np.ones(N) / N
                pi = compute_ot_coupling(mu_start, mu_source, graph_struct=graph_struct)
                cache.precompute_for_coupling(pi)
                triples.append((mu_source, node_ctx, global_ctx, pi))

        print(f"  Precomputed {len(triples)} OT couplings")

        self.samples = []
        for _ in range(n_samples):
            mu_source, node_ctx, global_ctx, pi = \
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_evaluation(model, test_pairs, R, A, edge_index, electrode_parcels,
                   parcel_hemis, network_assignments,
                   best_lam, best_alpha, device,
                   mode='posterior', K=20, seed=99):
    """
    Evaluate on held-out MNE simulation pairs.
    Returns list of result dicts.
    """
    rng    = np.random.default_rng(seed)
    N      = R.shape[0]
    results = []

    for pair in test_pairs:
        mu_source     = pair['mu_source']
        y             = pair['y']
        active_parcels = pair['active_parcels']
        n_active      = pair['n_active']

        node_ctx, global_ctx = build_eeg_context_spatial(y, electrode_parcels, N)

        if mode == 'point_estimate':
            mu_start = np.ones(N) / N
            _, traj  = sample_trajectory_film(
                model, mu_start, node_ctx, global_ctx, edge_index,
                n_steps=200, device=device)
            mu_learned        = traj[-1]
            posterior_samples = None
        else:
            samples = []
            for _ in range(K):
                mu_start = rng.dirichlet(np.ones(N))
                _, traj  = sample_trajectory_film(
                    model, mu_start, node_ctx, global_ctx, edge_index,
                    n_steps=200, device=device)
                samples.append(traj[-1])
            posterior_samples = samples
            mu_learned        = np.mean(samples, axis=0)

        mu_mne     = baseline_mne_eeg(y, A, lam=best_lam)
        mu_sloreta = baseline_sloreta(y, A, lam=best_lam)
        mu_lasso   = baseline_lasso_eeg(y, A, alpha=best_alpha)
        mu_bp      = baseline_backproj_eeg(y, A)

        r = {
            'mu_source':       mu_source,
            'mu_learned':      mu_learned,
            'mu_mne':          mu_mne,
            'mu_sloreta':      mu_sloreta,
            'mu_lasso':        mu_lasso,
            'mu_bp':           mu_bp,
            'y':               y,
            'active_parcels':  active_parcels,
            'n_active':        n_active,
            'tv_learned':  total_variation(mu_learned,  mu_source),
            'tv_mne':      total_variation(mu_mne,      mu_source),
            'tv_sloreta':  total_variation(mu_sloreta,  mu_source),
            'tv_lasso':    total_variation(mu_lasso,    mu_source),
            'tv_bp':       total_variation(mu_bp,       mu_source),
            'pk_learned':  peak_recovery_topk(mu_learned,  active_parcels),
            'pk_mne':      peak_recovery_topk(mu_mne,      active_parcels),
            'pk_sloreta':  peak_recovery_topk(mu_sloreta,  active_parcels),
            'pk_lasso':    peak_recovery_topk(mu_lasso,    active_parcels),
            'pk_bp':       peak_recovery_topk(mu_bp,       active_parcels),
            'hemi_learned': hemi_accuracy(mu_learned,  active_parcels, parcel_hemis),
            'hemi_mne':     hemi_accuracy(mu_mne,      active_parcels, parcel_hemis),
            'hemi_sloreta': hemi_accuracy(mu_sloreta,  active_parcels, parcel_hemis),
            'hemi_lasso':   hemi_accuracy(mu_lasso,    active_parcels, parcel_hemis),
            'hemi_bp':      hemi_accuracy(mu_bp,       active_parcels, parcel_hemis),
            'net_learned':  net_accuracy(mu_learned,  active_parcels, network_assignments),
            'net_mne':      net_accuracy(mu_mne,      active_parcels, network_assignments),
            'net_sloreta':  net_accuracy(mu_sloreta,  active_parcels, network_assignments),
            'net_lasso':    net_accuracy(mu_lasso,    active_parcels, network_assignments),
            'net_bp':       net_accuracy(mu_bp,       active_parcels, network_assignments),
            # Spatial extent: entropy ratio (1=uniform, 0=peaked)
            'ent_source':   entropy_ratio(mu_source),
            'ent_learned':  entropy_ratio(mu_learned),
            'ent_sloreta':  entropy_ratio(mu_sloreta),
            'ent_mne':      entropy_ratio(mu_mne),
            'ent_lasso':    entropy_ratio(mu_lasso),
        }

        if posterior_samples is not None:
            post_arr  = np.array(posterior_samples)
            post_mean = post_arr.mean(axis=0)
            post_std  = post_arr.std(axis=0)
            err       = np.abs(post_mean - mu_source)
            r['posterior_samples'] = posterior_samples
            r['posterior_mean']    = post_mean
            r['posterior_std']     = post_std
            r['post_tv']    = total_variation(post_mean, mu_source)
            if post_std.std() > 1e-8:
                r['calib_r'] = float(np.corrcoef(post_std, err)[0, 1])
            else:
                r['calib_r'] = 0.0
            divs = [total_variation(posterior_samples[i], posterior_samples[j])
                    for i in range(K) for j in range(i + 1, K)]
            r['diversity'] = float(np.mean(divs))

        results.append(r)

    return results


# ── Printing ──────────────────────────────────────────────────────────────────

def print_results(results, label, mode='posterior'):
    def mr(key):
        vals = [r[key] for r in results]
        return np.mean(vals), np.std(vals)

    methods = [
        ('tv_learned', 'pk_learned', 'hemi_learned', 'net_learned', 'ent_learned', 'Learned'),
        ('tv_sloreta', 'pk_sloreta', 'hemi_sloreta', 'net_sloreta', 'ent_sloreta', 'sLORETA'),
        ('tv_mne',     'pk_mne',     'hemi_mne',     'net_mne',     'ent_mne',     'MNE'),
        ('tv_lasso',   'pk_lasso',   'hemi_lasso',   'net_lasso',   'ent_lasso',   'LASSO'),
        ('tv_bp',      'pk_bp',      'hemi_bp',      'net_bp',      None,          'Backproj'),
    ]

    print(f"\n{label} ({len(results)} test cases):")
    print(f"  {'Method':12s} {'TV':>12s}  {'Peak%':>6s}  {'Hemi%':>6s}  {'Net%':>6s}  {'EntRatio':>9s}")
    print(f"  {'-'*60}")
    for tv_k, pk_k, hemi_k, net_k, ent_k, name in methods:
        tv_m, tv_s = mr(tv_k)
        pk_m   = np.mean([r[pk_k]   for r in results]) * 100
        hemi_m = np.mean([r[hemi_k] for r in results]) * 100
        net_m  = np.mean([r[net_k]  for r in results]) * 100
        ent_str = ''
        if ent_k and ent_k in results[0]:
            ent_m = np.mean([r[ent_k] for r in results])
            ent_ref = np.mean([r['ent_source'] for r in results])
            ent_str = f'  {ent_m:.3f} (ref {ent_ref:.3f})'
        print(f"  {name:12s}: {tv_m:.4f}±{tv_s:.4f}  {pk_m:5.0f}%  {hemi_m:5.0f}%  {net_m:5.0f}%{ent_str}")

    if mode in ('posterior', 'both') and results and 'calib_r' in results[0]:
        calib_r = np.mean([r['calib_r']  for r in results])
        div_m   = np.mean([r['diversity'] for r in results])
        post_tv = np.mean([r['post_tv']  for r in results])
        print(f"\n  Posterior:")
        print(f"    Mean TV (posterior mean): {post_tv:.4f}")
        print(f"    Calibration r:            {calib_r:.3f}")
        print(f"    Diversity (mean pairwise TV): {div_m:.4f}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _parcel_scatter(ax, parcel_centroids, vals, title='', cmap='hot',
                    vmin=0, vmax=None):
    x  = parcel_centroids[:, 0] * 1e3
    y  = parcel_centroids[:, 1] * 1e3
    vm = vmax if vmax is not None else vals.max()
    ax.scatter(x, y, c=vals, cmap=cmap, vmin=vmin, vmax=vm,
               s=55, edgecolors='k', linewidths=0.3)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def plot_main_figure(results, parcel_centroids, network_names,
                     losses=None, mode='posterior', out_path=None):
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f'Experiment 14c: Realistic EEG Source Reconstruction (mode={mode})\n'
        'MNE-simulated sources · 100-parcel Schaefer · 64-ch EEG',
        fontsize=11)
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.45)

    # Panel A: training loss
    ax_A = fig.add_subplot(gs[0, 0])
    if losses is not None:
        ax_A.plot(losses, color='steelblue', lw=1.0, alpha=0.6, label='loss')
        if len(losses) > 20:
            k      = max(1, len(losses) // 50)
            smooth = np.convolve(losses, np.ones(k) / k, mode='valid')
            ax_A.plot(np.arange(k - 1, len(losses)), smooth,
                      color='red', lw=1.5, label='smoothed')
        ax_A.set_yscale('log')
        ax_A.legend(fontsize=7)
    else:
        ax_A.text(0.5, 0.5, 'Loaded from\ncheckpoint',
                  transform=ax_A.transAxes, ha='center', va='center', fontsize=12)
    ax_A.set_xlabel('Epoch'); ax_A.set_ylabel('Loss')
    ax_A.set_title('A: Training Loss', fontsize=9)
    ax_A.grid(True, alpha=0.3)

    # Panel B: brain scatter — pick one 1-active example
    example = next((r for r in results if r['n_active'] == 1), results[0])
    ax_B    = fig.add_subplot(gs[0, 1])
    ax_B.axis('off')
    ax_B.set_title('B: Brain Surface Reconstruction (axial)', fontsize=9, pad=4)
    inner_B = gs[0, 1].subgridspec(3, 1, hspace=0.6)
    vm_B    = max(example['mu_source'].max(), 0.05)
    for row_i, (key, label_str, col_c) in enumerate([
        ('mu_source',  'True source', 'black'),
        ('mu_learned', 'Learned',     '#2166ac'),
        ('mu_sloreta', 'sLORETA',     '#4363d8'),
    ]):
        axi = fig.add_subplot(inner_B[row_i])
        _parcel_scatter(axi, parcel_centroids, example[key], vmin=0, vmax=vm_B)
        peaks = example['active_parcels']
        axi.scatter(parcel_centroids[peaks, 0] * 1e3,
                    parcel_centroids[peaks, 1] * 1e3,
                    s=120, marker='*', c='cyan', zorder=5,
                    edgecolors='k', linewidths=0.5)
        tv_str = ''
        if row_i > 0:
            tv_key = key.replace('mu_', 'tv_')
            tv_str = f' (TV={example[tv_key]:.3f})'
        axi.set_title(label_str + tv_str, fontsize=7, color=col_c)

    # Panel C: TV comparison bar
    ax_C = fig.add_subplot(gs[0, 2])
    methods_tv = [
        ('tv_learned', 'Learned',  '#2166ac'),
        ('tv_sloreta', 'sLORETA',  '#4363d8'),
        ('tv_mne',     'MNE',      '#3cb44b'),
        ('tv_lasso',   'LASSO',    '#f58231'),
        ('tv_bp',      'Backproj', '#e6194b'),
    ]
    means_tv = [np.mean([r[k] for r in results]) for k, _, _ in methods_tv]
    stds_tv  = [np.std( [r[k] for r in results]) for k, _, _ in methods_tv]
    labels_c = [l for _, l, _ in methods_tv]
    colors_c = [c for _, _, c in methods_tv]
    ax_C.bar(np.arange(len(methods_tv)), means_tv, yerr=stds_tv, capsize=4,
             color=colors_c, edgecolor='k', linewidth=0.4, alpha=0.85)
    ax_C.set_xticks(np.arange(len(methods_tv)))
    ax_C.set_xticklabels(labels_c, rotation=20, ha='right', fontsize=8)
    ax_C.set_ylabel('Total Variation')
    ax_C.set_title(f'C: TV Comparison ({len(results)} test cases)', fontsize=9)
    ax_C.grid(True, alpha=0.3, axis='y')

    # Panel D: network accuracy
    ax_D = fig.add_subplot(gs[1, 0])
    methods_net = [
        ('net_learned', 'Learned',  '#2166ac'),
        ('net_sloreta', 'sLORETA',  '#4363d8'),
        ('net_mne',     'MNE',      '#3cb44b'),
        ('net_lasso',   'LASSO',    '#f58231'),
        ('net_bp',      'Backproj', '#e6194b'),
    ]
    means_net = [np.mean([r[k] for r in results]) * 100 for k, _, _ in methods_net]
    labels_d  = [l for _, l, _ in methods_net]
    colors_d  = [c for _, _, c in methods_net]
    ax_D.bar(np.arange(len(methods_net)), means_net, color=colors_d,
             edgecolor='k', linewidth=0.4, alpha=0.85)
    ax_D.set_xticks(np.arange(len(methods_net)))
    ax_D.set_xticklabels(labels_d, rotation=20, ha='right', fontsize=8)
    ax_D.set_ylabel('Network accuracy (%)')
    ax_D.set_ylim(0, 115)
    ax_D.set_title('D: Network Accuracy (correct of 7)', fontsize=9)
    ax_D.grid(True, alpha=0.3, axis='y')

    # Panel E: spatial extent (entropy ratio comparison) or calibration
    ax_E = fig.add_subplot(gs[1, 1])
    if mode in ('posterior', 'both') and results and 'calib_r' in results[0]:
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
        # Entropy ratio comparison
        ent_keys  = [('ent_source', 'True source', 'black'),
                     ('ent_learned', 'Learned', '#2166ac'),
                     ('ent_sloreta', 'sLORETA', '#4363d8'),
                     ('ent_mne',     'MNE',     '#3cb44b'),
                     ('ent_lasso',   'LASSO',   '#f58231')]
        means_ent = [np.mean([r[k] for r in results]) for k, _, _ in ent_keys]
        colors_e  = [c for _, _, c in ent_keys]
        labels_e  = [l for _, l, _ in ent_keys]
        ax_E.bar(np.arange(len(ent_keys)), means_ent, color=colors_e,
                 edgecolor='k', linewidth=0.4, alpha=0.85)
        ax_E.set_xticks(np.arange(len(ent_keys)))
        ax_E.set_xticklabels(labels_e, rotation=20, ha='right', fontsize=8)
        ax_E.set_ylabel('Entropy ratio (1=uniform)')
        ax_E.set_title('E: Spatial Extent (entropy ratio)', fontsize=9)
        ax_E.grid(True, alpha=0.3, axis='y')

    # Panel F: TV by number of active regions
    ax_F = fig.add_subplot(gs[1, 2])
    n_active_vals = sorted(set(r['n_active'] for r in results))
    for key, lbl, color in methods_tv:
        tv_by_na = [np.mean([r[key] for r in results if r['n_active'] == na])
                    for na in n_active_vals]
        ax_F.plot(n_active_vals, tv_by_na, 'o-', label=lbl, color=color, lw=1.5)
    ax_F.set_xlabel('Number of active regions')
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


def plot_posterior_figure(results, parcel_centroids, out_path=None):
    """Posterior analysis: mean and std for 1/2/3-active cases."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Experiment 14c: Posterior Analysis (realistic sources)', fontsize=11)

    cases = [next((r for r in results if r['n_active'] == k), results[k - 1])
             for k in [1, 2, 3]]

    for col, r in enumerate(cases):
        ax  = axes[0, col]
        vm  = max(r['mu_source'].max(), 0.05)
        if 'posterior_mean' in r:
            _parcel_scatter(ax, parcel_centroids, r['posterior_mean'],
                            vmin=0, vmax=vm, cmap='hot')
        else:
            _parcel_scatter(ax, parcel_centroids, r['mu_learned'],
                            vmin=0, vmax=vm, cmap='hot')
        peaks = r['active_parcels']
        ax.scatter(parcel_centroids[peaks, 0] * 1e3,
                   parcel_centroids[peaks, 1] * 1e3,
                   s=120, marker='*', c='cyan', zorder=5,
                   edgecolors='k', linewidths=0.5)
        tv_val = r.get('post_tv', r.get('tv_learned', 0))
        ax.set_title(f'{r["n_active"]}-active\nTV={tv_val:.3f}, '
                     f'calib r={r.get("calib_r", 0):.2f}', fontsize=8)
        if col == 0:
            ax.set_ylabel('Posterior mean', fontsize=8)

    for col, r in enumerate(cases):
        ax = axes[1, col]
        if 'posterior_std' in r:
            _parcel_scatter(ax, parcel_centroids, r['posterior_std'],
                            vmin=0, cmap='viridis')
            ax.set_title(f'Diversity={r.get("diversity", 0):.3f}', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'N/A\n(point estimate)',
                    transform=ax.transAxes, ha='center', va='center', fontsize=10)
        if col == 0:
            ax.set_ylabel('Posterior std', fontsize=8)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out_path}")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ex14c: Realistic EEG source reconstruction with MNE simulation')
    parser.add_argument('--mode', type=str, default='posterior',
                        choices=['point_estimate', 'posterior', 'both'])
    parser.add_argument('--data-path',        type=str,   default='ex14_eeg_data.npz')
    parser.add_argument('--n-simulations',    type=int,   default=1000)
    parser.add_argument('--n-test',           type=int,   default=200)
    parser.add_argument('--spatial-extent',   type=float, default=10.0,
                        help='Source spatial extent in mm')
    parser.add_argument('--snr-db',           type=float, default=20.0)
    parser.add_argument('--n-starts-per-pair',type=int,   default=10)
    parser.add_argument('--n-samples',        type=int,   default=20000)
    parser.add_argument('--n-epochs',         type=int,   default=1000)
    parser.add_argument('--hidden-dim',       type=int,   default=128)
    parser.add_argument('--n-layers',         type=int,   default=6)
    parser.add_argument('--lr',               type=float, default=5e-4)
    parser.add_argument('--ema-decay',        type=float, default=0.999)
    parser.add_argument('--loss-type',        type=str,   default='rate_kl',
                        choices=['rate_kl', 'mse'])
    parser.add_argument('--posterior-k',      type=int,   default=20)
    parser.add_argument('--regenerate',       action='store_true',
                        help='Force regeneration of simulation cache')
    parser.add_argument('--recompute-fwd',    action='store_true',
                        help='Recompute MNE forward solution')
    parser.add_argument('--seed',             type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,  exist_ok=True)
    torch.manual_seed(args.seed)
    device = get_device()

    # ── 1. Load precomputed graph + parcellated leadfield from Ex14a ──────────
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(ROOT, data_path)
    print(f"=== Experiment 14c: Realistic EEG Source Reconstruction ===\n")
    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    R                   = data['R']
    A                   = data['A']
    parcel_centroids    = data['parcel_centroids']
    parcel_names        = data['parcel_names']
    network_assignments = data['network_assignments'].astype(int)
    network_names       = data['network_names']
    N          = R.shape[0]
    n_channels = A.shape[0]

    parcel_hemis = np.array(
        [0 if '_LH_' in str(n) else 1 for n in parcel_names], dtype=int)
    electrode_parcels = compute_electrode_parcels(A)
    edge_index        = rate_matrix_to_edge_index(R)
    global_dim        = n_channels   # raw EEG only — no tau_diff

    n_covered = len(set(electrode_parcels.tolist()))
    print(f"Graph: {N} parcels, {edge_index.shape[1]//2} edges, {n_channels} EEG channels")
    print(f"Electrode coverage: {n_covered}/{N} parcels")
    print(f"Mode: {args.mode}")

    # ── 2. Load or generate simulation cache ──────────────────────────────────
    sim_cache = os.path.join(
        DATA_DIR,
        f'ex14c_sims_n{args.n_simulations}_ext{int(args.spatial_extent)}'
        f'_snr{int(args.snr_db)}_seed{args.seed}.npz')

    if os.path.exists(sim_cache) and not args.regenerate:
        print(f"\nLoading cached simulations from {sim_cache}")
        train_pairs, test_pairs = load_simulation_cache(sim_cache)
        print(f"  Loaded {len(train_pairs)} train, {len(test_pairs)} test pairs")
    else:
        print("\nSetting up MNE (fsaverage + forward solution)...")
        import mne
        fs_dir       = mne.datasets.fetch_fsaverage(verbose=False)
        subjects_dir = os.path.dirname(fs_dir)
        src, labels, fwd, info = _setup_mne(
            subjects_dir, n_sensors=n_channels,
            recompute_fwd=args.recompute_fwd)
        print(f"  Source space: {src[0]['nuse']} LH + {src[1]['nuse']} RH vertices")

        print(f"\nGenerating {args.n_simulations} training simulations "
              f"(spatial_extent={args.spatial_extent}mm, SNR={args.snr_db}dB)...")
        train_pairs = generate_training_data(
            labels, src, fwd,
            n_simulations=args.n_simulations,
            spatial_extent=args.spatial_extent,
            snr_db=args.snr_db,
            seed=args.seed)

        print(f"\nGenerating {args.n_test} test simulations...")
        test_pairs = generate_training_data(
            labels, src, fwd,
            n_simulations=args.n_test,
            spatial_extent=args.spatial_extent,
            snr_db=args.snr_db,
            seed=args.seed + 1000)

        save_simulation_cache(sim_cache, train_pairs, test_pairs, {
            'n_simulations':  np.array(args.n_simulations),
            'spatial_extent': np.array(args.spatial_extent),
            'snr_db':         np.array(args.snr_db),
        })
        print(f"  Saved simulation cache to {sim_cache}")

    # ── 3. Baseline tuning on first 20 training pairs ────────────────────────
    print("\nTuning baselines...")
    best_lam, best_alpha = tune_baselines_eeg(train_pairs[:20], A)
    print(f"  Best MNE lambda: {best_lam}, LASSO alpha: {best_alpha}")

    # ── 4. Helper: build / load model ────────────────────────────────────────
    def _ckpt_path(suffix):
        return os.path.join(
            CKPT_DIR,
            f'meta_model_ex14c_{suffix}_{args.n_epochs}ep'
            f'_h{args.hidden_dim}_l{args.n_layers}.pt')

    def _make_model():
        return FiLMConditionalGNNRateMatrixPredictor(
            node_context_dim=2,
            global_dim=global_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers)

    def _train_or_load(tag, dataset, ckpt):
        model      = _make_model()
        losses_out = None
        if os.path.exists(ckpt):
            model.load_state_dict(
                torch.load(ckpt, map_location='cpu', weights_only=True))
            print(f"  Loaded checkpoint from {ckpt}")
        else:
            print(f"  Training ({args.n_epochs} epochs, {len(dataset)} samples)...")
            history    = train_film_conditional(
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

    # ── 5a. Point estimate mode ───────────────────────────────────────────────
    if args.mode in ('point_estimate', 'both'):
        print(f"\n--- Mode: Point Estimate ---")
        print(f"Building RealisticEEGDataset (uniform start, {args.n_samples} samples)...")
        dataset_point = RealisticEEGDataset(
            R, train_pairs, electrode_parcels,
            mode='uniform',
            n_starts_per_pair=1,
            n_samples=args.n_samples,
            seed=args.seed)
        print(f"  Dataset: {len(dataset_point)} samples")

        model_point, losses_point = _train_or_load(
            'point', dataset_point, _ckpt_path('point'))

        print(f"\nEvaluating on {len(test_pairs)} test cases...")
        results_point = run_evaluation(
            model_point, test_pairs, R, A, edge_index, electrode_parcels,
            parcel_hemis, network_assignments,
            best_lam, best_alpha, device,
            mode='point_estimate', seed=99)

        print_results(results_point, 'Point estimate', mode='point_estimate')

        plot_main_figure(
            results_point, parcel_centroids, network_names,
            losses=losses_point, mode='point_estimate',
            out_path=os.path.join(HERE, 'ex14c_point_estimate.png'))

    # ── 5b. Posterior mode ────────────────────────────────────────────────────
    if args.mode in ('posterior', 'both'):
        print(f"\n--- Mode: Posterior Sampling ---")
        print(f"Building RealisticEEGDataset "
              f"({len(train_pairs)} pairs × {args.n_starts_per_pair} starts, "
              f"{args.n_samples} samples)...")
        dataset_post = RealisticEEGDataset(
            R, train_pairs, electrode_parcels,
            mode='dirichlet',
            n_starts_per_pair=args.n_starts_per_pair,
            n_samples=args.n_samples,
            seed=args.seed)
        print(f"  Dataset: {len(dataset_post)} samples")

        model_post, losses_post = _train_or_load(
            'posterior', dataset_post, _ckpt_path('posterior'))

        print(f"\nEvaluating on {len(test_pairs)} test cases (K={args.posterior_k})...")
        results_post = run_evaluation(
            model_post, test_pairs, R, A, edge_index, electrode_parcels,
            parcel_hemis, network_assignments,
            best_lam, best_alpha, device,
            mode='posterior', K=args.posterior_k, seed=99)

        print_results(results_post, 'Posterior', mode='posterior')

        plot_main_figure(
            results_post, parcel_centroids, network_names,
            losses=losses_post, mode='posterior',
            out_path=os.path.join(HERE, 'ex14c_posterior.png'))

        plot_posterior_figure(
            results_post, parcel_centroids,
            out_path=os.path.join(HERE, 'ex14c_posterior_analysis.png'))

    print("\nDone.")


if __name__ == '__main__':
    main()
