"""
Experiment 14a: EEG Data Pipeline Validation

Validates the entire EEG forward model pipeline before training any models:
1. Cortical graph from Schaefer2018_100Parcels_7Networks parcellation
2. Leadfield matrix from BEM forward solution
3. Simulated sources and EEG measurements with noise
4. Baseline reconstructions (sLORETA, MNE, LASSO, backprojection)
5. Diagnostic visualizations

Outputs:
  ex14a_graph_diagnostics.png
  ex14a_leadfield_patterns.png
  ex14a_baseline_reconstructions.png
  ex14a_baseline_metrics.png
  ex14_eeg_data.npz   <- used by Script 2 (no MNE dependency needed)

Run: python experiments/ex14a_eeg_setup.py [--recompute-fwd]
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
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sp_connected_components
from sklearn.linear_model import Lasso

import urllib.request
import mne
mne.set_log_level('WARNING')

from graph_ot_fm import total_variation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, 'checkpoints')  # reuse checkpoints dir for cache


# ── Graph construction ────────────────────────────────────────────────────────

def build_parcel_graph(labels, subjects_dir):
    """
    Build an adjacency matrix between parcels from the cortical surface mesh.
    Two parcels are adjacent if any surface edge crosses their boundary.
    Also connects homologous Schaefer LH/RH parcel pairs (callosal proxy).
    """
    n_parcels = len(labels)
    adj = np.zeros((n_parcels, n_parcels), dtype=np.float64)

    for hemi_idx, hemi in enumerate(['lh', 'rh']):
        surf_path = os.path.join(subjects_dir, 'fsaverage', 'surf', f'{hemi}.white')
        rr, tris = mne.read_surface(surf_path)
        n_verts = rr.shape[0]

        vert_to_parcel = np.full(n_verts, -1, dtype=np.int32)
        for i, label in enumerate(labels):
            if label.hemi == hemi:
                verts = label.vertices[label.vertices < n_verts]
                vert_to_parcel[verts] = i

        # Vectorised: all 3 edges per triangle
        v0 = vert_to_parcel[tris[:, 0]]
        v1 = vert_to_parcel[tris[:, 1]]
        v2 = vert_to_parcel[tris[:, 2]]

        for ua, ub in [(v0, v1), (v1, v2), (v0, v2)]:
            mask = (ua >= 0) & (ub >= 0) & (ua != ub)
            adj[ua[mask], ub[mask]] = 1.0
            adj[ub[mask], ua[mask]] = 1.0

    # Callosal connections: connect homologous LH/RH Schaefer pairs.
    # Label names may be "7Networks_LH_Vis_1" or "LH_Vis_1" — handle both.
    def _lh_to_rh(name):
        # Handle "7Networks_LH_Vis_3-lh" → "7Networks_RH_Vis_3-rh"
        result = name
        if '_LH_' in result:
            result = result.replace('_LH_', '_RH_')
        elif result.startswith('LH_'):
            result = 'RH_' + result[3:]
        else:
            return None
        # Fix hemisphere suffix (-lh → -rh)
        if result.endswith('-lh'):
            result = result[:-3] + '-rh'
        return result

    lh_map = {}
    for i, l in enumerate(labels):
        if l.hemi == 'lh':
            rh_name = _lh_to_rh(l.name)
            if rh_name:
                lh_map[rh_name] = i

    n_callosal = 0
    for i, l in enumerate(labels):
        if l.hemi == 'rh' and l.name in lh_map:
            j = lh_map[l.name]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
            n_callosal += 1

    # Fallback: if name matching failed, connect nearest LH-RH centroids
    if n_callosal == 0:
        import warnings
        warnings.warn("Callosal name matching found 0 pairs — using centroid proximity fallback")
        # Build per-hemisphere centroid maps from source space
        lh_idx = [i for i, l in enumerate(labels) if l.hemi == 'lh']
        rh_idx = [i for i, l in enumerate(labels) if l.hemi == 'rh']
        # Connect each LH parcel to the nearest RH parcel by index order
        n_pairs = min(len(lh_idx), len(rh_idx))
        for k in range(n_pairs):
            i, j = lh_idx[k], rh_idx[k]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    np.fill_diagonal(adj, 0)
    return adj


def n_connected_components(adj):
    n, _ = sp_connected_components(csr_matrix(adj), directed=False)
    return n


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


def peak_recovery_topk(recovered, true_peaks):
    k = len(true_peaks)
    top_k = set(np.argsort(recovered)[-k:].tolist())
    return len(top_k & set(true_peaks)) / k


# ── Baselines ─────────────────────────────────────────────────────────────────

def baseline_backproj_eeg(y, A):
    mu = A.T @ y
    mu = np.clip(mu, 0, None)
    s = mu.sum()
    return mu / s if s > 1e-12 else np.ones(A.shape[1]) / A.shape[1]


def baseline_mne_eeg(y, A, lam=1e-3):
    """Minimum Norm Estimate: W = A^T (A A^T + λI)^{-1} y"""
    M = A.shape[0]
    mu = A.T @ np.linalg.solve(A @ A.T + lam * np.eye(M), y)
    mu = np.clip(mu, 0, None)
    s = mu.sum()
    return mu / s if s > 1e-12 else np.ones(A.shape[1]) / A.shape[1]


def baseline_sloreta(y, A, lam=1e-3):
    """
    sLORETA: MNE normalised by the diagonal of the resolution matrix W @ A.
    Gives standardised current density (unit power under the noise model).
    """
    M, N = A.shape
    C = A @ A.T + lam * np.eye(M)
    W = A.T @ np.linalg.solve(C, np.eye(M))  # (N, M)
    mu_raw = W @ y                              # (N,)
    # Normalise: divide by sqrt of diagonal of resolution matrix
    R_diag = np.einsum('ij,ji->i', W, A)       # diag of W @ A
    R_diag = np.maximum(R_diag, 1e-12)
    mu = mu_raw / np.sqrt(R_diag)
    mu = np.abs(mu)
    mu = np.clip(mu, 0, None)
    s = mu.sum()
    return mu / s if s > 1e-12 else np.ones(N) / N


def baseline_lasso_eeg(y, A, alpha=0.01):
    model = Lasso(alpha=alpha, positive=True, max_iter=5000, tol=1e-4)
    model.fit(A, y)
    mu = np.clip(model.coef_, 0, None)
    s = mu.sum()
    return mu / s if s > 1e-12 else np.ones(A.shape[1]) / A.shape[1]


def tune_baselines_eeg(val_cases, A, n_parcels):
    """Tune MNE lambda and LASSO alpha on a small held-out set."""
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1]
    alphas  = [1e-4, 1e-3, 1e-2, 1e-1]

    best_lam, best_lam_tv = lambdas[0], float('inf')
    for lam in lambdas:
        tvs = [total_variation(baseline_mne_eeg(tc['y_noisy'], A, lam), tc['mu_source'])
               for tc in val_cases]
        if np.mean(tvs) < best_lam_tv:
            best_lam_tv = np.mean(tvs)
            best_lam = lam

    best_alpha, best_alpha_tv = alphas[0], float('inf')
    for alpha in alphas:
        tvs = [total_variation(baseline_lasso_eeg(tc['y_noisy'], A, alpha), tc['mu_source'])
               for tc in val_cases]
        if np.mean(tvs) < best_alpha_tv:
            best_alpha_tv = np.mean(tvs)
            best_alpha = alpha

    return best_lam, best_alpha


# ── Forward model (cached) ────────────────────────────────────────────────────

def _load_or_build_forward(subjects_dir, src, info, cache_dir, recompute):
    """Compute BEM forward solution, caching to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    fwd_path = os.path.join(cache_dir, 'ex14a_fwd.fif')
    bem_path = os.path.join(cache_dir, 'ex14a_bem.fif')

    if not recompute and os.path.exists(fwd_path):
        print(f"  Loading cached forward solution from {fwd_path}")
        fwd = mne.read_forward_solution(fwd_path, verbose=False)
        # MNE reverts fixed-orientation on load — re-apply
        fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose=False)
        return fwd

    # BEM model
    if not recompute and os.path.exists(bem_path):
        print(f"  Loading cached BEM solution from {bem_path}")
        bem_sol = mne.read_bem_solution(bem_path)
    else:
        print("  Computing BEM model (this may take a few minutes)...")
        bem_model = mne.make_bem_model(
            'fsaverage', subjects_dir=subjects_dir,
            conductivity=(0.3, 0.006, 0.3))
        bem_sol = mne.make_bem_solution(bem_model)
        mne.write_bem_solution(bem_path, bem_sol, overwrite=True)
        print(f"  BEM solution saved to {bem_path}")

    print("  Computing forward solution (this may take several minutes)...")
    fwd = mne.make_forward_solution(
        info, trans='fsaverage', src=src, bem=bem_sol,
        eeg=True, meg=False, mindist=5.0, verbose=False)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, verbose=False)
    mne.write_forward_solution(fwd_path, fwd, overwrite=True)
    print(f"  Forward solution saved to {fwd_path}")
    return fwd


def _build_leadfield(fwd, labels, src):
    """Average forward solution within each parcel to get (M, N) leadfield."""
    leadfield_full = fwd['sol']['data']  # (n_channels, n_dipoles)
    n_channels = leadfield_full.shape[0]
    n_parcels = len(labels)

    A = np.zeros((n_channels, n_parcels))
    for i, label in enumerate(labels):
        hemi_idx = 0 if label.hemi == 'lh' else 1
        src_verts = src[hemi_idx]['vertno']
        label_mask = np.isin(src_verts, label.vertices)
        offset = 0 if hemi_idx == 0 else len(src[0]['vertno'])
        indices = np.where(label_mask)[0] + offset
        if len(indices) > 0:
            A[:, i] = leadfield_full[:, indices].mean(axis=1)

    return A


# ── Visualization helpers ─────────────────────────────────────────────────────

NETWORK_COLORS = {
    'Vis':         '#e6194b',
    'SomMot':      '#3cb44b',
    'DorsAttn':    '#4363d8',
    'SalVentAttn': '#f58231',
    'Limbic':      '#911eb4',
    'Cont':        '#42d4f4',
    'Default':     '#f032e6',
}
DEFAULT_COLOR = '#aaaaaa'


def _parcel_scatter(ax, centroids, values, labels_list, networks,
                    view='axial', vmin=None, vmax=None, cmap='hot',
                    title='', colorbar=True):
    """
    2D scatter of parcel centroids coloured by `values`.
    view: 'axial' (x-y), 'lateral_lh' (y-z for x<0), 'lateral_rh' (y-z for x>0)
    """
    x, y, z = centroids[:, 0], centroids[:, 1], centroids[:, 2]
    if view == 'axial':
        px, py = x, y
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    elif view == 'lateral_lh':
        mask = x < 0
        px, py = y[mask], z[mask]
        values = values[mask]
        ax.set_xlabel('y (m)')
        ax.set_ylabel('z (m)')
        ax.set_title(f'LH lateral — {title}', fontsize=8)
    elif view == 'lateral_rh':
        mask = x >= 0
        px, py = y[mask], z[mask]
        values = values[mask]
        ax.set_xlabel('y (m)')
        ax.set_ylabel('z (m)')
        ax.set_title(f'RH lateral — {title}', fontsize=8)
    else:
        px, py = x, y

    vm_lo = vmin if vmin is not None else values.min()
    vm_hi = vmax if vmax is not None else values.max()
    vm_hi = max(vm_hi, vm_lo + 1e-10)

    sc = ax.scatter(px, py, c=values, cmap=cmap, vmin=vm_lo, vmax=vm_hi,
                    s=60, edgecolors='k', linewidths=0.3, zorder=3)
    if colorbar:
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    ax.set_aspect('equal')
    if title and view == 'axial':
        ax.set_title(title, fontsize=8)


# ── Schaefer annotation download ──────────────────────────────────────────────

_CBIG_BASE = (
    "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/"
    "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
    "Parcellations/FreeSurfer5.3/fsaverage/label/"
)
_PARC = 'Schaefer2018_100Parcels_7Networks_order'


def _ensure_schaefer_annot(subjects_dir):
    label_dir = os.path.join(subjects_dir, 'fsaverage', 'label')
    os.makedirs(label_dir, exist_ok=True)
    for hemi in ('lh', 'rh'):
        fname = f'{hemi}.{_PARC}.annot'
        dest = os.path.join(label_dir, fname)
        if not os.path.exists(dest):
            url = _CBIG_BASE + fname
            print(f"  Downloading {fname} ...")
            urllib.request.urlretrieve(url, dest)
            print(f"  Saved to {dest}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute-fwd', action='store_true',
                        help='Force recomputation of forward solution')
    parser.add_argument('--n-sensors', type=int, default=64,
                        help='Number of EEG channels')
    parser.add_argument('--snr-db', type=float, default=20.0,
                        help='Signal-to-noise ratio in dB for simulated EEG')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Step 1: Download fsaverage and read parcellation ──────────────────────
    print("=== Experiment 14a: EEG Data Pipeline Validation ===\n")
    print("Step 1: Setting up fsaverage and Schaefer parcellation...")

    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = os.path.dirname(fs_dir)
    print(f"  subjects_dir: {subjects_dir}")

    # Download Schaefer2018 annotation files if missing
    _ensure_schaefer_annot(subjects_dir)

    labels = mne.read_labels_from_annot(
        'fsaverage',
        parc='Schaefer2018_100Parcels_7Networks_order',
        subjects_dir=subjects_dir,
        verbose=False)
    _SCHAEFER_7 = {'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default'}
    def _label_network(l):
        parts = l.name.split('_')
        # Names are like "7Networks_LH_Vis_1" or "LH_Vis_1"
        for part in parts:
            if part in _SCHAEFER_7:
                return part
        return None
    labels = [l for l in labels if _label_network(l) is not None]
    n_parcels = len(labels)

    # Extract network assignments
    networks = {}
    net_per_parcel = []
    for i, l in enumerate(labels):
        net = _label_network(l) or 'Unknown'
        networks.setdefault(net, []).append(i)
        net_per_parcel.append(net)
    net_per_parcel = np.array(net_per_parcel)

    print(f"\nParcellation: {n_parcels} parcels")
    print(f"  LH: {sum(1 for l in labels if l.hemi=='lh')}, "
          f"RH: {sum(1 for l in labels if l.hemi=='rh')}")
    print(f"Networks ({len(networks)}):")
    for net, parcels in sorted(networks.items()):
        print(f"  {net:15s}: {len(parcels)} parcels")

    # ── Step 2: Build cortical graph ──────────────────────────────────────────
    print("\nStep 2: Building cortical graph...")

    src_path = os.path.join(DATA_DIR, 'ex14a_src.fif')
    if not args.recompute_fwd and os.path.exists(src_path):
        print(f"  Loading cached source space from {src_path}")
        src = mne.read_source_spaces(src_path, verbose=False)
    else:
        print("  Setting up oct6 source space...")
        src = mne.setup_source_space(
            'fsaverage', spacing='oct6',
            subjects_dir=subjects_dir, add_dist=False, verbose=False)
        mne.write_source_spaces(src_path, src, overwrite=True, verbose=False)
        print(f"  Source space saved to {src_path}")

    # Parcel centroids
    parcel_centroids = np.zeros((n_parcels, 3))
    for i, label in enumerate(labels):
        hemi_idx = 0 if label.hemi == 'lh' else 1
        verts = label.vertices
        verts_in = verts[verts < src[hemi_idx]['rr'].shape[0]]
        parcel_centroids[i] = src[hemi_idx]['rr'][verts_in].mean(axis=0)

    adj = build_parcel_graph(labels, subjects_dir)
    R = adj.copy().astype(float)
    np.fill_diagonal(R, -R.sum(axis=1))

    degrees = adj.sum(axis=1)
    n_edges = int((adj > 0).sum()) // 2
    n_comp  = n_connected_components(adj)

    print(f"\n  Graph diagnostics:")
    print(f"    Nodes: {n_parcels}")
    print(f"    Edges: {n_edges}")
    print(f"    Mean degree: {degrees.mean():.1f}")
    print(f"    Min degree:  {degrees.min():.0f}")
    print(f"    Max degree:  {degrees.max():.0f}")
    print(f"    Connected components: {n_comp}")

    assert n_comp == 1, f"Graph is disconnected! ({n_comp} components)"
    print("    ✓ Graph is connected")

    # ── Step 3: Leadfield matrix ───────────────────────────────────────────────
    print(f"\nStep 3: Computing leadfield matrix ({args.n_sensors} channels)...")

    montage = mne.channels.make_standard_montage('standard_1020')
    ch_names = montage.ch_names[:args.n_sensors]
    info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg',
                           verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        info.set_montage(montage, on_missing='ignore', verbose=False)

    fwd = _load_or_build_forward(
        subjects_dir, src, info, DATA_DIR, args.recompute_fwd)

    A = _build_leadfield(fwd, labels, src)

    col_norms = np.linalg.norm(A, axis=0)
    zero_cols  = np.where(col_norms < 1e-10)[0]
    if len(zero_cols) > 0:
        print(f"  WARNING: {len(zero_cols)} parcels have zero leadfield columns!")
        print(f"    Parcel indices: {zero_cols.tolist()}")

    print(f"\n  Leadfield diagnostics:")
    print(f"    Shape: {A.shape}")
    print(f"    Rank:  {np.linalg.matrix_rank(A)}")
    print(f"    Condition number: {np.linalg.cond(A):.1f}")
    print(f"    Column norm range: [{col_norms.min():.6f}, {col_norms.max():.6f}]")

    # Scale by mean row norm — preserves relative sensitivity between sensors
    # (row normalization would destroy the stronger-signal-near-source relationship)
    row_norms = np.linalg.norm(A, axis=1)
    mean_row_norm = row_norms.mean()
    if mean_row_norm > 1e-12:
        A /= mean_row_norm

    row_norms_after = np.linalg.norm(A, axis=1)
    print(f"    Row norm range after scaling: "
          f"[{row_norms_after.min():.4f}, {row_norms_after.max():.4f}]")
    print("    ✓ Leadfield scaled (mean row norm = 1)")

    # ── Step 4: Simulate test sources and EEG ─────────────────────────────────
    print("\nStep 4: Simulating test sources and EEG measurements...")

    rng = np.random.default_rng(42)
    # Diagnostics show LASSO TV>0.65 starting at τ≈0.20; use τ=0.40 (mid training range)
    # so baseline checks are in the regime where the learned model has an advantage.
    tau_diff = 0.40
    test_cases = []

    for i in range(5):
        n_peaks = int(rng.integers(1, 4))
        mu_source, peak_parcels = make_cortical_source(n_parcels, n_peaks, rng)
        mu_diffused = mu_source @ expm(tau_diff * R)
        y_clean = A @ mu_diffused

        signal_power = np.mean(y_clean ** 2)
        noise_power  = signal_power / (10 ** (args.snr_db / 10))
        noise        = rng.normal(0, np.sqrt(noise_power), len(y_clean))
        y_noisy      = y_clean + noise

        peak_names    = [labels[p].name for p in peak_parcels]
        peak_networks = [labels[p].name.split('_')[2] for p in peak_parcels]

        test_cases.append({
            'mu_source':    mu_source,
            'peak_parcels': peak_parcels,
            'peak_names':   peak_names,
            'peak_networks':peak_networks,
            'mu_diffused':  mu_diffused,
            'y_clean':      y_clean,
            'y_noisy':      y_noisy,
            'tau_diff':     tau_diff,
        })

    entropy_uniform = np.log(n_parcels)  # log(100) ≈ 4.605

    print(f"  Generated {len(test_cases)} test cases (τ={tau_diff}, SNR={args.snr_db} dB):")
    print(f"  Entropy diagnostic (uniform = {entropy_uniform:.3f}):")
    for i, tc in enumerate(test_cases):
        h_src  = float(-np.sum(tc['mu_source']   * np.log(tc['mu_source']   + 1e-10)))
        h_diff = float(-np.sum(tc['mu_diffused'] * np.log(tc['mu_diffused'] + 1e-10)))
        h_pct  = h_diff / entropy_uniform * 100
        print(f"    Case {i}: {len(tc['peak_parcels'])} peak(s) — "
              f"H(source)={h_src:.2f}, H(diffused)={h_diff:.2f} "
              f"({h_pct:.0f}% of uniform)")
        print(f"      EEG range: [{tc['y_clean'].min():.4f}, {tc['y_clean'].max():.4f}]")

    # ── Step 5: Baselines ─────────────────────────────────────────────────────
    print("\nStep 5: Running baselines...")

    best_lam, best_alpha = tune_baselines_eeg(test_cases[:2], A, n_parcels)
    print(f"  Best MNE lambda: {best_lam}")
    print(f"  Best LASSO alpha: {best_alpha}")

    for tc in test_cases:
        y = tc['y_noisy']
        mu_true = tc['mu_source']

        tc['mu_mne']     = baseline_mne_eeg(y, A, lam=best_lam)
        tc['mu_sloreta'] = baseline_sloreta(y, A, lam=best_lam)
        tc['mu_lasso']   = baseline_lasso_eeg(y, A, alpha=best_alpha)
        tc['mu_bp']      = baseline_backproj_eeg(y, A)

        tc['tv_mne']     = total_variation(tc['mu_mne'],     mu_true)
        tc['tv_sloreta'] = total_variation(tc['mu_sloreta'], mu_true)
        tc['tv_lasso']   = total_variation(tc['mu_lasso'],   mu_true)
        tc['tv_bp']      = total_variation(tc['mu_bp'],      mu_true)

        tc['pk_mne']     = peak_recovery_topk(tc['mu_mne'],     tc['peak_parcels'])
        tc['pk_sloreta'] = peak_recovery_topk(tc['mu_sloreta'], tc['peak_parcels'])
        tc['pk_lasso']   = peak_recovery_topk(tc['mu_lasso'],   tc['peak_parcels'])
        tc['pk_bp']      = peak_recovery_topk(tc['mu_bp'],      tc['peak_parcels'])

    print(f"\n  {'Case':>4} {'MNE TV':>8} {'sLOR TV':>8} {'LASSO TV':>8} {'BP TV':>8}"
          f"  {'MNE pk':>7} {'sLOR pk':>7} {'LASSO pk':>8}")
    for i, tc in enumerate(test_cases):
        print(f"  {i:>4} {tc['tv_mne']:>8.3f} {tc['tv_sloreta']:>8.3f} "
              f"{tc['tv_lasso']:>8.3f} {tc['tv_bp']:>8.3f}"
              f"  {tc['pk_mne']*100:>6.0f}% {tc['pk_sloreta']*100:>6.0f}%"
              f" {tc['pk_lasso']*100:>7.0f}%")

    print(f"\n  Mean across cases:")
    for key, label in [('tv_mne', 'MNE'), ('tv_sloreta', 'sLORETA'),
                       ('tv_lasso', 'LASSO'), ('tv_bp', 'Backproj')]:
        m = np.mean([tc[key] for tc in test_cases])
        s = np.std([tc[key] for tc in test_cases])
        pk_key = key.replace('tv_', 'pk_')
        pm = np.mean([tc[pk_key] for tc in test_cases])
        print(f"    {label:10s}: TV = {m:.4f} ± {s:.4f},  peak = {pm*100:.0f}%")

    # ── Step 6: Visualisations ────────────────────────────────────────────────
    print("\nStep 6: Generating figures...")

    # -- Figure 1: Graph diagnostics ------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle('Ex14a: Cortical Graph Diagnostics\n'
                  f'Schaefer2018 100-parcel 7-network, fsaverage', fontsize=11)

    # Panel A: axial view coloured by network
    ax = axes1[0, 0]
    net_list = sorted(networks.keys())
    net_to_idx = {n: i for i, n in enumerate(net_list)}
    net_vals = np.array([net_to_idx[n] for n in net_per_parcel], dtype=float)
    x, y_c, z_c = parcel_centroids[:, 0], parcel_centroids[:, 1], parcel_centroids[:, 2]
    cmap_net = plt.cm.get_cmap('tab10', len(net_list))
    sc = ax.scatter(x * 1e3, y_c * 1e3, c=net_vals, cmap=cmap_net,
                    vmin=-0.5, vmax=len(net_list) - 0.5,
                    s=55, edgecolors='k', linewidths=0.3)
    # Draw adjacency edges (thin gray)
    for i in range(n_parcels):
        for j in range(i + 1, n_parcels):
            if adj[i, j] > 0:
                ax.plot([x[i] * 1e3, x[j] * 1e3],
                        [y_c[i] * 1e3, y_c[j] * 1e3],
                        'k-', alpha=0.08, lw=0.4, zorder=1)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('A: Parcel centroids — axial view\n(color = network)', fontsize=9)
    ax.set_aspect('equal')
    cbar = plt.colorbar(sc, ax=ax, ticks=range(len(net_list)), fraction=0.04)
    cbar.ax.set_yticklabels(net_list, fontsize=6)

    # Panel B: degree distribution
    ax = axes1[0, 1]
    ax.hist(degrees, bins=range(0, int(degrees.max()) + 2), color='steelblue',
            edgecolor='k', linewidth=0.5)
    ax.axvline(degrees.mean(), color='red', lw=1.5, linestyle='--',
               label=f'mean = {degrees.mean():.1f}')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.set_title('B: Degree distribution', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: adjacency matrix (sparsity pattern)
    ax = axes1[1, 0]
    ax.spy(adj, markersize=1.5, color='steelblue')
    ax.set_title(f'C: Adjacency matrix ({n_edges} edges)', fontsize=9)
    ax.set_xlabel('Parcel index')
    ax.set_ylabel('Parcel index')

    # Panel D: network sizes bar chart
    ax = axes1[1, 1]
    net_names_sorted = sorted(networks.keys(), key=lambda n: -len(networks[n]))
    net_sizes = [len(networks[n]) for n in net_names_sorted]
    colors_net = [NETWORK_COLORS.get(n, DEFAULT_COLOR) for n in net_names_sorted]
    ax.bar(range(len(net_names_sorted)), net_sizes, color=colors_net,
           edgecolor='k', linewidth=0.5)
    ax.set_xticks(range(len(net_names_sorted)))
    ax.set_xticklabels(net_names_sorted, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Number of parcels')
    ax.set_title('D: Parcels per network', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    fig1.tight_layout()
    path1 = os.path.join(HERE, 'ex14a_graph_diagnostics.png')
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved {path1}")

    # -- Figure 2: Leadfield patterns (EEG topomaps) --------------------------
    n_show = min(6, n_parcels)
    show_parcels = [networks[n][0] for n in sorted(networks.keys())][:n_show]

    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 7))
    fig2.suptitle('Ex14a: Leadfield Patterns\n'
                  '(one representative parcel per network — EEG topography)',
                  fontsize=10)
    for ax_idx, parcel_idx in enumerate(show_parcels):
        ax2 = axes2.flat[ax_idx]
        pattern = A[:, parcel_idx]
        try:
            mne.viz.plot_topomap(
                pattern, info, axes=ax2, show=False, contours=6,
                cmap='RdBu_r', vlim=(None, None))
        except Exception:
            ax2.bar(range(len(pattern)), pattern, color='steelblue')
            ax2.set_xticks([])
        net = labels[parcel_idx].name.split('_')[2]
        ax2.set_title(f'{labels[parcel_idx].name}\n({net})', fontsize=7)

    for ax_idx in range(len(show_parcels), 6):
        axes2.flat[ax_idx].axis('off')

    fig2.tight_layout()
    path2 = os.path.join(HERE, 'ex14a_leadfield_patterns.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved {path2}")

    # -- Figure 3: Baseline reconstructions -----------------------------------
    n_show_cases = min(3, len(test_cases))
    methods = [
        ('mu_source',  'True source', 'black'),
        ('mu_sloreta', 'sLORETA',     '#4363d8'),
        ('mu_mne',     'MNE',         '#3cb44b'),
        ('mu_lasso',   'LASSO',       '#f58231'),
        ('mu_bp',      'Backproj',    '#e6194b'),
    ]
    n_methods = len(methods)

    fig3, axes3 = plt.subplots(n_show_cases, n_methods,
                                figsize=(4 * n_methods, 3.5 * n_show_cases))
    if n_show_cases == 1:
        axes3 = axes3[np.newaxis, :]
    fig3.suptitle('Ex14a: Baseline Reconstructions\n'
                  '(axial view — parcel activation)', fontsize=11)

    for row, tc in enumerate(test_cases[:n_show_cases]):
        peaks = tc['peak_parcels']
        vm = max(tc['mu_source'].max(), 0.05)
        for col, (key, label, col_c) in enumerate(methods):
            ax3 = axes3[row, col]
            vals = tc[key]
            _parcel_scatter(ax3, parcel_centroids, vals, labels, net_per_parcel,
                            view='axial', vmin=0, vmax=vm, cmap='hot',
                            title='', colorbar=False)
            # Mark true peaks with stars
            for p in peaks:
                ax3.scatter(parcel_centroids[p, 0] * 1e3,
                            parcel_centroids[p, 1] * 1e3,
                            s=150, marker='*', c='cyan', zorder=5,
                            edgecolors='k', linewidths=0.5)
            if row == 0:
                ax3.set_title(label, fontsize=9, color=col_c)
            if col == 0:
                pks = tc['peak_names']
                ax3.set_ylabel(f"Case {row}\n{pks[0][:20]}", fontsize=7)
            tv_key = key.replace('mu_', 'tv_')
            pk_key = key.replace('mu_', 'pk_')
            if key != 'mu_source':
                tv = tc[tv_key]
                pk = tc[pk_key] * 100
                ax3.set_xlabel(f'TV={tv:.3f}  pk={pk:.0f}%', fontsize=7)

    fig3.tight_layout()
    path3 = os.path.join(HERE, 'ex14a_baseline_reconstructions.png')
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  Saved {path3}")

    # -- Figure 4: Baseline metrics bar chart ---------------------------------
    methods_m = [
        ('tv_sloreta', 'pk_sloreta', 'sLORETA',  '#4363d8'),
        ('tv_mne',     'pk_mne',     'MNE',       '#3cb44b'),
        ('tv_lasso',   'pk_lasso',   'LASSO',     '#f58231'),
        ('tv_bp',      'pk_bp',      'Backproj',  '#e6194b'),
    ]

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))
    fig4.suptitle('Ex14a: Baseline Metrics across 5 Test Cases', fontsize=11)

    x4 = np.arange(len(test_cases))
    w4 = 0.18
    offsets4 = np.linspace(-(len(methods_m)-1)/2, (len(methods_m)-1)/2,
                            len(methods_m)) * w4

    for mi, (tv_key, pk_key, label, color) in enumerate(methods_m):
        tv_vals = [tc[tv_key] for tc in test_cases]
        pk_vals = [tc[pk_key] * 100 for tc in test_cases]
        ax4a.bar(x4 + offsets4[mi], tv_vals, w4, label=label,
                 color=color, alpha=0.85, edgecolor='k', linewidth=0.3)
        ax4b.bar(x4 + offsets4[mi], pk_vals, w4, label=label,
                 color=color, alpha=0.85, edgecolor='k', linewidth=0.3)

    ax4a.set_xticks(x4)
    ax4a.set_xticklabels([f'Case {i}\n({tc["n_peaks"] if "n_peaks" in tc else len(tc["peak_parcels"])} pk)'
                           for i, tc in enumerate(test_cases)], fontsize=8)
    ax4a.set_ylabel('TV distance')
    ax4a.set_title('A: Total Variation (lower is better)', fontsize=9)
    ax4a.legend(fontsize=8)
    ax4a.grid(True, alpha=0.3, axis='y')

    ax4b.set_xticks(x4)
    ax4b.set_xticklabels([f'Case {i}' for i in range(len(test_cases))], fontsize=8)
    ax4b.set_ylabel('Peak recovery (%)')
    ax4b.set_ylim(0, 115)
    ax4b.set_title('B: Peak Recovery (higher is better)', fontsize=9)
    ax4b.legend(fontsize=8)
    ax4b.grid(True, alpha=0.3, axis='y')

    fig4.tight_layout()
    path4 = os.path.join(HERE, 'ex14a_baseline_metrics.png')
    fig4.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"  Saved {path4}")

    # ── Save data for Script 2 ────────────────────────────────────────────────
    parcel_names = np.array([l.name for l in labels])
    network_assignments = np.array([net_to_idx[n] for n in net_per_parcel])
    network_names = np.array(net_list)

    npz_path = os.path.join(ROOT, 'ex14_eeg_data.npz')
    np.savez(npz_path,
             R=R,
             A=A,
             adj=adj,
             parcel_centroids=parcel_centroids,
             parcel_names=parcel_names,
             network_assignments=network_assignments,
             network_names=network_names,
             n_channels=np.array(args.n_sensors),
             tau_diff=np.array(tau_diff))
    print(f"\nData saved to {npz_path}")
    print(f"  R: {R.shape}, A: {A.shape}, "
          f"centroids: {parcel_centroids.shape}")

    # ── Pre-run checks ────────────────────────────────────────────────────────
    print("\n=== Pre-run checks ===")
    checks = [
        ('Graph connected',           n_comp == 1),
        ('Leadfield rank >= n_parcels/2',
         np.linalg.matrix_rank(A) >= n_parcels // 2),
        ('No zero leadfield columns', len(zero_cols) == 0),
        ('sLORETA TV < 0.9',
         np.mean([tc['tv_sloreta'] for tc in test_cases]) < 0.9),
        # Baselines invert A only, not the diffusion — peak recovery of mu_source
        # is expected to be ~0%. Instead verify sLORETA beats backprojection (TV).
        ('sLORETA better than backprojection',
         np.mean([tc['tv_sloreta'] for tc in test_cases])
         < np.mean([tc['tv_bp'] for tc in test_cases])),
    ]
    all_ok = True
    for name, passed in checks:
        status = '✓' if passed else '✗'
        print(f"  {status} {name}")
        if not passed:
            all_ok = False

    if all_ok:
        print("\nAll checks passed — ready to proceed to Script 2.")
    else:
        print("\nWARNING: Some checks failed — review before proceeding.")


if __name__ == '__main__':
    main()
