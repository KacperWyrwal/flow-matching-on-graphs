"""
Experiment 13: Sparse Sensors on 5×5×5 Cube.

Builds on Ex12b by replacing full boundary observation with 20 sparse sensors
that measure linear mixtures of source activations through a mixing matrix.

Key additions vs Ex12b:
- 20 sensors (vs 98 boundary nodes)
- M×N mixing matrix A: sensor m measures weighted average of nearby sources
- Backprojection A^T y used as context (not the mask/boundary values)
- Baselines: Backprojection, MNE (min-norm estimate), LASSO
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.linalg import expm

from graph_ot_fm import (
    total_variation,
    make_cube_graph,
    cube_boundary_mask,
    cube_node_depth,
)
from meta_fm import (
    FlexibleConditionalGNNRateMatrixPredictor,
    FiLMConditionalGNNRateMatrixPredictor,
    DirectGNNPredictor,
    CubeBoundaryDataset,
    train_flexible_conditional,
    train_film_conditional,
    train_direct_gnn,
    sample_trajectory_flexible,
    sample_trajectory_film,
    get_device,
)
from meta_fm.model import rate_matrix_to_edge_index


# ── Sensor placement ──────────────────────────────────────────────────────────

def manhattan_3d(node_a, node_b, size):
    ax, ay, az = node_a // (size*size), (node_a // size) % size, node_a % size
    bx, by, bz = node_b // (size*size), (node_b // size) % size, node_b % size
    return abs(ax-bx) + abs(ay-by) + abs(az-bz)


def place_sensors(size=5, n_sensors=20, seed=42):
    """
    Place sensors at a well-spread subset of boundary nodes using
    farthest-point sampling with Manhattan distance.
    """
    rng = np.random.default_rng(seed)
    mask = cube_boundary_mask(size)
    boundary_nodes = np.where(mask == 1)[0]

    selected = [int(rng.choice(boundary_nodes))]
    for _ in range(n_sensors - 1):
        dists = []
        for b in boundary_nodes:
            if b in selected:
                dists.append(0)
            else:
                min_d = min(manhattan_3d(int(b), s, size) for s in selected)
                dists.append(min_d)
        selected.append(int(boundary_nodes[np.argmax(dists)]))
    return selected


def compute_mixing_matrix(sensor_nodes, N, size, sigma=1.5):
    """
    Compute the M × N mixing matrix A.
    A[m, n] = exp(-d(sensor_m, source_n)^2 / (2*sigma^2)), row-normalized.
    """
    M = len(sensor_nodes)
    A = np.zeros((M, N))
    for m, s_node in enumerate(sensor_nodes):
        sx = s_node // (size * size)
        sy = (s_node // size) % size
        sz = s_node % size
        for n in range(N):
            nx = n // (size * size)
            ny = (n // size) % size
            nz = n % size
            d_sq = (sx-nx)**2 + (sy-ny)**2 + (sz-nz)**2
            A[m, n] = np.exp(-d_sq / (2 * sigma**2))
        A[m] /= A[m].sum()
    return A


# ── Observation model ─────────────────────────────────────────────────────────

def compute_sensor_observation(mu_source, R, A, tau_diff):
    """Diffuse source, apply mixing matrix → sensor readings y (M,)."""
    mu_diffused = mu_source @ expm(tau_diff * R)
    y = A @ mu_diffused
    return y


def backproject(y, A):
    """Backproject sensor readings to node space and normalize."""
    mu_bp = A.T @ y
    mu_bp = np.clip(mu_bp, 0, None)
    s = mu_bp.sum()
    return mu_bp / s if s > 1e-12 else np.ones(len(mu_bp)) / len(mu_bp)


def build_sensor_context(y, sensor_nodes, N, tau_diff):
    """
    Build per-node and global context from sensor readings.

    Returns:
        node_context: (N, 2) — [sensor_val * is_sensor, is_sensor]
        global_cond:  (M+1,) — [y (M dims), tau_diff (1 dim)]
    """
    sensor_vals = np.zeros(N)
    is_sensor = np.zeros(N)
    for m, node in enumerate(sensor_nodes):
        sensor_vals[node] = y[m]
        is_sensor[node] = 1.0
    node_context = np.stack([sensor_vals, is_sensor], axis=-1)  # (N, 2)
    global_cond = np.concatenate([y, [tau_diff]])               # (M+1,)
    return node_context, global_cond


# ── Source distribution ────────────────────────────────────────────────────────

def make_cube_multipeak(N, n_peaks, rng):
    """1-3 peaks placed anywhere on the N-node cube."""
    peak_nodes = rng.choice(N, size=n_peaks, replace=False).tolist()
    weights = rng.dirichlet(np.full(n_peaks, 2.0))
    dist = np.ones(N) * 0.2 / N
    for node, w in zip(peak_nodes, weights):
        dist[node] += 0.8 * w
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist, peak_nodes


# ── Baselines ─────────────────────────────────────────────────────────────────

def baseline_backproj(y, A):
    mu = A.T @ y
    mu = np.clip(mu, 0, None)
    s = mu.sum()
    return mu / s if s > 1e-12 else np.ones(A.shape[1]) / A.shape[1]


def baseline_mne(y, A, lam=1e-3):
    """mu_hat = A^T (A A^T + lambda I)^{-1} y"""
    M = A.shape[0]
    mu = A.T @ np.linalg.solve(A @ A.T + lam * np.eye(M), y)
    mu = np.clip(mu, 0, None)
    s = mu.sum()
    return mu / s if s > 1e-12 else np.ones(A.shape[1]) / A.shape[1]


def baseline_lasso(y, A, alpha=0.01):
    """Sparse reconstruction via LASSO."""
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=alpha, positive=True, max_iter=5000, tol=1e-4)
    model.fit(A, y)
    mu = model.coef_
    mu = np.clip(mu, 0, None)
    s = mu.sum()
    return mu / s if s > 1e-12 else np.ones(A.shape[1]) / A.shape[1]


def tune_baselines(R, A, size, n_cases=20, seed=77):
    """Tune MNE lambda and LASSO alpha on a small validation set."""
    rng = np.random.default_rng(seed)
    N = R.shape[0]
    depth = cube_node_depth(size)

    val_cases = []
    for _ in range(n_cases):
        n_peaks = int(rng.integers(1, 4))
        mu_src, _ = make_cube_multipeak(N, n_peaks, rng)
        tau_diff = float(rng.uniform(0.5, 2.0))
        y = compute_sensor_observation(mu_src, R, A, tau_diff)
        val_cases.append((mu_src, y))

    lambdas = [1e-4, 1e-3, 1e-2, 1e-1]
    alphas  = [1e-4, 1e-3, 1e-2, 1e-1]

    best_lam, best_lam_tv = lambdas[0], float('inf')
    for lam in lambdas:
        tvs = [total_variation(baseline_mne(y, A, lam), mu)
               for mu, y in val_cases]
        mean_tv = float(np.mean(tvs))
        if mean_tv < best_lam_tv:
            best_lam_tv = mean_tv
            best_lam = lam

    best_alpha, best_alpha_tv = alphas[0], float('inf')
    for alpha in alphas:
        tvs = [total_variation(baseline_lasso(y, A, alpha), mu)
               for mu, y in val_cases]
        mean_tv = float(np.mean(tvs))
        if mean_tv < best_alpha_tv:
            best_alpha_tv = mean_tv
            best_alpha = alpha

    return best_lam, best_alpha


# ── Metrics ───────────────────────────────────────────────────────────────────

def peak_recovery_topk(recovered, true_peaks):
    k = len(true_peaks)
    top_k = set(np.argsort(recovered)[-k:].tolist())
    return len(top_k & set(true_peaks)) / k


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_test_case(model, R, edge_index, A, sensor_nodes, depth,
                       mu_source, peak_nodes, tau_diff, device,
                       best_lam, best_alpha, direct_model=None,
                       conditioning='backproj'):
    N = R.shape[0]
    interior_idx = np.where(depth > 0)[0]

    y = compute_sensor_observation(mu_source, R, A, tau_diff)
    mu_bp = backproject(y, A)
    mu_start = np.ones(N) / N

    if conditioning == 'film':
        node_ctx, global_ctx = build_sensor_context(y, sensor_nodes, N, tau_diff)
        _, traj = sample_trajectory_film(
            model, mu_start, node_ctx, global_ctx, edge_index,
            n_steps=200, device=device)
    else:
        # Context: [mu_backproj(a), tau_diff] per node — context_dim=2
        context = np.stack([mu_bp, np.full(N, tau_diff)], axis=-1)
        _, traj = sample_trajectory_flexible(
            model, mu_start, context, edge_index, n_steps=200, device=device)
    mu_learned = traj[-1]
    mu_mne    = baseline_mne(y, A, lam=best_lam)
    mu_lasso  = baseline_lasso(y, A, alpha=best_alpha)
    mu_backpj = baseline_backproj(y, A)

    # Direct GNN prediction (optional) — uses backprojection context regardless
    if direct_model is not None:
        bp_ctx = np.stack([mu_bp, np.full(N, tau_diff)], axis=-1)
        ctx_t = torch.tensor(bp_ctx, dtype=torch.float32).to(device)
        ei_d  = edge_index.to(device)
        with torch.no_grad():
            mu_direct = direct_model(ctx_t, ei_d).cpu().numpy()
    else:
        mu_direct = None

    # Peak depth classification
    peak_depths = [int(depth[p]) for p in peak_nodes]

    # Per-depth MAE
    depths_all = sorted(set(depth.astype(int).tolist()))
    hats = [('learned', mu_learned), ('mne', mu_mne),
            ('lasso', mu_lasso), ('backproj', mu_backpj)]
    if mu_direct is not None:
        hats.append(('direct', mu_direct))
    mae_d = {key: {d: float(np.mean(np.abs(mu_hat[depth == d] - mu_source[depth == d])))
                   for d in depths_all}
             for key, mu_hat in hats}

    result = {
        'mu_source':   mu_source,
        'mu_bp':       mu_bp,
        'mu_learned':  mu_learned,
        'mu_mne':      mu_mne,
        'mu_lasso':    mu_lasso,
        'mu_backproj': mu_backpj,
        'y':           y,
        'peak_nodes':  peak_nodes,
        'peak_depths': peak_depths,
        'tau_diff':    tau_diff,
        'n_peaks':     len(peak_nodes),
        'tv_learned':  total_variation(mu_learned, mu_source),
        'tv_mne':      total_variation(mu_mne,     mu_source),
        'tv_lasso':    total_variation(mu_lasso,   mu_source),
        'tv_backproj': total_variation(mu_backpj,  mu_source),
        'tv_int_learned':  total_variation(mu_learned[interior_idx], mu_source[interior_idx]),
        'tv_int_mne':      total_variation(mu_mne[interior_idx],     mu_source[interior_idx]),
        'tv_int_lasso':    total_variation(mu_lasso[interior_idx],   mu_source[interior_idx]),
        'tv_int_backproj': total_variation(mu_backpj[interior_idx],  mu_source[interior_idx]),
        'pk_learned':  peak_recovery_topk(mu_learned, peak_nodes),
        'pk_mne':      peak_recovery_topk(mu_mne,     peak_nodes),
        'pk_lasso':    peak_recovery_topk(mu_lasso,   peak_nodes),
        'pk_backproj': peak_recovery_topk(mu_backpj,  peak_nodes),
        'mae_d': mae_d,
    }
    if mu_direct is not None:
        result['mu_direct']      = mu_direct
        result['tv_direct']      = total_variation(mu_direct, mu_source)
        result['tv_int_direct']  = total_variation(mu_direct[interior_idx],
                                                   mu_source[interior_idx])
        result['pk_direct']      = peak_recovery_topk(mu_direct, peak_nodes)
    return result


# ── Plot helpers ──────────────────────────────────────────────────────────────

def show_cube_slices(ax, mu, size, title, vmax=None):
    """Show z=0, z=mid, z=size-1 slices of a 5×5×5 distribution as images."""
    cube = mu.reshape(size, size, size)
    mid = size // 2
    slices = [cube[:, :, 0], cube[:, :, mid], cube[:, :, size-1]]
    labels = [f'z=0', f'z={mid}', f'z={size-1}']
    vm = vmax or max(s.max() for s in slices)

    inner = ax.get_subplotspec().subgridspec(1, 3, wspace=0.05)
    fig = ax.get_figure()
    ax.axis('off')
    ax.set_title(title, fontsize=8, pad=4)
    for col, (sl, lbl) in enumerate(zip(slices, labels)):
        a = fig.add_subplot(inner[col])
        a.imshow(sl, vmin=0, vmax=vm, cmap='hot', origin='lower')
        a.set_title(lbl, fontsize=6)
        a.set_xticks([])
        a.set_yticks([])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-weighting', type=str, default='uniform',
                        choices=['original', 'uniform', 'linear'])
    parser.add_argument('--n-sensors',     type=int,   default=20)
    parser.add_argument('--sensor-sigma',  type=float, default=1.5)
    parser.add_argument('--n-epochs',      type=int,   default=1000)
    parser.add_argument('--hidden-dim',    type=int,   default=128)
    parser.add_argument('--n-layers',      type=int,   default=6)
    parser.add_argument('--lr',            type=float, default=5e-4)
    parser.add_argument('--n-train-dists', type=int,   default=300)
    parser.add_argument('--n-samples',     type=int,   default=15000)
    parser.add_argument('--train-direct-gnn', action='store_true',
                        help='Also train and evaluate GNN+softmax direct baseline')
    parser.add_argument('--conditioning', type=str, default='film',
                        choices=['backproj', 'film'],
                        help='Conditioning method: backprojection only or FiLM + spatial')
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='EMA decay rate (0 to disable)')
    parser.add_argument('--loss-type', type=str, default='rate_kl',
                        choices=['rate_kl', 'mse'],
                        help='rate_kl: principled path-measure KL; mse: fallback')
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    device = get_device()

    size = 5
    N    = size ** 3
    R    = make_cube_graph(size)
    depth = cube_node_depth(size)
    interior_idx = np.where(depth > 0)[0]
    depths_all   = sorted(set(depth.astype(int).tolist()))
    depth_counts = {d: int((depth == d).sum()) for d in depths_all}
    n_boundary   = int((depth == 0).sum())
    n_interior   = N - n_boundary

    print(f"=== Experiment 13: Sparse Sensors on {size}×{size}×{size} Cube ===")
    print(f"Sensors: {args.n_sensors} (of {n_boundary} boundary nodes), "
          f"sigma={args.sensor_sigma}")
    print(f"Measurements: {args.n_sensors}-dimensional, sources: {N}-dimensional")

    # ── Sensor placement and mixing matrix ────────────────────────────────────
    sensor_nodes = place_sensors(size, args.n_sensors, seed=42)
    A = compute_mixing_matrix(sensor_nodes, N, size, sigma=args.sensor_sigma)

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_suffix = args.conditioning
    ckpt_path = os.path.join(
        checkpoint_dir,
        f'meta_model_ex13_{ckpt_suffix}_{args.loss_weighting}_{args.n_epochs}ep'
        f'_h{args.hidden_dim}_l{args.n_layers}_s{args.n_sensors}.pt')

    print(f"Conditioning: {args.conditioning}")

    global_dim = args.n_sensors + 1  # raw sensor vector + tau_diff

    # ── Model ──────────────────────────────────────────────────────────────────
    if args.conditioning == 'film':
        model = FiLMConditionalGNNRateMatrixPredictor(
            node_context_dim=2, global_dim=global_dim,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    else:
        model = FlexibleConditionalGNNRateMatrixPredictor(
            context_dim=2, hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    losses = None
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        # Generate training data
        print(f"\nGenerating {args.n_train_dists} training distributions...")
        train_sources, train_obs, train_tds = [], [], []
        per_count = args.n_train_dists // 3
        for n_peaks in [1, 2, 3]:
            for _ in range(per_count):
                mu_src, _ = make_cube_multipeak(N, n_peaks, rng)
                td = float(rng.uniform(0.5, 2.0))
                y = compute_sensor_observation(mu_src, R, A, td)
                mu_bp = backproject(y, A)
                train_sources.append(mu_src)
                train_obs.append(mu_bp)
                train_tds.append(td)

        print(f"Building dataset ({args.n_samples} samples)...")

        if args.conditioning == 'film':
            dataset = _SparseSensorFiLMDataset(
                R=R, sensor_nodes=sensor_nodes, mixing_matrix=A,
                clean_distributions=train_sources, tau_diffs=train_tds,
                n_samples=args.n_samples, seed=42)
        else:
            dataset = _SparseSensorDataset(
                R=R, clean_distributions=train_sources,
                backprojections=train_obs, tau_diffs=train_tds,
                n_samples=args.n_samples, seed=42)
        print(f"Dataset built: {len(dataset)} samples")

        print(f"Training ({args.n_epochs} epochs, lr={args.lr}, "
              f"hidden={args.hidden_dim}, layers={args.n_layers})...")

        if args.conditioning == 'film':
            history = train_film_conditional(
                model, dataset,
                n_epochs=args.n_epochs,
                batch_size=256,
                lr=args.lr,
                device=device,
                loss_weighting=args.loss_weighting,
                loss_type=args.loss_type,
                ema_decay=args.ema_decay,
            )
        else:
            history = train_flexible_conditional(
                model, dataset,
                n_epochs=args.n_epochs,
                batch_size=256,
                lr=args.lr,
                device=device,
                loss_weighting=args.loss_weighting,
                loss_type=args.loss_type,
                ema_decay=args.ema_decay,
            )
        losses = history['losses']
        print(f"Training: initial loss = {losses[0]:.6f}, "
              f"final loss = {losses[-1]:.6f}")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    model.eval()
    edge_index = rate_matrix_to_edge_index(R)

    # ── Direct GNN baseline ───────────────────────────────────────────────────
    direct_model = None
    direct_losses = None
    if args.train_direct_gnn:
        ckpt_direct = ckpt_path.replace('_ex13_sparse_', '_ex13_direct_')
        direct_model = DirectGNNPredictor(
            context_dim=2, hidden_dim=args.hidden_dim, n_layers=args.n_layers)

        if os.path.exists(ckpt_direct):
            direct_model.load_state_dict(torch.load(ckpt_direct, map_location='cpu'))
            print(f"Loaded direct-GNN checkpoint from {ckpt_direct}")
        else:
            # Build train_pairs: (context_np, mu_source, edge_index) for each training dist
            print(f"Training DirectGNNPredictor ({args.n_epochs} epochs)...")
            train_pairs = []
            rng_tp = np.random.default_rng(43)
            per_count = args.n_train_dists // 3
            for n_peaks in [1, 2, 3]:
                for _ in range(per_count):
                    mu_src, _ = make_cube_multipeak(N, n_peaks, rng_tp)
                    td = float(rng_tp.uniform(0.5, 2.0))
                    y = compute_sensor_observation(mu_src, R, A, td)
                    mu_bp = backproject(y, A)
                    ctx = np.stack([mu_bp, np.full(N, td)], axis=-1)
                    train_pairs.append((ctx, mu_src, edge_index))

            direct_hist = train_direct_gnn(
                direct_model, train_pairs,
                n_epochs=args.n_epochs, lr=args.lr,
                device=device, seed=0)
            direct_losses = direct_hist['losses']
            print(f"DirectGNN: initial KL = {direct_losses[0]:.6f}, "
                  f"final KL = {direct_losses[-1]:.6f}")
            torch.save(direct_model.state_dict(), ckpt_direct)
            print(f"DirectGNN checkpoint saved to {ckpt_direct}")

        direct_model = direct_model.to(device)
        direct_model.eval()

    # ── Baseline tuning ───────────────────────────────────────────────────────
    print("\nBaseline tuning...")
    best_lam, best_alpha = tune_baselines(R, A, size, n_cases=20, seed=77)
    print(f"  Best MNE lambda: {best_lam}")
    print(f"  Best LASSO alpha: {best_alpha}")

    # ── Structured evaluation: 3 tau_diff × 3 n_peaks × 10 cases = 90 ────────
    rng_eval    = np.random.default_rng(99)
    TAU_DIFFS   = [0.5, 1.0, 1.5]
    N_PEAKS_ALL = [1, 2, 3]
    N_PER_CELL  = 10

    results = {}
    for td in TAU_DIFFS:
        results[td] = {}
        for np_ in N_PEAKS_ALL:
            results[td][np_] = []
            for _ in range(N_PER_CELL):
                mu_src, peak_nodes = make_cube_multipeak(N, np_, rng_eval)
                r = evaluate_test_case(
                    model, R, edge_index, A, sensor_nodes, depth,
                    mu_src, peak_nodes, td, device,
                    best_lam, best_alpha,
                    direct_model=direct_model,
                    conditioning=args.conditioning)
                results[td][np_].append(r)

    all_r = [r for td in TAU_DIFFS for np_ in N_PEAKS_ALL for r in results[td][np_]]

    # ── Console output ────────────────────────────────────────────────────────
    def mr(key, rs=None):
        rs = rs or all_r
        vals = [r[key] for r in rs]
        return np.mean(vals), np.std(vals)

    tv_entries = [('tv_learned', 'Learned (flow)'), ('tv_lasso', 'LASSO'),
                  ('tv_mne', 'MNE'), ('tv_backproj', 'Backprojection')]
    if direct_model is not None:
        tv_entries.insert(1, ('tv_direct', 'GNN+softmax'))

    print(f"\nFull TV:")
    for key, label in tv_entries:
        m, s = mr(key)
        print(f"  {label:20s}: {m:.4f} ± {s:.4f}")

    int_tv_entries = [('tv_int_learned', 'Learned (flow)'), ('tv_int_lasso', 'LASSO'),
                      ('tv_int_mne', 'MNE'), ('tv_int_backproj', 'Backprojection')]
    if direct_model is not None:
        int_tv_entries.insert(1, ('tv_int_direct', 'GNN+softmax'))

    print(f"\nInterior TV:")
    for key, label in int_tv_entries:
        m, s = mr(key)
        print(f"  {label:20s}: {m:.4f} ± {s:.4f}")

    def peak_recovery_by_depth(all_results, mu_key, target_depth):
        found, total = 0.0, 0
        for r in all_results:
            for i, p in enumerate(r['peak_nodes']):
                if r['peak_depths'][i] == target_depth:
                    total += 1
                    recovered = r[mu_key]
                    k = len(r['peak_nodes'])
                    top_k = set(np.argsort(recovered)[-k:].tolist())
                    found += float(p in top_k)
        return found / total if total > 0 else float('nan'), total

    print(f"\nPeak recovery by depth:")
    for d in depths_all:
        label = 'Boundary' if d == 0 else (
                'Depth-2 (center)' if d == depths_all[-1] else f'Depth-{d}')
        pk_l, cnt = peak_recovery_by_depth(all_r, 'mu_learned', d)
        pk_lo, _  = peak_recovery_by_depth(all_r, 'mu_lasso',   d)
        pk_m, _   = peak_recovery_by_depth(all_r, 'mu_mne',     d)
        line = (f"  {label:20s} ({cnt:3d} total): "
                f"learned={pk_l*100:.0f}%, lasso={pk_lo*100:.0f}%, "
                f"mne={pk_m*100:.0f}%")
        if direct_model is not None:
            pk_d, _ = peak_recovery_by_depth(all_r, 'mu_direct', d)
            line += f", gnn+sm={pk_d*100:.0f}%"
        print(line)

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f'Experiment 13: Sparse Sensors on {size}³ Cube '
        f'({args.n_sensors} sensors, σ={args.sensor_sigma})\n'
        f'{args.n_sensors}-dim measurements → {N}-dim source reconstruction',
        fontsize=11)
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.4)

    # ── Panel A: Training loss ─────────────────────────────────────────────────
    ax_A = fig.add_subplot(gs[0, 0])
    if losses is not None:
        ax_A.plot(losses, color='steelblue', lw=1.5)
        ax_A.set_yscale('log')
        ax_A.set_xlabel('Epoch')
        ax_A.set_ylabel('Loss')
    else:
        ax_A.text(0.5, 0.5, 'Loaded from\ncheckpoint',
                  transform=ax_A.transAxes, ha='center', va='center', fontsize=12)
    ax_A.set_title('A: Training Loss')
    ax_A.grid(True, alpha=0.3)

    # ── Panel B: Example reconstruction (cube slices) ─────────────────────────
    # Pick a 1-peak case at tau=1.0 with an interior peak
    example = next(
        (r for r in results[1.0][1] if any(d > 0 for d in r['peak_depths'])),
        results[1.0][1][0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_B.axis('off')
    ax_B.set_title('B: Example Reconstruction (z-slices, 1 interior peak)',
                   fontsize=8, pad=4)

    inner_B = gs[0, 1].subgridspec(4, 1, hspace=0.8)
    row_configs = [
        ('mu_bp',      'Backproj (input)'),
        ('mu_learned', 'Learned'),
        ('mu_lasso',   'LASSO'),
        ('mu_source',  'True source'),
    ]
    colors_B = ['gray', 'steelblue', 'darkorange', 'black']
    for row_i, (key, label) in enumerate(row_configs):
        cube = example[key].reshape(size, size, size)
        mid  = size // 2
        slices = [cube[:, :, 0], cube[:, :, mid], cube[:, :, size-1]]
        # Normalize each row independently so shape is visible regardless of magnitude
        vm = max(s.max() for s in slices)
        vm = vm if vm > 1e-12 else 1.0
        inner_inner = inner_B[row_i].subgridspec(1, 3, wspace=0.05)
        for col, (sl, zlbl) in enumerate(zip(slices, [f'z=0', f'z={mid}', f'z={size-1}'])):
            ax_bb = fig.add_subplot(inner_inner[col])
            ax_bb.imshow(sl, vmin=0, vmax=vm, cmap='hot', origin='lower')
            if col == 0:
                ax_bb.set_ylabel(label, fontsize=6, rotation=0, labelpad=30, va='center',
                                 color=colors_B[row_i])
            if row_i == 0:
                ax_bb.set_title(zlbl, fontsize=6)
            ax_bb.set_xticks([])
            ax_bb.set_yticks([])

    # ── Build method lists (extend with GNN+softmax if trained) ───────────────
    methods  = ['Learned', 'LASSO', 'MNE', 'Backproj']
    tv_keys  = ['tv_learned', 'tv_lasso', 'tv_mne', 'tv_backproj']
    itv_keys = ['tv_int_learned', 'tv_int_lasso', 'tv_int_mne', 'tv_int_backproj']
    pk_keys  = ['mu_learned', 'mu_lasso', 'mu_mne', 'mu_backproj']
    colors_m = ['steelblue', 'darkorange', 'forestgreen', 'salmon']
    ls_list  = ['-', '--', '-.', ':']
    if direct_model is not None:
        methods.insert(1,  'GNN+softmax')
        tv_keys.insert(1,  'tv_direct')
        itv_keys.insert(1, 'tv_int_direct')
        pk_keys.insert(1,  'mu_direct')
        colors_m.insert(1, 'mediumpurple')
        ls_list.insert(1,  (0, (3, 1, 1, 1)))

    n_methods = len(methods)

    # ── Panel C: Full TV by method and n_peaks ────────────────────────────────
    ax_C = fig.add_subplot(gs[0, 2])
    x_C = np.arange(len(N_PEAKS_ALL))
    w_C = 0.8 / n_methods
    offset_C = np.arange(n_methods) * w_C - (n_methods - 1) * w_C / 2
    for mi, (key, color, meth) in enumerate(zip(tv_keys, colors_m, methods)):
        vals = [np.mean([r[key] for td in TAU_DIFFS for r in results[td][np_]])
                for np_ in N_PEAKS_ALL]
        ax_C.bar(x_C + offset_C[mi], vals, w_C, label=meth, color=color, alpha=0.85)
    ax_C.set_xticks(x_C)
    ax_C.set_xticklabels([f'{np_} peak(s)' for np_ in N_PEAKS_ALL])
    ax_C.set_ylabel('Mean TV (full)')
    ax_C.set_title('C: Full TV by n_peaks')
    ax_C.legend(fontsize=6)
    ax_C.grid(True, alpha=0.3, axis='y')

    # ── Panel D: Interior TV by method ────────────────────────────────────────
    ax_D = fig.add_subplot(gs[1, 0])
    x_D = np.arange(n_methods)
    vals_D = [np.mean([r[key] for r in all_r]) for key in itv_keys]
    bars_D = ax_D.bar(x_D, vals_D, color=colors_m, alpha=0.85)
    for bar, val in zip(bars_D, vals_D):
        ax_D.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                  f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    ax_D.set_xticks(x_D)
    ax_D.set_xticklabels(methods, fontsize=8, rotation=15, ha='right')
    ax_D.set_ylabel(f'Mean TV (interior, {n_interior} nodes)')
    ax_D.set_title('D: Interior TV by Method')
    ax_D.grid(True, alpha=0.3, axis='y')

    # ── Panel E: Peak recovery by method and peak depth ───────────────────────
    ax_E = fig.add_subplot(gs[1, 1])
    x_E = np.arange(len(depths_all))
    w_E = 0.8 / n_methods
    offset_E = np.arange(n_methods) * w_E - (n_methods - 1) * w_E / 2
    depth_labels = (['Boundary\n(d=0)'] +
                    [f'Interior\n(d={d})' for d in depths_all[1:]])
    for mi, (mu_key, color, meth) in enumerate(zip(pk_keys, colors_m, methods)):
        vals = []
        for d in depths_all:
            pct, _ = peak_recovery_by_depth(all_r, mu_key, d)
            vals.append(pct * 100 if not np.isnan(pct) else 0)
        ax_E.bar(x_E + offset_E[mi], vals, w_E, label=meth, color=color, alpha=0.85)
    ax_E.set_xticks(x_E)
    ax_E.set_xticklabels(depth_labels, fontsize=8)
    ax_E.set_ylim(0, 115)
    ax_E.set_ylabel('Peak recovery (%)')
    ax_E.set_title('E: Peak Recovery by Depth\n(Learned vs LASSO: graph-aware advantage)')
    ax_E.legend(fontsize=6)
    ax_E.grid(True, alpha=0.3, axis='y')

    # ── Panel F: TV vs tau_diff ────────────────────────────────────────────────
    ax_F = fig.add_subplot(gs[1, 2])
    for mu_key, color, meth, ls in zip(pk_keys, colors_m, methods, ls_list):
        tv_key = mu_key.replace('mu_', 'tv_')
        vals = [np.mean([r[tv_key] for np_ in N_PEAKS_ALL for r in results[td][np_]])
                for td in TAU_DIFFS]
        ax_F.plot(TAU_DIFFS, vals, color=color, linestyle=ls, lw=2,
                  marker='o', ms=6, label=meth)
    ax_F.set_xlabel('τ_diff (diffusion time)')
    ax_F.set_ylabel('Mean Full TV')
    ax_F.set_title('F: TV vs Diffusion Time')
    ax_F.legend(fontsize=8)
    ax_F.grid(True, alpha=0.3)
    ax_F.set_xticks(TAU_DIFFS)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'ex13_sparse_sensors.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


# ── Thin dataset wrapper for backprojection context ───────────────────────────

class _SparseSensorDataset(torch.utils.data.Dataset):
    """
    Like CubeBoundaryDataset but context = [mu_backproj(a), tau_diff]
    (context_dim=2, no mask column).
    """

    def __init__(self, R, clean_distributions, backprojections, tau_diffs,
                 n_samples=10000, seed=42):
        from graph_ot_fm import GraphStructure
        from graph_ot_fm.geodesic_cache import GeodesicCache
        from graph_ot_fm.flow import marginal_distribution_fast, marginal_rate_matrix_fast
        from graph_ot_fm.ot_solver import compute_ot_coupling

        rng = np.random.default_rng(seed)
        N = R.shape[0]
        graph_struct = GraphStructure(R)
        cache = GeodesicCache(graph_struct)
        edge_index = rate_matrix_to_edge_index(R)

        mu_start = np.ones(N) / N  # always start from uniform

        pairs = []
        for mu_clean, mu_bp, td in zip(clean_distributions, backprojections, tau_diffs):
            pi = compute_ot_coupling(mu_start, mu_clean, graph_struct=graph_struct)
            cache.precompute_for_coupling(pi)
            pairs.append((mu_bp, td, mu_clean, pi))

        self.samples = []
        for _ in range(n_samples):
            mu_bp, td, mu_clean, pi = pairs[int(rng.integers(0, len(pairs)))]
            tau = float(rng.uniform(0.0, 0.999))

            mu_tau = marginal_distribution_fast(cache, pi, tau)
            R_target_np = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi, tau)
            context = np.stack([mu_bp, np.full(N, td)], axis=-1)  # (N, 2)

            self.samples.append((
                torch.tensor(mu_tau,       dtype=torch.float32),
                torch.tensor([tau],        dtype=torch.float32),
                torch.tensor(context,      dtype=torch.float32),
                torch.tensor(R_target_np,  dtype=torch.float32),
                edge_index,
                N,
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class _SparseSensorFiLMDataset(torch.utils.data.Dataset):
    """
    Training data for sparse sensor reconstruction with FiLM conditioning.

    Each sample includes:
        mu_tau:       (N,)    distribution at flow time
        tau:          (1,)    flow time
        node_context: (N, 2)  sensor values at sensor nodes + binary mask
        global_cond:  (M+1,)  raw sensor vector y + tau_diff
        R_target:     (N, N)  factorized target rate matrix
        edge_index:   (2, E)
        n_nodes:      int
    """

    def __init__(self, R, sensor_nodes, mixing_matrix,
                 clean_distributions, tau_diffs,
                 n_samples=15000, seed=42):
        from graph_ot_fm import GraphStructure
        from graph_ot_fm.geodesic_cache import GeodesicCache
        from graph_ot_fm.flow import marginal_distribution_fast, marginal_rate_matrix_fast
        from graph_ot_fm.ot_solver import compute_ot_coupling

        rng = np.random.default_rng(seed)
        N = R.shape[0]
        A = mixing_matrix
        graph_struct = GraphStructure(R)
        cache = GeodesicCache(graph_struct)
        edge_index = rate_matrix_to_edge_index(R)

        mu_start = np.ones(N) / N

        pairs = []
        for mu_clean, td in zip(clean_distributions, tau_diffs):
            mu_diffused = mu_clean @ expm(td * R)
            y = A @ mu_diffused

            node_ctx, global_ctx = build_sensor_context(y, sensor_nodes, N, td)

            pi = compute_ot_coupling(mu_start, mu_clean, graph_struct=graph_struct)
            cache.precompute_for_coupling(pi)
            pairs.append((mu_clean, node_ctx, global_ctx, pi))

        self.samples = []
        for _ in range(n_samples):
            mu_clean, node_ctx, global_ctx, pi = pairs[int(rng.integers(len(pairs)))]
            tau = float(rng.uniform(0.0, 0.999))

            mu_tau = marginal_distribution_fast(cache, pi, tau)
            R_target_np = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi, tau)

            self.samples.append((
                torch.tensor(mu_tau,      dtype=torch.float32),
                torch.tensor([tau],       dtype=torch.float32),
                torch.tensor(node_ctx,    dtype=torch.float32),
                torch.tensor(global_ctx,  dtype=torch.float32),
                torch.tensor(R_target_np, dtype=torch.float32),
                edge_index,
                N,
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == '__main__':
    main()
