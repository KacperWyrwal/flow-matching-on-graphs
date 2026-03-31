"""
Experiment 12b: 5×5×5 Cube, Multi-Peak, Variable Diffusion Time.

Builds on Ex12 (4×4×4, single interior peak) by scaling to:
- 5×5×5 cube: 125 nodes, 98 boundary, 27 interior (depths 1–2)
- 1-3 peaks placed anywhere (boundary or interior)
- Variable tau_diff in [0.5, 2.0] as conditioning (context_dim=3)

Depth-2 recovery (single center node) is the stretch goal.
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
    CubeBoundaryDataset,
    train_flexible_conditional,
    sample_trajectory_flexible,
    get_device,
)
from meta_fm.model import rate_matrix_to_edge_index


# ── Source / observation helpers ──────────────────────────────────────────────

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


def compute_boundary_observation(mu_source, R, mask, tau_diff):
    """Diffuse source, observe boundary only (renormalized)."""
    mu_diffused = mu_source @ expm(tau_diff * R)
    mu_obs = mu_diffused * mask
    mu_obs = np.clip(mu_obs, 1e-12, None)
    mu_obs /= mu_obs.sum()
    return mu_obs


# ── Baselines ─────────────────────────────────────────────────────────────────

def baseline_naive(mu_obs, mask):
    result = mu_obs.copy()
    result[mask == 0] = mu_obs[mask == 1].mean()
    result /= result.sum()
    return result


def baseline_laplacian(mu_obs, mask, R):
    L = -R.copy()
    boundary_idx = np.where(mask == 1)[0]
    interior_idx = np.where(mask == 0)[0]
    L_II = L[np.ix_(interior_idx, interior_idx)]
    L_IB = L[np.ix_(interior_idx, boundary_idx)]
    u_B  = mu_obs[boundary_idx]
    u_I  = np.linalg.solve(L_II, -L_IB @ u_B)
    result = mu_obs.copy()
    result[interior_idx] = u_I
    result = np.clip(result, 0, None)
    s = result.sum()
    return result / s if s > 1e-12 else result


# ── Metrics ───────────────────────────────────────────────────────────────────

def peak_recovery_topk(recovered, true_peaks):
    k = len(true_peaks)
    top_k = set(np.argsort(recovered)[-k:].tolist())
    return len(top_k & set(true_peaks)) / k


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_test_case(model, R, edge_index, mask, depth,
                       mu_source, peak_nodes, tau_diff, device):
    N = R.shape[0]
    interior_idx = np.where(mask == 0)[0]

    mu_obs = compute_boundary_observation(mu_source, R, mask, tau_diff)
    context = np.stack([mu_obs, mask, np.full(N, tau_diff)], axis=-1)
    mu_start = np.ones(N) / N

    _, traj = sample_trajectory_flexible(
        model, mu_start, context, edge_index, n_steps=200, device=device)
    mu_learned = traj[-1]
    mu_lap   = baseline_laplacian(mu_obs, mask, R)
    mu_naive = baseline_naive(mu_obs, mask)

    # Classify each peak by depth
    peak_depths = [int(depth[p]) for p in peak_nodes]

    # Per-depth MAE
    depths_all = sorted(set(depth.astype(int).tolist()))
    mae_d_learned = {d: float(np.mean(np.abs(mu_learned[depth == d] - mu_source[depth == d])))
                     for d in depths_all}
    mae_d_lap     = {d: float(np.mean(np.abs(mu_lap[depth == d]     - mu_source[depth == d])))
                     for d in depths_all}
    mae_d_naive   = {d: float(np.mean(np.abs(mu_naive[depth == d]   - mu_source[depth == d])))
                     for d in depths_all}

    return {
        'mu_source':  mu_source,
        'mu_obs':     mu_obs,
        'mu_learned': mu_learned,
        'mu_lap':     mu_lap,
        'mu_naive':   mu_naive,
        'peak_nodes': peak_nodes,
        'peak_depths': peak_depths,
        'tau_diff':   tau_diff,
        'n_peaks':    len(peak_nodes),
        'tv_learned': total_variation(mu_learned, mu_source),
        'tv_lap':     total_variation(mu_lap,     mu_source),
        'tv_naive':   total_variation(mu_naive,   mu_source),
        'tv_int_learned': total_variation(mu_learned[interior_idx], mu_source[interior_idx]),
        'tv_int_lap':     total_variation(mu_lap[interior_idx],     mu_source[interior_idx]),
        'tv_int_naive':   total_variation(mu_naive[interior_idx],   mu_source[interior_idx]),
        'pk_learned': peak_recovery_topk(mu_learned, peak_nodes),
        'pk_lap':     peak_recovery_topk(mu_lap,     peak_nodes),
        'pk_naive':   peak_recovery_topk(mu_naive,   peak_nodes),
        'mae_d_learned': mae_d_learned,
        'mae_d_lap':     mae_d_lap,
        'mae_d_naive':   mae_d_naive,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-weighting', type=str, default='uniform',
                        choices=['original', 'uniform', 'linear'])
    parser.add_argument('--n-epochs',      type=int,   default=1000)
    parser.add_argument('--hidden-dim',    type=int,   default=128)
    parser.add_argument('--n-layers',      type=int,   default=6)
    parser.add_argument('--lr',            type=float, default=5e-4)
    parser.add_argument('--cube-size',     type=int,   default=5)
    parser.add_argument('--n-train-dists', type=int,   default=300)
    parser.add_argument('--n-samples',     type=int,   default=15000)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    device = get_device()

    size         = args.cube_size
    N            = size ** 3
    R            = make_cube_graph(size)
    mask         = cube_boundary_mask(size)
    depth        = cube_node_depth(size)
    interior_idx = np.where(mask == 0)[0]
    depths_all   = sorted(set(depth.astype(int).tolist()))
    n_boundary   = int(mask.sum())
    n_interior   = N - n_boundary
    depth_counts = {d: int((depth == d).sum()) for d in depths_all}

    print(f"=== Experiment 12b: {size}×{size}×{size} Cube, Multi-Peak ===")
    print(f"Graph: {N} nodes, {n_boundary} boundary, {n_interior} interior")
    print(f"Depths: " + ", ".join(f"{d} ({depth_counts[d]} nodes)" for d in depths_all))
    print()

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        checkpoint_dir,
        f'meta_model_ex12b_cube5_{args.loss_weighting}_{args.n_epochs}ep'
        f'_h{args.hidden_dim}_l{args.n_layers}_s{size}.pt')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FlexibleConditionalGNNRateMatrixPredictor(
        context_dim=3, hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"Loaded checkpoint from {ckpt_path}")
        losses = None
    else:
        print(f"Generating {args.n_train_dists} training distributions...")
        train_sources, train_obs, train_tds = [], [], []
        for _ in range(args.n_train_dists):
            n_peaks  = int(rng.integers(1, 4))
            mu_src, _ = make_cube_multipeak(N, n_peaks, rng)
            td        = float(rng.uniform(0.5, 2.0))
            mu_obs    = compute_boundary_observation(mu_src, R, mask, td)
            train_sources.append(mu_src)
            train_obs.append(mu_obs)
            train_tds.append(td)

        print(f"Building CubeBoundaryDataset ({args.n_samples} samples)...")
        dataset = CubeBoundaryDataset(
            R=R, mask=mask,
            clean_distributions=train_sources,
            boundary_observations=train_obs,
            tau_diffs=train_tds,
            n_samples=args.n_samples,
            start_from='uniform',
            seed=42)
        print(f"Dataset built: {len(dataset)} samples")

        print(f"Training ({args.n_epochs} epochs, lr={args.lr}, "
              f"hidden={args.hidden_dim}, layers={args.n_layers})...")
        history = train_flexible_conditional(
            model, dataset,
            n_epochs=args.n_epochs,
            batch_size=256,
            lr=args.lr,
            device=device,
            loss_weighting=args.loss_weighting,
        )
        losses = history['losses']
        print(f"Training: initial loss = {losses[0]:.6f}, "
              f"final loss = {losses[-1]:.6f}")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    model.eval()

    # ── Structured evaluation: 3 tau_diff × 3 n_peaks × 10 cases = 90 ────────
    rng_eval    = np.random.default_rng(99)
    TAU_DIFFS   = [0.5, 1.0, 1.5]
    N_PEAKS_ALL = [1, 2, 3]
    N_PER_CELL  = 10
    edge_index  = rate_matrix_to_edge_index(R)

    # results[tau_diff][n_peaks] = list of result dicts
    results = {}
    for td in TAU_DIFFS:
        results[td] = {}
        for np_ in N_PEAKS_ALL:
            results[td][np_] = []
            for _ in range(N_PER_CELL):
                mu_src, peak_nodes = make_cube_multipeak(N, np_, rng_eval)
                r = evaluate_test_case(
                    model, R, edge_index, mask, depth,
                    mu_src, peak_nodes, td, device)
                results[td][np_].append(r)

    all_r = [r for td in TAU_DIFFS for np_ in N_PEAKS_ALL for r in results[td][np_]]

    # ── Console output ────────────────────────────────────────────────────────
    def mean_r(key, rs=None):
        rs = rs or all_r
        return np.mean([r[key] for r in rs])

    print(f"\n=== Experiment 12b: {size}×{size}×{size} Cube, Multi-Peak ===")
    print(f"Graph: {N} nodes, {n_boundary} boundary, {n_interior} interior")
    print(f"Depths: " + ", ".join(f"{d} ({depth_counts[d]} nodes)" for d in depths_all))

    print(f"\nFull TV by n_peaks:")
    for np_ in N_PEAKS_ALL:
        rs = [r for td in TAU_DIFFS for r in results[td][np_]]
        print(f"  {np_} peak(s): learned={mean_r('tv_learned', rs):.4f}, "
              f"laplacian={mean_r('tv_lap', rs):.4f}")

    print(f"\nInterior TV by n_peaks:")
    for np_ in N_PEAKS_ALL:
        rs = [r for td in TAU_DIFFS for r in results[td][np_]]
        print(f"  {np_} peak(s): learned={mean_r('tv_int_learned', rs):.4f}, "
              f"laplacian={mean_r('tv_int_lap', rs):.4f}")

    # Peak recovery by peak depth
    def peak_recovery_by_depth(all_results, key, target_depth):
        """Fraction of peaks at target_depth recovered by method `key`."""
        found, total = 0.0, 0
        for r in all_results:
            for i, p in enumerate(r['peak_nodes']):
                if r['peak_depths'][i] == target_depth:
                    total += 1
                    recovered = r[key]
                    k = len(r['peak_nodes'])
                    top_k = set(np.argsort(recovered)[-k:].tolist())
                    found += float(p in top_k)
        return found / total if total > 0 else float('nan'), total

    print(f"\nPeak recovery by location:")
    for d in depths_all:
        label = 'Boundary peaks' if d == 0 else (
                'Depth-2 peaks'  if d == depths_all[-1] else f'Depth-{d} peaks')
        pk_l, cnt = peak_recovery_by_depth(all_r, 'mu_learned', d)
        pk_p, _   = peak_recovery_by_depth(all_r, 'mu_lap',     d)
        print(f"  {label} ({cnt} total): "
              f"learned={pk_l*100:.0f}%, laplacian={pk_p*100:.0f}%")

    print(f"\nDepth MAE:")
    for d in depths_all:
        l  = np.mean([r['mae_d_learned'][d] for r in all_r])
        lp = np.mean([r['mae_d_lap'][d]     for r in all_r])
        nv = np.mean([r['mae_d_naive'][d]   for r in all_r])
        label = " (boundary)" if d == 0 else (
                " (center)"   if d == depths_all[-1] else "")
        print(f"  Depth {d}{label}: learned={l:.4f}, "
              f"laplacian={lp:.4f}, naive={nv:.4f}")

    # Comparison to 4×4×4: single interior peak, tau=1.0
    rs_1pk_t1 = [r for r in results[1.0][1]
                 if any(r['peak_depths'][i] > 0 for i in range(len(r['peak_nodes'])))]
    if rs_1pk_t1:
        print(f"\nComparison to 4×4×4 (single interior peak, tau=1.0):")
        print(f"  5×5×5 interior TV (learned):    "
              f"{np.mean([r['tv_int_learned'] for r in rs_1pk_t1]):.4f}")
        print(f"  5×5×5 interior TV (laplacian):  "
              f"{np.mean([r['tv_int_lap'] for r in rs_1pk_t1]):.4f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f'Experiment 12b: {size}³ Cube Boundary — Multi-Peak, Variable τ_diff\n'
        f'{n_boundary} observed / {n_interior} interior — context_dim=3',
        fontsize=11)
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.40)

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

    # ── Panel B: Interior node values — depth-1 and depth-2 examples ──────────
    ax_B = fig.add_subplot(gs[0, 1])
    ax_B.axis('off')
    ax_B.set_title('B: Interior Node Values (27 nodes)', pad=12)
    inner_B = gs[0, 1].subgridspec(2, 1, hspace=0.7)

    # Find one depth-1 and one depth-2 peak case (tau=1.0, 1 peak)
    case_d1 = next((r for r in results[1.0][1] if r['peak_depths'][0] == 1), all_r[0])
    case_d2 = next((r for r in results[1.0][1] if r['peak_depths'][0] == 2), all_r[1])

    for row_i, (case, label) in enumerate([(case_d1, 'depth-1 peak'),
                                            (case_d2, 'depth-2 (center) peak')]):
        ax_in = fig.add_subplot(inner_B[row_i])
        x_int = np.arange(n_interior)
        w = 0.25
        ax_in.bar(x_int - w, case['mu_source'][interior_idx],  w,
                  color='dimgray',     alpha=0.9,  label='True')
        ax_in.bar(x_int,      case['mu_learned'][interior_idx], w,
                  color='steelblue',   alpha=0.85, label='Learned')
        ax_in.bar(x_int + w,  case['mu_lap'][interior_idx],     w,
                  color='forestgreen', alpha=0.85, label='Laplacian')
        pk = case['peak_nodes'][0]
        pk_pos = int(np.where(interior_idx == pk)[0][0]) if pk in interior_idx else -1
        if pk_pos >= 0:
            ax_in.axvline(pk_pos, color='red', lw=1.1, linestyle='--', alpha=0.7)
        ax_in.set_title(f'Example: {label}', fontsize=8)
        ax_in.set_xticks([])
        ax_in.set_ylabel('Mass', fontsize=7)
        if row_i == 0:
            ax_in.legend(fontsize=7, loc='upper right')

    # ── Panel C: Full TV by n_peaks ───────────────────────────────────────────
    ax_C = fig.add_subplot(gs[0, 2])
    methods = ['Learned', 'Laplacian', 'Naive']
    colors  = ['steelblue', 'forestgreen', 'salmon']
    x_C = np.arange(len(N_PEAKS_ALL))
    w_C = 0.25
    for mi, (key, color) in enumerate(zip(['tv_learned', 'tv_lap', 'tv_naive'], colors)):
        vals = [mean_r(key, [r for td in TAU_DIFFS for r in results[td][np_]])
                for np_ in N_PEAKS_ALL]
        ax_C.bar(x_C + (mi - 1) * w_C, vals, w_C, label=methods[mi],
                 color=color, alpha=0.85)
    ax_C.set_xticks(x_C)
    ax_C.set_xticklabels([f'{np_} peak(s)' for np_ in N_PEAKS_ALL])
    ax_C.set_ylabel('Mean TV (full)')
    ax_C.set_title('C: Full TV by n_peaks')
    ax_C.legend(fontsize=8)
    ax_C.grid(True, alpha=0.3, axis='y')

    # ── Panel D: Interior TV by n_peaks ───────────────────────────────────────
    ax_D = fig.add_subplot(gs[1, 0])
    for mi, (key, color) in enumerate(zip(
            ['tv_int_learned', 'tv_int_lap', 'tv_int_naive'], colors)):
        vals = [mean_r(key, [r for td in TAU_DIFFS for r in results[td][np_]])
                for np_ in N_PEAKS_ALL]
        ax_D.bar(x_C + (mi - 1) * w_C, vals, w_C, label=methods[mi],
                 color=color, alpha=0.85)
    ax_D.set_xticks(x_C)
    ax_D.set_xticklabels([f'{np_} peak(s)' for np_ in N_PEAKS_ALL])
    ax_D.set_ylabel(f'Mean TV (interior, {n_interior} nodes)')
    ax_D.set_title('D: Interior TV by n_peaks')
    ax_D.legend(fontsize=8)
    ax_D.grid(True, alpha=0.3, axis='y')

    # ── Panel E: Reconstruction error vs depth ────────────────────────────────
    ax_E = fig.add_subplot(gs[1, 1])
    mae_l  = [np.mean([r['mae_d_learned'][d] for r in all_r]) for d in depths_all]
    mae_lp = [np.mean([r['mae_d_lap'][d]     for r in all_r]) for d in depths_all]
    mae_nv = [np.mean([r['mae_d_naive'][d]   for r in all_r]) for d in depths_all]
    ax_E.plot(depths_all, mae_l,  'o-',  color='steelblue',   lw=2, ms=7, label='Learned')
    ax_E.plot(depths_all, mae_lp, 's--', color='forestgreen',  lw=2, ms=7, label='Laplacian')
    ax_E.plot(depths_all, mae_nv, '^:',  color='salmon',       lw=2, ms=7, label='Naive')
    ax_E.set_xlabel('Depth from boundary')
    ax_E.set_ylabel('Mean abs error per node')
    ax_E.set_xticks(depths_all)
    ax_E.set_xticklabels([f'Depth {d}' + (' (center)' if d == depths_all[-1] else '')
                          for d in depths_all], fontsize=8)
    ax_E.set_title('E: Reconstruction Error vs Depth')
    ax_E.legend(fontsize=9)
    ax_E.grid(True, alpha=0.3)

    # ── Panel F: Peak recovery by peak depth ──────────────────────────────────
    ax_F = fig.add_subplot(gs[1, 2])
    depth_labels = ['Boundary\n(depth 0)'] + \
                   [f'Interior\ndepth {d}' for d in depths_all[1:]]
    x_F = np.arange(len(depths_all))
    w_F = 0.35
    pk_learned_by_d = [peak_recovery_by_depth(all_r, 'mu_learned', d)[0] * 100
                       for d in depths_all]
    pk_lap_by_d     = [peak_recovery_by_depth(all_r, 'mu_lap',     d)[0] * 100
                       for d in depths_all]
    b1 = ax_F.bar(x_F - w_F/2, pk_learned_by_d, w_F,
                  color='steelblue',   alpha=0.85, label='Learned')
    b2 = ax_F.bar(x_F + w_F/2, pk_lap_by_d,     w_F,
                  color='forestgreen', alpha=0.85, label='Laplacian')
    for bar, val in zip(list(b1) + list(b2),
                        pk_learned_by_d + pk_lap_by_d):
        if not np.isnan(val):
            ax_F.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                      f'{val:.0f}%', ha='center', va='bottom', fontsize=7)
    ax_F.set_xticks(x_F)
    ax_F.set_xticklabels(depth_labels, fontsize=8)
    ax_F.set_ylim(0, 115)
    ax_F.set_ylabel('Peak recovery (%)')
    ax_F.set_title('F: Peak Recovery by Peak Depth')
    ax_F.legend(fontsize=9)
    ax_F.grid(True, alpha=0.3, axis='y')

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'ex12b_cube5_boundary.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
