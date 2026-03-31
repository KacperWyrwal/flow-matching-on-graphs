"""
Experiment 12c: Posterior Sampling on 5×5×5 Cube Boundary.

Adds posterior sampling to the Ex12b boundary observation setup. Trains with
a configurable prior over the simplex (Dirichlet or logistic-normal) so
inference can produce K posterior samples from K different starting points.

Key question: does posterior mean maintain 12b's quality while also providing
calibrated uncertainty estimates?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.linalg import expm
from scipy.stats import pearsonr

from graph_ot_fm import (
    total_variation,
    make_cube_graph,
    cube_boundary_mask,
    cube_node_depth,
)
from meta_fm import (
    FlexibleConditionalGNNRateMatrixPredictor,
    CubePosteriorDataset,
    train_flexible_conditional,
    sample_trajectory_flexible,
    get_device,
)
from meta_fm.model import rate_matrix_to_edge_index

# Reuse baselines from ex12b
from ex12b_cube5_boundary import (
    make_cube_multipeak,
    compute_boundary_observation,
    baseline_laplacian,
    baseline_naive,
    peak_recovery_topk,
)

# Reuse octant utilities from ex13b
from ex13b_posterior_sampling import (
    build_octant_regions,
    octant_of_node,
    conditional_analysis,
    pairwise_tv,
    posterior_summary,
)


# ── Init distribution helpers ─────────────────────────────────────────────────

def precompute_laplacian_pseudoinverse(R):
    """
    Compute the pseudo-inverse of the graph Laplacian L = -R.
    Used as covariance for the logistic-normal prior over the simplex.
    """
    L = -R.copy()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    pinv_eigenvalues = np.where(np.abs(eigenvalues) > 1e-10,
                                1.0 / eigenvalues, 0.0)
    return eigenvectors @ np.diag(pinv_eigenvalues) @ eigenvectors.T


def sample_init_distribution(N, rng, method='dirichlet', alpha=10.0,
                              beta=1.0, L_inv=None):
    """
    Sample a starting distribution from the chosen prior over the simplex.

    Args:
        N:      number of nodes
        rng:    numpy random generator
        method: 'dirichlet' or 'logistic-normal'
        alpha:  Dirichlet concentration (higher = closer to uniform)
        beta:   logistic-normal scale (higher = more diverse)
        L_inv:  pseudo-inverse of graph Laplacian (N, N), required for
                logistic-normal

    Returns: (N,) array on the simplex
    """
    if method == 'dirichlet':
        return rng.dirichlet(np.full(N, alpha))
    elif method == 'logistic-normal':
        z = rng.multivariate_normal(np.zeros(N), beta * L_inv)
        z -= z.max()
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()
    else:
        raise ValueError(f"Unknown init distribution: {method}")


# ── Posterior sampling ────────────────────────────────────────────────────────

def sample_posterior_cube(model, mu_obs, mask, tau_diff, edge_index, N,
                          K=20, n_steps=200, device='cpu', rng=None,
                          init_dist_fn=None):
    """Generate K posterior samples for one boundary observation."""
    if rng is None:
        rng = np.random.default_rng(42)
    if init_dist_fn is None:
        init_dist_fn = lambda rng: rng.dirichlet(np.ones(N))
    context = np.stack([mu_obs, mask, np.full(N, tau_diff)], axis=-1)
    samples = []
    for _ in range(K):
        mu_start = init_dist_fn(rng)
        _, traj = sample_trajectory_flexible(
            model, mu_start, context, edge_index,
            n_steps=n_steps, device=device)
        samples.append(traj[-1])
    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-weighting',    type=str,   default='uniform',
                        choices=['original', 'uniform', 'linear'])
    parser.add_argument('--n-epochs',          type=int,   default=1000)
    parser.add_argument('--hidden-dim',        type=int,   default=128)
    parser.add_argument('--n-layers',          type=int,   default=6)
    parser.add_argument('--lr',                type=float, default=5e-4)
    parser.add_argument('--n-train-dists',     type=int,   default=300)
    parser.add_argument('--n-starts-per-pair', type=int,   default=10)
    parser.add_argument('--n-samples',         type=int,   default=15000)
    parser.add_argument('--K',                 type=int,   default=20)
    parser.add_argument('--n-test',            type=int,   default=50)
    parser.add_argument('--init-dist',         type=str,   default='dirichlet',
                        choices=['dirichlet', 'logistic-normal'])
    parser.add_argument('--dirichlet-alpha',   type=float, default=10.0)
    parser.add_argument('--logistic-normal-beta', type=float, default=1.0)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    device = get_device()

    size = 5
    N    = size ** 3
    R    = make_cube_graph(size)
    mask = cube_boundary_mask(size)
    depth = cube_node_depth(size)
    interior_idx = np.where(mask == 0)[0]
    depths_all   = sorted(set(depth.astype(int).tolist()))
    n_boundary   = int(mask.sum())
    n_interior   = N - n_boundary

    # ── Init distribution setup ───────────────────────────────────────────────
    L_inv = None
    if args.init_dist == 'logistic-normal':
        print("Precomputing Laplacian pseudo-inverse for logistic-normal prior...")
        L_inv = precompute_laplacian_pseudoinverse(R)

    init_dist_fn = lambda rng: sample_init_distribution(
        N, rng,
        method=args.init_dist,
        alpha=args.dirichlet_alpha,
        beta=args.logistic_normal_beta,
        L_inv=L_inv,
    )

    if args.init_dist == 'dirichlet':
        init_str = f'dirichlet_a{args.dirichlet_alpha}'
    else:
        init_str = f'logistic-normal_b{args.logistic_normal_beta}'

    print(f"=== Experiment 12c: Posterior Sampling on Cube Boundary ===")
    print(f"Graph: {size}³ cube ({N} nodes, {n_boundary} boundary, "
          f"{n_interior} interior)")
    print(f"Init distribution: {init_str}")
    print(f"K={args.K} posterior samples per test case")

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        checkpoint_dir,
        f'meta_model_ex12c_{init_str}_{args.n_epochs}ep'
        f'_h{args.hidden_dim}_l{args.n_layers}.pt')

    edge_index = rate_matrix_to_edge_index(R)
    OCTANTS    = build_octant_regions(size)
    octant_names = list(OCTANTS.keys())

    model = FlexibleConditionalGNNRateMatrixPredictor(
        context_dim=3, hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    losses = None
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"\nGenerating {args.n_train_dists} training distributions "
              f"with boundary observations...")
        source_obs_pairs = []
        per_count = args.n_train_dists // 3
        for n_peaks in [1, 2, 3]:
            for _ in range(per_count):
                mu_src, _ = make_cube_multipeak(N, n_peaks, rng)
                td        = float(rng.uniform(0.5, 2.0))
                mu_obs    = compute_boundary_observation(mu_src, R, mask, td)
                source_obs_pairs.append({
                    'mu_source': mu_src,
                    'mu_obs':    mu_obs,
                    'tau_diff':  td,
                })

        n_couplings = len(source_obs_pairs) * args.n_starts_per_pair
        print(f"Building CubePosteriorDataset "
              f"({args.n_starts_per_pair} starts/pair × "
              f"{len(source_obs_pairs)} pairs = {n_couplings} OT couplings)...")
        t0 = time.time()
        dataset = CubePosteriorDataset(
            R=R, mask=mask,
            source_obs_pairs=source_obs_pairs,
            n_starts_per_pair=args.n_starts_per_pair,
            n_samples=args.n_samples, seed=42,
            init_dist_fn=init_dist_fn)
        dt = time.time() - t0
        print(f"Dataset built: {len(dataset)} samples in {dt:.1f}s")

        print(f"Training ({args.n_epochs} epochs, lr={args.lr}, "
              f"hidden={args.hidden_dim}, layers={args.n_layers})...")
        history = train_flexible_conditional(
            model, dataset,
            n_epochs=args.n_epochs, batch_size=256, lr=args.lr,
            device=device, loss_weighting=args.loss_weighting)
        losses = history['losses']
        print(f"Training: initial={losses[0]:.6f}, final={losses[-1]:.6f}")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    model.eval()

    # ── Test evaluation ───────────────────────────────────────────────────────
    rng_eval = np.random.default_rng(99)
    N_PER_PEAK = args.n_test // 3
    K = args.K

    print(f"\nRunning evaluation ({args.n_test} test cases, K={K} samples)...")
    test_results = []

    for n_peaks in [1, 2, 3]:
        for _ in range(N_PER_PEAK):
            mu_src, peak_nodes = make_cube_multipeak(N, n_peaks, rng_eval)
            td     = float(rng_eval.uniform(0.5, 2.0))
            mu_obs = compute_boundary_observation(mu_src, R, mask, td)

            # Posterior samples
            rng_k = np.random.default_rng(int(rng_eval.integers(1000000)))
            post_samples = sample_posterior_cube(
                model, mu_obs, mask, td, edge_index, N,
                K=K, n_steps=200, device=device, rng=rng_k,
                init_dist_fn=init_dist_fn)
            summ = posterior_summary(post_samples)

            # Baselines (point estimates)
            mu_lap   = baseline_laplacian(mu_obs, mask, R)
            mu_naive = baseline_naive(mu_obs, mask)

            # Peak depths
            peak_depths = [int(depth[p]) for p in peak_nodes]
            max_peak_depth = max(peak_depths)

            # Conditional probabilities (octants)
            act_probs, _ = conditional_analysis(post_samples, OCTANTS, N)
            true_octants = set(octant_of_node(p, size) for p in peak_nodes)

            # Interior node indices
            interior_idx_l = list(interior_idx)

            test_results.append({
                'mu_source':       mu_src,
                'mu_obs':          mu_obs,
                'peak_nodes':      peak_nodes,
                'peak_depths':     peak_depths,
                'max_peak_depth':  max_peak_depth,
                'n_peaks':         n_peaks,
                'tau_diff':        td,
                'post_mean':       summ['mean'],
                'post_std':        summ['std'],
                'post_q05':        summ['q05'],
                'post_q95':        summ['q95'],
                'post_samples':    post_samples,
                'mu_lap':          mu_lap,
                'mu_naive':        mu_naive,
                'act_probs':       act_probs,
                'true_octants':    true_octants,
                # TV metrics
                'tv_post_mean':    total_variation(summ['mean'], mu_src),
                'tv_lap':          total_variation(mu_lap,  mu_src),
                'tv_naive':        total_variation(mu_naive, mu_src),
                'tv_int_post_mean': total_variation(
                    summ['mean'][interior_idx], mu_src[interior_idx]),
                'tv_int_lap':       total_variation(
                    mu_lap[interior_idx],       mu_src[interior_idx]),
                # Calibration
                'node_std':        summ['std'],
                'node_err':        np.abs(summ['mean'] - mu_src),
                # Diversity
                'pairwise_tv':     pairwise_tv(post_samples),
                'depth':           depth,
            })

    # ── Console output ────────────────────────────────────────────────────────
    def mr(key, rs=None):
        rs = rs or test_results
        vals = [r[key] for r in rs]
        return np.mean(vals), np.std(vals)

    print(f"\n=== Experiment 12c: Posterior Sampling on Cube Boundary ===")
    print(f"K={K} posterior samples (init: {init_str})\n")

    print("Reconstruction (posterior mean vs baselines):")
    m, s = mr('tv_post_mean')
    print(f"  Posterior mean TV:    {m:.4f} ± {s:.4f}")
    m, s = mr('tv_lap')
    print(f"  Laplacian TV:         {m:.4f} ± {s:.4f}")

    print("\nInterior TV:")
    m, s = mr('tv_int_post_mean')
    print(f"  Posterior mean:       {m:.4f} ± {s:.4f}")
    m, s = mr('tv_int_lap')
    print(f"  Laplacian:            {m:.4f} ± {s:.4f}")

    # Calibration by depth
    all_std = np.concatenate([r['node_std'] for r in test_results])
    all_err = np.concatenate([r['node_err'] for r in test_results])
    all_dep = np.concatenate([r['depth']    for r in test_results])
    r_overall, _ = pearsonr(all_std, all_err)
    print(f"\nCalibration (correlation of std vs |error|):")
    print(f"  Overall:    r = {r_overall:.3f}")
    for d in depths_all:
        mask_d = all_dep == d
        if mask_d.sum() < 2:
            continue
        r_d, _ = pearsonr(all_std[mask_d], all_err[mask_d])
        label = 'Boundary' if d == 0 else ('Center' if d == 2 else f'Depth {d}')
        print(f"  {label:10s}: r = {r_d:.3f}")

    # Diversity by peak depth
    div_boundary = np.mean([r['pairwise_tv'] for r in test_results
                            if r['max_peak_depth'] == 0])
    div_interior = np.mean([r['pairwise_tv'] for r in test_results
                            if r['max_peak_depth'] > 0])
    print(f"\nDiversity (mean pairwise TV):")
    print(f"  Boundary peak cases:  {div_boundary:.4f}")
    print(f"  Interior peak cases:  {div_interior:.4f}")
    print(f"  (interior should be higher — more ambiguity)")

    # Conditional probability accuracy
    n_correct    = sum(max(r['act_probs'], key=r['act_probs'].get)
                       in r['true_octants'] for r in test_results)
    rs_int = [r for r in test_results if r['max_peak_depth'] > 0]
    n_correct_int = sum(max(r['act_probs'], key=r['act_probs'].get)
                        in r['true_octants'] for r in rs_int)
    n_total = len(test_results)
    n_int   = len(rs_int)
    print(f"\nConditional probability (correct octant identified):")
    print(f"  All cases:           {n_correct}/{n_total} "
          f"({n_correct/n_total*100:.0f}%)")
    if n_int > 0:
        print(f"  Interior peak cases: {n_correct_int}/{n_int} "
              f"({n_correct_int/n_int*100:.0f}%)")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f'Experiment 12c: Posterior Sampling — {size}³ Cube Boundary Observation\n'
        f'K={K} posterior samples (init: {init_str})',
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

    # ── Panel B: Posterior visualization (middle slice z=2) ───────────────────
    # Pick test case with an interior peak
    mid = size // 2
    ex_B = next((r for r in test_results if r['max_peak_depth'] > 0),
                test_results[0])

    ax_B = fig.add_subplot(gs[0, 1])
    ax_B.axis('off')
    ax_B.set_title(f'B: Posterior visualization (z={mid} slice,\n'
                   f'interior peak case)', fontsize=8, pad=4)

    inner_B = gs[0, 1].subgridspec(2, 3, hspace=0.5, wspace=0.1)
    panels_B = [
        (ex_B['mu_source'],       'True source', 'gray'),
        (ex_B['post_mean'],       'Post mean',   'steelblue'),
        (ex_B['post_std'],        'Post std',    'purple'),
        (ex_B['post_samples'][0], 'Sample 1',    'steelblue'),
        (ex_B['post_samples'][1], 'Sample 2',    'steelblue'),
        (ex_B['post_samples'][2], 'Sample 3',    'steelblue'),
    ]
    for idx_p, (mu_p, lbl, col) in enumerate(panels_B):
        row_p, col_p = divmod(idx_p, 3)
        ax_bb = fig.add_subplot(inner_B[row_p, col_p])
        sl = mu_p.reshape(size, size, size)[:, :, mid]
        vm = sl.max()
        vm = vm if vm > 1e-12 else 1.0
        ax_bb.imshow(sl, vmin=0, vmax=vm, cmap='hot', origin='lower')
        ax_bb.set_title(lbl, fontsize=6, color=col)
        ax_bb.set_xticks([])
        ax_bb.set_yticks([])

    # ── Panel C: Calibration scatter ──────────────────────────────────────────
    ax_C = fig.add_subplot(gs[0, 2])
    depth_colors = {0: 'salmon', 1: 'steelblue', 2: 'navy'}
    depth_labels_cal = {0: 'boundary', 1: 'depth-1', 2: 'depth-2 (center)'}
    for d in depths_all:
        mask_d = all_dep == d
        if mask_d.sum() == 0:
            continue
        ax_C.scatter(all_std[mask_d], all_err[mask_d],
                     alpha=0.12, s=3,
                     c=depth_colors.get(d, 'gray'),
                     label=depth_labels_cal.get(d, f'depth-{d}'))
    ax_C.set_xlabel('Posterior std per node')
    ax_C.set_ylabel('|mean − true| per node')
    ax_C.set_title(f'C: Calibration (r={r_overall:.2f})\nColored by depth')
    ax_C.legend(fontsize=7, markerscale=3)
    ax_C.grid(True, alpha=0.3)

    # ── Panel D: Reconstruction comparison by peak location ────────────────────
    ax_D = fig.add_subplot(gs[1, 0])
    rs_bnd = [r for r in test_results if r['max_peak_depth'] == 0]
    rs_int = [r for r in test_results if r['max_peak_depth'] >  0]
    groups = [('Boundary\npeaks', rs_bnd), ('Interior\npeaks', rs_int)]
    x_D = np.arange(len(groups))
    w_D = 0.28
    offsets_D = [-w_D, 0, w_D]
    for off, (key, color, meth) in zip(offsets_D, [
            ('tv_post_mean', 'steelblue',   'Post mean'),
            ('tv_lap',       'forestgreen', 'Laplacian'),
            ('tv_naive',     'salmon',      'Naive')]):
        vals = [np.mean([r[key] for r in rs]) if rs else 0
                for _, rs in groups]
        ax_D.bar(x_D + off, vals, w_D, color=color, alpha=0.85, label=meth)
    ax_D.set_xticks(x_D)
    ax_D.set_xticklabels([g[0] for g in groups])
    ax_D.set_ylabel('Mean TV')
    ax_D.set_title('D: Reconstruction Quality\nby Peak Location')
    ax_D.legend(fontsize=8)
    ax_D.grid(True, alpha=0.3, axis='y')

    # ── Panel E: Diversity vs peak depth ─────────────────────────────────────
    ax_E = fig.add_subplot(gs[1, 1])
    rng_jitter = np.random.default_rng(0)
    for d in depths_all:
        rs_d = [r for r in test_results if r['max_peak_depth'] == d]
        if not rs_d:
            continue
        divs = [r['pairwise_tv'] for r in rs_d]
        jit  = rng_jitter.uniform(-0.15, 0.15, len(divs))
        ax_E.scatter([d + j for j in jit], divs,
                     alpha=0.7, s=30,
                     c=depth_colors.get(d, 'gray'),
                     label=depth_labels_cal.get(d, f'depth-{d}'))
        ax_E.plot([d - 0.2, d + 0.2], [np.mean(divs)] * 2,
                  color='black', lw=2)
    ax_E.set_xlabel('True peak depth')
    ax_E.set_ylabel('Mean pairwise TV between samples')
    ax_E.set_title('E: Diversity vs Ambiguity\n(deeper = more ambiguous)')
    ax_E.set_xticks(depths_all)
    ax_E.set_xticklabels([f'Depth {d}' for d in depths_all])
    ax_E.legend(fontsize=7)
    ax_E.grid(True, alpha=0.3)

    # ── Panel F: Conditional probabilities (3 interior-peak cases) ────────────
    ax_F = fig.add_subplot(gs[1, 2])
    ax_F.axis('off')
    ax_F.set_title('F: P(octant active) — 3 interior-peak cases', fontsize=9, pad=4)

    rs_int_cases = [r for r in test_results if r['max_peak_depth'] > 0][:3]
    n_cases_F = len(rs_int_cases)
    if n_cases_F > 0:
        inner_F = gs[1, 2].subgridspec(n_cases_F, 1, hspace=0.9)
        for ci, r_f in enumerate(rs_int_cases):
            ax_ff = fig.add_subplot(inner_F[ci])
            true_oct = list(r_f['true_octants'])[0]
            probs = [r_f['act_probs'].get(o, 0) for o in octant_names]
            colors_F = ['tomato' if o in r_f['true_octants'] else 'steelblue'
                        for o in octant_names]
            ax_ff.bar(range(len(octant_names)), probs, color=colors_F, alpha=0.85)
            ax_ff.set_xticks(range(len(octant_names)))
            ax_ff.set_xticklabels(
                [n.replace('_', '\n') for n in octant_names], fontsize=4)
            ax_ff.set_ylabel('P(active)', fontsize=6)
            ax_ff.set_title(
                f'Case {ci+1}: depth={r_f["max_peak_depth"]} peak '
                f'(red = true octant)', fontsize=6)
            ax_ff.grid(True, alpha=0.3, axis='y')

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'ex12c_cube_posterior.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
