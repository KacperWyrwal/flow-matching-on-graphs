"""
Experiment 13b: Posterior Sampling on 5×5×5 Cube.

Builds on Ex13 (sparse sensors, point estimates) by producing posterior
samples: multiple plausible source reconstructions for the same observation.
Training uses Dirichlet(1,...,1) starts rather than fixed uniform, so at
inference K different starting points yield K posterior samples.

Comparison baseline: MC dropout on DirectGNNPredictor (p=0.1).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    FiLMConditionalGNNRateMatrixPredictor,
    DirectGNNPredictor,
    PosteriorSamplingDataset,
    train_flexible_conditional,
    train_film_conditional,
    train_direct_gnn,
    sample_trajectory_flexible,
    sample_trajectory_film,
    get_device,
)
from meta_fm.model import rate_matrix_to_edge_index

# Re-use sensor utilities from ex13
from ex13_sparse_sensors import (
    place_sensors,
    compute_mixing_matrix,
    compute_sensor_observation,
    backproject,
    build_sensor_context,
    make_cube_multipeak,
    baseline_mne,
    baseline_lasso,
    tune_baselines,
)


# ── Region definitions ────────────────────────────────────────────────────────

def build_octant_regions(size=5):
    """8 octants of the size^3 cube, indexed by (x<mid, y<mid, z<mid)."""
    mid = size / 2.0
    regions = {}
    for xi, xlab in [(True, 'front'), (False, 'back')]:
        for yi, ylab in [(True, 'left'), (False, 'right')]:
            for zi, zlab in [(True, 'bottom'), (False, 'top')]:
                name = f'{xlab}_{ylab}_{zlab}'
                nodes = []
                for n in range(size ** 3):
                    x = n // (size * size)
                    y = (n // size) % size
                    z = n % size
                    if (x < mid) == xi and (y < mid) == yi and (z < mid) == zi:
                        nodes.append(n)
                regions[name] = nodes
    return regions


def build_depth_regions(depth):
    """Simple depth-based regions."""
    depths_all = sorted(set(depth.astype(int).tolist()))
    labels = {0: 'boundary', 1: 'shallow', 2: 'center'}
    return {labels.get(d, f'depth_{d}'): list(np.where(depth == d)[0])
            for d in depths_all}


# ── Posterior sampling ────────────────────────────────────────────────────────

def sample_posterior(model, y, mu_backproj, tau_diff, sensor_nodes, edge_index, N,
                     K=20, n_steps=200, device='cpu', rng=None,
                     conditioning='backproj'):
    """Generate K posterior samples for one observation."""
    if rng is None:
        rng = np.random.default_rng(42)

    if conditioning == 'film':
        node_ctx, global_ctx = build_sensor_context(y, sensor_nodes, N, tau_diff)
    else:
        context = np.stack([mu_backproj, np.full(N, tau_diff)], axis=-1)

    samples = []
    for _ in range(K):
        mu_start = rng.dirichlet(np.ones(N))
        if conditioning == 'film':
            _, traj = sample_trajectory_film(
                model, mu_start, node_ctx, global_ctx, edge_index,
                n_steps=n_steps, device=device)
        else:
            _, traj = sample_trajectory_flexible(
                model, mu_start, context, edge_index,
                n_steps=n_steps, device=device)
        samples.append(traj[-1])
    return samples


def mc_dropout_samples(model, context_np, edge_index, K=20, device='cpu'):
    """Run DirectGNNPredictor K times with dropout active."""
    ctx_t = torch.tensor(context_np, dtype=torch.float32).to(device)
    ei    = edge_index.to(device)
    model.train()  # activate dropout
    samples = []
    with torch.no_grad():
        for _ in range(K):
            mu = model(ctx_t, ei).cpu().numpy()
            samples.append(mu)
    model.eval()
    return samples


# ── Summary statistics ────────────────────────────────────────────────────────

def posterior_summary(samples):
    S = np.array(samples)  # (K, N)
    return {
        'mean': S.mean(axis=0),
        'std':  S.std(axis=0),
        'q05':  np.percentile(S, 5,  axis=0),
        'q95':  np.percentile(S, 95, axis=0),
    }


def pairwise_tv(samples):
    """Mean pairwise TV between all K*(K-1)/2 pairs."""
    K = len(samples)
    tvs = []
    for i in range(K):
        for j in range(i + 1, K):
            tvs.append(total_variation(samples[i], samples[j]))
    return float(np.mean(tvs))


def conditional_analysis(samples, regions, N):
    """Compute P(region active) and P(B active | A active)."""
    K = len(samples)
    threshold = 1.5 / N  # active if any node > 1.5x uniform

    activations = {}
    for name, nodes in regions.items():
        activations[name] = [
            any(s[n] > threshold for n in nodes)
            for s in samples
        ]

    activation_probs = {name: float(np.mean(acts))
                        for name, acts in activations.items()}

    conditional_probs = {}
    for a_name in regions:
        for b_name in regions:
            if a_name == b_name:
                continue
            a_acts = activations[a_name]
            b_acts = activations[b_name]
            n_a = sum(a_acts)
            if n_a > 0:
                n_ab = sum(a and b for a, b in zip(a_acts, b_acts))
                conditional_probs[(a_name, b_name)] = n_ab / n_a
    return activation_probs, conditional_probs


def octant_of_node(n, size=5):
    """Return the octant name for node n."""
    mid = size / 2.0
    x = n // (size * size)
    y = (n // size) % size
    z = n % size
    xlab = 'front' if x < mid else 'back'
    ylab = 'left'  if y < mid else 'right'
    zlab = 'bottom' if z < mid else 'top'
    return f'{xlab}_{ylab}_{zlab}'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-weighting',   type=str,   default='uniform',
                        choices=['original', 'uniform', 'linear'])
    parser.add_argument('--n-sensors',        type=int,   default=20)
    parser.add_argument('--sensor-sigma',     type=float, default=1.5)
    parser.add_argument('--n-epochs',         type=int,   default=1000)
    parser.add_argument('--hidden-dim',       type=int,   default=128)
    parser.add_argument('--n-layers',         type=int,   default=6)
    parser.add_argument('--lr',               type=float, default=5e-4)
    parser.add_argument('--n-train-dists',    type=int,   default=200)
    parser.add_argument('--n-starts-per-pair',type=int,   default=10)
    parser.add_argument('--n-samples',        type=int,   default=15000)
    parser.add_argument('--K',                type=int,   default=20,
                        help='Number of posterior samples per test case')
    parser.add_argument('--conditioning',     type=str,   default='film',
                        choices=['backproj', 'film'],
                        help='Conditioning method for the flow model')
    parser.add_argument('--ema-decay',        type=float, default=0.999,
                        help='EMA decay rate (0 to disable)')
    parser.add_argument('--loss-type',        type=str,   default='rate_kl',
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
    n_boundary = int((depth == 0).sum())
    n_interior = N - n_boundary
    depths_all = sorted(set(depth.astype(int).tolist()))

    print(f"=== Experiment 13b: Posterior Sampling ===")
    print(f"Graph: {size}³ cube ({N} nodes, {n_boundary} boundary, {n_interior} interior)")
    print(f"Sensors: {args.n_sensors}, K={args.K} posterior samples per test case")

    # ── Sensor placement and mixing matrix ────────────────────────────────────
    sensor_nodes = place_sensors(size, args.n_sensors, seed=42)
    A = compute_mixing_matrix(sensor_nodes, N, size, sigma=args.sensor_sigma)
    edge_index = rate_matrix_to_edge_index(R)

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_flow = os.path.join(
        checkpoint_dir,
        f'meta_model_ex13b_{args.conditioning}_{args.loss_weighting}_{args.n_epochs}ep'
        f'_h{args.hidden_dim}_l{args.n_layers}_s{args.n_sensors}.pt')
    ckpt_drop = ckpt_flow.replace(f'_ex13b_{args.conditioning}_', '_ex13b_dropout_')

    global_dim = args.n_sensors + 1  # raw sensor vector + tau_diff
    print(f"Conditioning: {args.conditioning}")

    # ── Flow model ────────────────────────────────────────────────────────────
    if args.conditioning == 'film':
        flow_model = FiLMConditionalGNNRateMatrixPredictor(
            node_context_dim=2, global_dim=global_dim,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    else:
        flow_model = FlexibleConditionalGNNRateMatrixPredictor(
            context_dim=2, hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    flow_losses = None
    if os.path.exists(ckpt_flow):
        flow_model.load_state_dict(torch.load(ckpt_flow, map_location='cpu'))
        print(f"Loaded flow checkpoint from {ckpt_flow}")
    else:
        print(f"\nGenerating {args.n_train_dists} training source distributions...")
        source_obs_pairs = []
        per_count = args.n_train_dists // 3
        for n_peaks in [1, 2, 3]:
            for _ in range(per_count):
                mu_src, _ = make_cube_multipeak(N, n_peaks, rng)
                td = float(rng.uniform(0.5, 2.0))
                y  = compute_sensor_observation(mu_src, R, A, td)
                mu_bp = backproject(y, A)
                source_obs_pairs.append({
                    'mu_source':   mu_src,
                    'mu_backproj': mu_bp,
                    'y':           y,
                    'tau_diff':    td,
                })

        print(f"Building dataset "
              f"({args.n_starts_per_pair} starts/pair × {len(source_obs_pairs)} pairs, "
              f"{args.n_samples} total samples)...")
        if args.conditioning == 'film':
            dataset = _PosteriorSamplingFiLMDataset(
                R=R, sensor_nodes=sensor_nodes,
                source_obs_pairs=source_obs_pairs,
                n_starts_per_pair=args.n_starts_per_pair,
                n_samples=args.n_samples, seed=42)
        else:
            dataset = PosteriorSamplingDataset(
                R=R, source_obs_pairs=source_obs_pairs,
                n_starts_per_pair=args.n_starts_per_pair,
                n_samples=args.n_samples, seed=42)
        print(f"Dataset built: {len(dataset)} samples")

        print(f"Training flow model ({args.n_epochs} epochs, lr={args.lr})...")
        if args.conditioning == 'film':
            history = train_film_conditional(
                flow_model, dataset,
                n_epochs=args.n_epochs, batch_size=256, lr=args.lr,
                device=device, loss_weighting=args.loss_weighting,
                loss_type=args.loss_type, ema_decay=args.ema_decay)
        else:
            history = train_flexible_conditional(
                flow_model, dataset,
                n_epochs=args.n_epochs, batch_size=256, lr=args.lr,
                device=device, loss_weighting=args.loss_weighting,
                loss_type=args.loss_type, ema_decay=args.ema_decay)
        flow_losses = history['losses']
        print(f"Flow: initial={flow_losses[0]:.6f}, final={flow_losses[-1]:.6f}")
        torch.save(flow_model.state_dict(), ckpt_flow)
        print(f"Checkpoint saved to {ckpt_flow}")

    flow_model.eval()

    # ── MC dropout model (DirectGNNPredictor with p=0.1) ─────────────────────
    dropout_model = DirectGNNPredictor(
        context_dim=2, hidden_dim=args.hidden_dim,
        n_layers=args.n_layers, dropout=0.1)

    if os.path.exists(ckpt_drop):
        dropout_model.load_state_dict(torch.load(ckpt_drop, map_location='cpu'))
        print(f"Loaded dropout checkpoint from {ckpt_drop}")
    else:
        print(f"\nTraining MC-dropout GNN ({args.n_epochs} epochs)...")
        rng_dp = np.random.default_rng(43)
        train_pairs = []
        per_count = args.n_train_dists // 3
        for n_peaks in [1, 2, 3]:
            for _ in range(per_count):
                mu_src, _ = make_cube_multipeak(N, n_peaks, rng_dp)
                td = float(rng_dp.uniform(0.5, 2.0))
                y  = compute_sensor_observation(mu_src, R, A, td)
                mu_bp = backproject(y, A)
                ctx = np.stack([mu_bp, np.full(N, td)], axis=-1)
                train_pairs.append((ctx, mu_src, edge_index))

        hist_dp = train_direct_gnn(
            dropout_model, train_pairs,
            n_epochs=args.n_epochs, lr=args.lr, device=device, seed=0,
            ema_decay=args.ema_decay)
        dp_l = hist_dp['losses']
        print(f"Dropout GNN: initial={dp_l[0]:.6f}, final={dp_l[-1]:.6f}")
        torch.save(dropout_model.state_dict(), ckpt_drop)
        print(f"Checkpoint saved to {ckpt_drop}")

    dropout_model = dropout_model.to(device)
    dropout_model.eval()

    # ── Baseline tuning ───────────────────────────────────────────────────────
    print("\nTuning LASSO/MNE baselines...")
    best_lam, best_alpha = tune_baselines(R, A, size, n_cases=20, seed=77)
    print(f"  Best MNE lambda: {best_lam},  Best LASSO alpha: {best_alpha}")

    # ── Test cases: 30 total (10 per peak count) ──────────────────────────────
    N_PER_PEAK = 10
    N_PEAKS_ALL = [1, 2, 3]
    K = args.K
    rng_eval = np.random.default_rng(99)
    TAU_TEST  = 1.0  # fixed tau for posterior tests
    OCTANTS   = build_octant_regions(size)
    octant_names = list(OCTANTS.keys())

    test_results = []
    print(f"\nRunning evaluation ({len(N_PEAKS_ALL)*N_PER_PEAK} test cases, "
          f"K={K} samples each)...")
    for n_peaks in N_PEAKS_ALL:
        for _ in range(N_PER_PEAK):
            mu_src, peak_nodes = make_cube_multipeak(N, n_peaks, rng_eval)
            td = TAU_TEST
            y  = compute_sensor_observation(mu_src, R, A, td)
            mu_bp = backproject(y, A)
            context_np = np.stack([mu_bp, np.full(N, td)], axis=-1)

            # Flow posterior samples
            rng_k = np.random.default_rng(int(rng_eval.integers(1000000)))
            flow_samples = sample_posterior(
                flow_model, y, mu_bp, td, sensor_nodes, edge_index, N,
                K=K, n_steps=200, device=device, rng=rng_k,
                conditioning=args.conditioning)
            flow_sum = posterior_summary(flow_samples)

            # MC dropout samples
            dp_samples = mc_dropout_samples(
                dropout_model, context_np, edge_index, K=K, device=device)
            dp_sum = posterior_summary(dp_samples)

            # Classical baselines (point estimates)
            mu_mne   = baseline_mne(y, A, lam=best_lam)
            mu_lasso = baseline_lasso(y, A, alpha=best_alpha)

            # Octant conditional analysis
            flow_act, flow_cond = conditional_analysis(flow_samples, OCTANTS, N)
            dp_act,   _         = conditional_analysis(dp_samples,   OCTANTS, N)

            # True peak octant(s)
            true_octants = set(octant_of_node(p, size) for p in peak_nodes)

            test_results.append({
                'mu_source':    mu_src,
                'mu_bp':        mu_bp,
                'peak_nodes':   peak_nodes,
                'n_peaks':      n_peaks,
                'flow_samples': flow_samples,
                'dp_samples':   dp_samples,
                'flow_mean':    flow_sum['mean'],
                'flow_std':     flow_sum['std'],
                'dp_mean':      dp_sum['mean'],
                'dp_std':       dp_sum['std'],
                'tv_flow_mean': total_variation(flow_sum['mean'], mu_src),
                'tv_dp_mean':   total_variation(dp_sum['mean'],  mu_src),
                'tv_mne':       total_variation(mu_mne,  mu_src),
                'tv_lasso':     total_variation(mu_lasso, mu_src),
                'tv_backproj':  total_variation(mu_bp,    mu_src),
                'pairwise_tv_flow': pairwise_tv(flow_samples),
                'pairwise_tv_dp':   pairwise_tv(dp_samples),
                'flow_act':     flow_act,
                'dp_act':       dp_act,
                'true_octants': true_octants,
                # Calibration: per-node arrays
                'flow_node_std': flow_sum['std'],
                'flow_node_err': np.abs(flow_sum['mean'] - mu_src),
                'dp_node_std':   dp_sum['std'],
                'dp_node_err':   np.abs(dp_sum['mean']  - mu_src),
                'depth':         depth,
            })

    # ── Console output ────────────────────────────────────────────────────────
    def mr(key):
        return np.mean([r[key] for r in test_results]), \
               np.std([r[key]  for r in test_results])

    print(f"\n=== Experiment 13b: Posterior Sampling ===")
    print(f"K={K} posterior samples per test case\n")

    print("Reconstruction (posterior mean):")
    for key, label in [('tv_flow_mean', 'Flow posterior mean TV'),
                        ('tv_dp_mean',  'MC dropout mean TV'),
                        ('tv_lasso',    '(Ex13 LASSO)'),
                        ('tv_backproj', '(Ex13 Backprojection)')]:
        m, s = mr(key)
        print(f"  {label:30s}: {m:.4f} ± {s:.4f}")

    # Calibration
    flow_stds = np.concatenate([r['flow_node_std'] for r in test_results])
    flow_errs = np.concatenate([r['flow_node_err'] for r in test_results])
    dp_stds   = np.concatenate([r['dp_node_std']   for r in test_results])
    dp_errs   = np.concatenate([r['dp_node_err']   for r in test_results])
    r_flow, _ = pearsonr(flow_stds, flow_errs)
    r_dp,   _ = pearsonr(dp_stds,   dp_errs)

    print(f"\nCalibration (correlation of std vs |error|):")
    print(f"  Flow model:    r = {r_flow:.3f}")
    print(f"  MC dropout:    r = {r_dp:.3f}")

    div_flow = np.mean([r['pairwise_tv_flow'] for r in test_results])
    div_dp   = np.mean([r['pairwise_tv_dp']   for r in test_results])
    print(f"\nDiversity (mean pairwise TV between samples):")
    print(f"  Flow model:    {div_flow:.4f}")
    print(f"  MC dropout:    {div_dp:.4f}")

    # Conditional accuracy: does highest-prob octant match true octant?
    flow_correct = sum(
        max(r['flow_act'], key=r['flow_act'].get) in r['true_octants']
        for r in test_results)
    dp_correct = sum(
        max(r['dp_act'], key=r['dp_act'].get) in r['true_octants']
        for r in test_results)
    n_test = len(test_results)
    print(f"\nConditional probability accuracy (highest-prob octant = true):")
    print(f"  Flow model:    {flow_correct}/{n_test} ({flow_correct/n_test*100:.0f}%)")
    print(f"  MC dropout:    {dp_correct}/{n_test} ({dp_correct/n_test*100:.0f}%)")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f'Experiment 13b: Posterior Sampling — {size}³ Cube, '
        f'{args.n_sensors} Sensors\n'
        f'K={K} posterior samples from Dirichlet prior',
        fontsize=11)
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.40)

    # ── Panel A: Training loss ─────────────────────────────────────────────────
    ax_A = fig.add_subplot(gs[0, 0])
    if flow_losses is not None:
        ax_A.plot(flow_losses, color='steelblue', lw=1.5, label='Flow')
        ax_A.set_yscale('log')
        ax_A.set_xlabel('Epoch')
        ax_A.set_ylabel('Loss')
        ax_A.legend(fontsize=8)
    else:
        ax_A.text(0.5, 0.5, 'Loaded from\ncheckpoint',
                  transform=ax_A.transAxes, ha='center', va='center', fontsize=12)
    ax_A.set_title('A: Training Loss')
    ax_A.grid(True, alpha=0.3)

    # ── Panel B: Posterior visualization (middle slice z=mid) ─────────────────
    mid = size // 2
    ex = test_results[0]  # first test case

    ax_B = fig.add_subplot(gs[0, 1])
    ax_B.axis('off')
    ax_B.set_title('B: Posterior (z=2 slice)', fontsize=9, pad=4)

    inner_B = gs[0, 1].subgridspec(2, 4, hspace=0.5, wspace=0.1)
    panels_B = [
        (ex['mu_source'],   'True source', 'gray'),
        (ex['flow_mean'],   'Flow mean',   'steelblue'),
        (ex['flow_std'],    'Flow std',    'purple'),
        (ex['flow_samples'][0], 'Sample 1', 'steelblue'),
        (ex['dp_mean'],     'Dropout mean', 'darkorange'),
        (ex['dp_std'],      'Dropout std',  'darkorange'),
        (ex['dp_samples'][0],   'Drop samp1','darkorange'),
        (ex['flow_samples'][1], 'Sample 2',  'steelblue'),
    ]
    for idx_p, (mu_p, lbl, col) in enumerate(panels_B):
        row_p, col_p = divmod(idx_p, 4)
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
    depth_flat = np.concatenate([r['depth'] for r in test_results])
    # boundary in red, interior in blue (flow)
    for d_val, d_col, d_lbl in [(0, 'salmon', 'boundary'), (1, 'steelblue', 'depth-1'),
                                  (2, 'navy', 'depth-2')]:
        mask_d = depth_flat == d_val
        if mask_d.sum() == 0:
            continue
        ax_C.scatter(flow_stds[mask_d], flow_errs[mask_d],
                     alpha=0.15, s=3, c=d_col, label=f'Flow {d_lbl}')
    ax_C.set_xlabel('Posterior std per node')
    ax_C.set_ylabel('|mean − true| per node')
    ax_C.set_title(f'C: Calibration\nFlow r={r_flow:.2f}  Dropout r={r_dp:.2f}')
    ax_C.legend(fontsize=7, markerscale=3)
    ax_C.grid(True, alpha=0.3)

    # ── Panel D: Sample diversity box plot ────────────────────────────────────
    ax_D = fig.add_subplot(gs[1, 0])
    div_flow_all = [r['pairwise_tv_flow'] for r in test_results]
    div_dp_all   = [r['pairwise_tv_dp']   for r in test_results]
    bp = ax_D.boxplot([div_flow_all, div_dp_all],
                      tick_labels=['Flow', 'MC Dropout'],
                      patch_artist=True,
                      medianprops=dict(color='black', lw=2))
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('darkorange')
    for patch in bp['boxes']:
        patch.set_alpha(0.7)
    ax_D.set_ylabel('Mean pairwise TV')
    ax_D.set_title('D: Sample Diversity\n(higher = less collapsed)')
    ax_D.grid(True, alpha=0.3, axis='y')

    # ── Panel E: Conditional probabilities (one test case) ────────────────────
    ax_E = fig.add_subplot(gs[1, 1])
    # Pick a 1-peak test case for clarity
    ex_E = next((r for r in test_results if r['n_peaks'] == 1), test_results[0])
    true_oct = list(ex_E['true_octants'])[0]
    x_E = np.arange(len(octant_names))
    flow_probs = [ex_E['flow_act'].get(o, 0) for o in octant_names]
    dp_probs   = [ex_E['dp_act'].get(o, 0)   for o in octant_names]
    w_E = 0.35
    ax_E.bar(x_E - w_E/2, flow_probs, w_E, color='steelblue',   alpha=0.85, label='Flow')
    ax_E.bar(x_E + w_E/2, dp_probs,   w_E, color='darkorange',  alpha=0.85, label='Dropout')
    # Highlight true octant
    true_idx = octant_names.index(true_oct)
    ax_E.axvline(true_idx, color='red', lw=2, linestyle='--', alpha=0.7,
                 label=f'True: {true_oct}')
    ax_E.set_xticks(x_E)
    short_names = [n.replace('_', '\n') for n in octant_names]
    ax_E.set_xticklabels(short_names, fontsize=5)
    ax_E.set_ylabel('P(octant active)')
    ax_E.set_title('E: Conditional Probabilities\n(1 test case, octant regions)')
    ax_E.legend(fontsize=7)
    ax_E.grid(True, alpha=0.3, axis='y')

    # ── Panel F: Reconstruction quality comparison ─────────────────────────────
    ax_F = fig.add_subplot(gs[1, 2])
    labels_F = ['Flow\nposterior\nmean', 'MC Dropout\nmean', 'LASSO\n(point)', 'Backproj\n(point)']
    vals_F   = [np.mean([r[k] for r in test_results])
                for k in ['tv_flow_mean', 'tv_dp_mean', 'tv_lasso', 'tv_backproj']]
    colors_F = ['steelblue', 'darkorange', 'forestgreen', 'salmon']
    bars_F = ax_F.bar(range(len(labels_F)), vals_F, color=colors_F, alpha=0.85)
    for bar, val in zip(bars_F, vals_F):
        ax_F.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                  f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax_F.set_xticks(range(len(labels_F)))
    ax_F.set_xticklabels(labels_F, fontsize=8)
    ax_F.set_ylabel('Mean Full TV')
    ax_F.set_title('F: Reconstruction Quality\n(posterior mean vs point estimates)')
    ax_F.grid(True, alpha=0.3, axis='y')

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'ex13b_posterior_sampling.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


class _PosteriorSamplingFiLMDataset(torch.utils.data.Dataset):
    """
    Like PosteriorSamplingDataset but with FiLM dual conditioning.

    source_obs_pairs must include 'y' (raw sensor readings) in addition to
    'mu_source', 'mu_backproj', and 'tau_diff'.

    Returns (mu, tau, node_context, global_cond, R_target, edge_index, N)
    compatible with train_film_conditional.
    """

    def __init__(self, R: np.ndarray, sensor_nodes,
                 source_obs_pairs,
                 n_starts_per_pair: int = 10,
                 n_samples: int = 15000, seed: int = 42):
        from graph_ot_fm import GraphStructure
        from graph_ot_fm.geodesic_cache import GeodesicCache
        from graph_ot_fm.flow import marginal_distribution_fast, marginal_rate_matrix_fast
        from graph_ot_fm.ot_solver import compute_ot_coupling

        rng = np.random.default_rng(seed)
        N = R.shape[0]
        graph_struct = GraphStructure(R)
        cache = GeodesicCache(graph_struct)
        edge_index = rate_matrix_to_edge_index(R)

        all_triples = []  # (mu_source, node_ctx, global_ctx, coupling)
        for pair in source_obs_pairs:
            mu_source = pair['mu_source']
            y         = pair['y']
            tau_diff  = pair['tau_diff']
            node_ctx, global_ctx = build_sensor_context(y, sensor_nodes, N, tau_diff)
            for _ in range(n_starts_per_pair):
                mu_start = rng.dirichlet(np.ones(N))
                pi = compute_ot_coupling(mu_start, mu_source, graph_struct=graph_struct)
                cache.precompute_for_coupling(pi)
                all_triples.append((mu_source, node_ctx, global_ctx, pi))

        self.samples = []
        for _ in range(n_samples):
            mu_source, node_ctx, global_ctx, pi = \
                all_triples[int(rng.integers(len(all_triples)))]
            tau = float(rng.uniform(0.0, 0.999))

            mu_tau      = marginal_distribution_fast(cache, pi, tau)
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
