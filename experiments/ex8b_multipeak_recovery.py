"""
Experiment 8.2: Multi-Peak Source Recovery via Conditional Flow Matching on a 5x5 Grid.

Extends Experiment 8 to sources with 1-3 peaks. The model must reconstruct how many
peaks exist, where they are, and how mass is split among them — a harder problem than
single-peak recovery.
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.linalg import expm

from graph_ot_fm import GraphStructure, total_variation, make_grid_graph
from meta_fm import (
    ConditionalGNNRateMatrixPredictor,
    ConditionalMetaFlowMatchingDataset,
    rate_matrix_to_edge_index,
    train_conditional,
    sample_trajectory_conditional,
    get_device,
)


ROWS, COLS, N = 5, 5, 25
PEAK_MASS_FRAC = 0.80   # fraction of mass at peak nodes


def node_to_rc(node):
    return divmod(node, COLS)


def manhattan(a, b):
    ra, ca = node_to_rc(a)
    rb, cb = node_to_rc(b)
    return abs(ra - rb) + abs(ca - cb)


def make_multipeak_dist(n_peaks, rng, noise_std=0.01):
    """
    Generate a distribution with n_peaks peaks on a 5x5 grid.

    80% of mass split among peak nodes by Dirichlet(alpha=2).
    Remaining 20% uniformly across all other nodes.
    Small Gaussian noise, clip, renormalize.
    """
    peak_nodes = rng.choice(N, size=n_peaks, replace=False).tolist()
    weights = rng.dirichlet(np.full(n_peaks, 2.0))

    dist = np.full(N, 0.20 / (N - n_peaks) if N > n_peaks else 1e-6)
    for node, w in zip(peak_nodes, weights):
        dist[node] = PEAK_MASS_FRAC * w

    dist += rng.normal(0, noise_std, N)
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist, peak_nodes


def diffuse(mu, R, tau_diff):
    return mu @ expm(tau_diff * R)


def exact_inverse(mu_obs, R, tau_diff):
    result = mu_obs @ expm(-tau_diff * R)
    result = np.clip(result, 0.0, None)
    s = result.sum()
    if s > 1e-12:
        result /= s
    return result


def peak_recovery_topk(recovered, true_peaks):
    """
    Fraction of true peak nodes found in the top-k nodes by mass.
    k = len(true_peaks). Threshold-free.
    """
    k = len(true_peaks)
    top_k = set(np.argsort(recovered)[-k:].tolist())
    return len(top_k & set(true_peaks)) / k


def peak_location_topk(recovered, true_peaks):
    """
    Fraction of true peaks that have a top-k recovered node within Manhattan dist 1.
    Threshold-free version of peak_location_recovery.
    """
    k = len(true_peaks)
    top_k = np.argsort(recovered)[-k:].tolist()
    if not true_peaks:
        return 1.0
    covered = 0
    for tp in true_peaks:
        if any(manhattan(tp, rp) <= 1 for rp in top_k):
            covered += 1
    return covered / len(true_peaks)


def plot_grid_hm(ax, dist, title='', vmax=None):
    ax.imshow(dist.reshape(ROWS, COLS), cmap='YlOrRd', vmin=0,
              vmax=vmax or dist.max(), aspect='equal')
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-weighting', type=str, default='uniform',
                        choices=['original', 'uniform', 'linear'],
                        help='Loss weighting scheme (default: uniform)')
    parser.add_argument('--n-epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=4)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    device = get_device()

    R = make_grid_graph(ROWS, COLS, weighted=False)
    graph = GraphStructure(R)
    edge_index = rate_matrix_to_edge_index(R)

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        checkpoint_dir,
        f'meta_model_ex8b_multipeak_{args.loss_weighting}_{args.n_epochs}ep_h{args.hidden_dim}_l{args.n_layers}.pt')

    print(f"=== Experiment 8.2: Multi-Peak Source Recovery ({ROWS}×{COLS} grid) ===\n")

    # ── Training data: 100 per n_peaks ────────────────────────────────────────
    tau_diff_range = (0.3, 1.5)
    n_per_peaks = 100
    source_obs_pairs = []

    for n_peaks in [1, 2, 3]:
        for _ in range(n_per_peaks):
            tau_diff = float(rng.uniform(*tau_diff_range))
            mu_source, peak_nodes = make_multipeak_dist(n_peaks, rng)
            mu_obs = diffuse(mu_source, R, tau_diff)
            source_obs_pairs.append({
                'mu_source': mu_source,
                'mu_obs': mu_obs,
                'tau_diff': tau_diff,
                'n_peaks': n_peaks,
                'peak_nodes': peak_nodes,
            })

    n_train = len(source_obs_pairs)
    print(f"Generated {n_train} training pairs ({n_per_peaks} per n_peaks), "
          f"tau_diff in {tau_diff_range}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConditionalGNNRateMatrixPredictor(
        edge_index=edge_index, n_nodes=N, context_dim=2, hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"Loaded checkpoint from {ckpt_path}")
        losses = None
    else:
        print("\nBuilding ConditionalMetaFlowMatchingDataset (7500 samples)...")
        dataset = ConditionalMetaFlowMatchingDataset(
            graph=graph, source_obs_pairs=source_obs_pairs, n_samples=7500, seed=42)

        print(f"Training ConditionalGNNRateMatrixPredictor ({args.n_epochs} epochs)...")
        history = train_conditional(
            model, dataset, n_epochs=args.n_epochs, batch_size=256, lr=1e-3,
            device=device, loss_weighting=args.loss_weighting)
        losses = history['losses']
        print(f"Initial loss: {losses[0]:.6f}, Final loss: {losses[-1]:.6f}")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    model.eval()

    # ── Evaluation: 30 held-out cases (10 per n_peaks) at tau_diff=0.8 ───────
    test_tau = 0.8
    n_test_per = 10
    rng_test = np.random.default_rng(77)

    test_cases = []
    for n_peaks in [1, 2, 3]:
        for _ in range(n_test_per):
            mu_source, peak_nodes = make_multipeak_dist(n_peaks, rng_test)
            mu_obs = diffuse(mu_source, R, test_tau)
            test_cases.append({
                'mu_source': mu_source,
                'mu_obs': mu_obs,
                'tau_diff': test_tau,
                'n_peaks': n_peaks,
                'peak_nodes': peak_nodes,
            })

    results = []
    for case in test_cases:
        mu_obs = case['mu_obs']
        mu_source = case['mu_source']
        tau_diff = case['tau_diff']
        n_peaks = case['n_peaks']
        true_peaks = case['peak_nodes']

        context = np.stack([mu_obs, np.full(N, tau_diff)], axis=-1)
        _, traj = sample_trajectory_conditional(model, mu_obs, context,
                                                n_steps=200, device=device)
        mu_learned = traj[-1]
        mu_exact = exact_inverse(mu_obs, R, tau_diff)

        results.append({
            'mu_obs': mu_obs,
            'mu_source': mu_source,
            'mu_learned': mu_learned,
            'mu_exact': mu_exact,
            'n_peaks': n_peaks,
            'true_peaks': true_peaks,
            'traj': traj,
            'tv_learned': total_variation(mu_learned, mu_source),
            'tv_exact': total_variation(mu_exact, mu_source),
            'peak_topk_learned': peak_recovery_topk(mu_learned, true_peaks),
            'peak_topk_exact': peak_recovery_topk(mu_exact, true_peaks),
            'loc_topk_learned': peak_location_topk(mu_learned, true_peaks),
            'loc_topk_exact': peak_location_topk(mu_exact, true_peaks),
        })

    # ── Console output ─────────────────────────────────────────────────────────
    print(f"\n=== Experiment 8.2: Multi-Peak Recovery Results ===")
    if losses is not None:
        print(f"Training: initial loss = {losses[0]:.6f}, final loss = {losses[-1]:.6f}")

    print(f"\nBy number of peaks (tau_diff={test_tau}):")
    for n_peaks in [1, 2, 3]:
        sr = [r for r in results if r['n_peaks'] == n_peaks]
        tv_l = np.mean([r['tv_learned'] for r in sr])
        tv_e = np.mean([r['tv_exact'] for r in sr])
        pk_acc = np.mean([r['peak_topk_learned'] for r in sr]) * 100
        print(f"  n_peaks={n_peaks}: TV_learned={tv_l:.4f}, TV_exact={tv_e:.4f}, "
              f"peak_count_acc={pk_acc:.0f}%")

    print(f"\nPeak location recovery (top-k, within Manhattan dist 1):")
    for n_peaks in [1, 2, 3]:
        sr = [r for r in results if r['n_peaks'] == n_peaks]
        loc_l = np.mean([r['loc_topk_learned'] for r in sr]) * 100
        loc_e = np.mean([r['loc_topk_exact'] for r in sr]) * 100
        print(f"  n_peaks={n_peaks}: learned={loc_l:.0f}%, exact={loc_e:.0f}%")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle('Experiment 8.2: Multi-Peak Source Recovery (5×5 grid)', fontsize=13)
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.35)

    # Panel A: training loss
    ax_A = fig.add_subplot(gs[0, 0])
    if losses is not None:
        ax_A.plot(losses, color='steelblue', lw=1.5)
        ax_A.set_yscale('log')
    else:
        ax_A.text(0.5, 0.5, 'Loaded from\ncheckpoint',
                  transform=ax_A.transAxes, ha='center', va='center', fontsize=12)
    ax_A.set_xlabel('Epoch'); ax_A.set_ylabel('MSE Loss')
    ax_A.set_title('Panel A: Training Loss'); ax_A.grid(True, alpha=0.3)

    # Panel B: example recoveries — one row per n_peaks (3 rows × 4 heatmaps)
    ax_B = fig.add_subplot(gs[0, 1])
    ax_B.axis('off')
    ax_B.set_title('Panel B: Example recoveries\n(rows = n_peaks 1/2/3)', pad=12)
    inner_B = gs[0, 1].subgridspec(3, 4, hspace=0.6, wspace=0.3)

    col_titles = ['Observation', 'Recovered\n(learned)', 'Recovered\n(exact)', 'True source']
    for row, n_peaks in enumerate([1, 2, 3]):
        ex = next(r for r in results if r['n_peaks'] == n_peaks)
        vmax_b = max(ex['mu_source'].max(), ex['mu_obs'].max(),
                     ex['mu_learned'].max(), ex['mu_exact'].max())
        for col, (dist, title) in enumerate([
                (ex['mu_obs'], col_titles[0]),
                (ex['mu_learned'], col_titles[1]),
                (ex['mu_exact'], col_titles[2]),
                (ex['mu_source'], col_titles[3])]):
            ax_in = fig.add_subplot(inner_B[row, col])
            lbl = title if row == 0 else (f'{n_peaks}pk' if col == 0 else '')
            plot_grid_hm(ax_in, dist, lbl, vmax=vmax_b)

    # Panel C: TV by n_peaks (grouped bars: learned vs exact)
    ax_C = fig.add_subplot(gs[0, 2])
    peak_counts = [1, 2, 3]
    x = np.arange(len(peak_counts))
    w = 0.35
    tv_l_means = [np.mean([r['tv_learned'] for r in results if r['n_peaks'] == k])
                  for k in peak_counts]
    tv_e_means = [np.mean([r['tv_exact'] for r in results if r['n_peaks'] == k])
                  for k in peak_counts]
    ax_C.bar(x - w/2, tv_l_means, w, label='Learned', color='steelblue', alpha=0.85)
    ax_C.bar(x + w/2, tv_e_means, w, label='Exact inverse', color='salmon', alpha=0.85)
    ax_C.set_xticks(x); ax_C.set_xticklabels([f'{k} peak{"s" if k>1 else ""}' for k in peak_counts])
    ax_C.set_ylabel('Mean TV to true source')
    ax_C.set_title('Panel C: TV by number of peaks')
    ax_C.legend(fontsize=8); ax_C.grid(True, alpha=0.3, axis='y')

    # Panel D: peak recovery (top-k exact match) by n_peaks
    ax_D = fig.add_subplot(gs[1, 0])
    pk_acc_l = [np.mean([r['peak_topk_learned'] for r in results if r['n_peaks'] == k]) * 100
                for k in peak_counts]
    pk_acc_e = [np.mean([r['peak_topk_exact'] for r in results if r['n_peaks'] == k]) * 100
                for k in peak_counts]
    ax_D.bar(x - w/2, pk_acc_l, w, label='Learned', color='steelblue', alpha=0.85)
    ax_D.bar(x + w/2, pk_acc_e, w, label='Exact', color='salmon', alpha=0.85)
    ax_D.set_xticks(x); ax_D.set_xticklabels([f'{k} peak{"s" if k>1 else ""}' for k in peak_counts])
    ax_D.set_ylabel('Peak recovery (%)')
    ax_D.set_ylim(0, 115)
    ax_D.set_title('Panel D: Peak recovery (top-k exact match)')
    ax_D.legend(fontsize=8); ax_D.grid(True, alpha=0.3, axis='y')

    # Panel E: peak location recovery (top-k, Manhattan ≤ 1)
    ax_E = fig.add_subplot(gs[1, 1])
    loc_l = [np.mean([r['loc_topk_learned'] for r in results if r['n_peaks'] == k]) * 100
             for k in peak_counts]
    loc_e = [np.mean([r['loc_topk_exact'] for r in results if r['n_peaks'] == k]) * 100
             for k in peak_counts]
    ax_E.bar(x - w/2, loc_l, w, label='Learned', color='steelblue', alpha=0.85)
    ax_E.bar(x + w/2, loc_e, w, label='Exact', color='salmon', alpha=0.85)
    ax_E.set_xticks(x); ax_E.set_xticklabels([f'{k} peak{"s" if k>1 else ""}' for k in peak_counts])
    ax_E.set_ylabel('Location recovery (%)')
    ax_E.set_ylim(0, 115)
    ax_E.set_title('Panel E: Peak location recovery\n(top-k, Manhattan ≤ 1)')
    ax_E.legend(fontsize=8); ax_E.grid(True, alpha=0.3, axis='y')

    # Panel F: trajectory for first 2-peak test case
    ax_F = fig.add_subplot(gs[1, 2])
    ax_F.axis('off')
    ax_F.set_title('Panel F: Recovery trajectory (2 peaks)', pad=12)
    ex2 = next(r for r in results if r['n_peaks'] == 2)
    traj = ex2['traj']
    n_steps = len(traj)
    snap_idx = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]
    snap_labels = ['t=0', 't=0.25', 't=0.5', 't=0.75', 't=1']
    inner_F = gs[1, 2].subgridspec(1, 5, wspace=0.3)
    vmax_f = max(traj[i].max() for i in snap_idx)
    for si, (idx, lbl) in enumerate(zip(snap_idx, snap_labels)):
        ax_in = fig.add_subplot(inner_F[0, si])
        plot_grid_hm(ax_in, traj[idx], lbl, vmax=vmax_f)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'ex8b_multipeak_recovery.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
