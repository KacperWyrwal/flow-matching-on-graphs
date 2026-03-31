"""
Experiment 6: Source Localization on a 5x5 Grid Graph.

Trains a backward GNNRateMatrixPredictor to recover peaked source distributions
from heat-diffused observations. The well-conditioned inverse problem (different
source locations produce spatially distinct diffusion patterns) is the key
difference from Experiment 5.

The backward model is trained with:
  Pi_0 = diffused bumps  (observations)
  Pi_1 = peaked sources  (to recover)
And run with standard forward integration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.linalg import expm
from matplotlib.patches import Patch

from graph_ot_fm import (
    GraphStructure,
    total_variation,
    make_grid_graph,
)
from meta_fm import (
    MetaFlowMatchingDataset,
    GNNRateMatrixPredictor,
    rate_matrix_to_edge_index,
    train,
    sample_trajectory,
    get_device,
)


# ── helpers ────────────────────────────────────────────────────────────────────

ROWS, COLS = 5, 5
N = ROWS * COLS


def node_to_rc(node):
    return divmod(node, COLS)


def rc_to_node(r, c):
    return r * COLS + c


def manhattan(node_a, node_b):
    r1, c1 = node_to_rc(node_a)
    r2, c2 = node_to_rc(node_b)
    return abs(r1 - r2) + abs(c1 - c2)


def make_peaked_dist(peak_node, noise_std=0.02, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    dist = np.full(N, 0.2 / (N - 1))
    dist[peak_node] = 0.8
    dist += rng.normal(0, noise_std, N)
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist


def diffuse(mu, R, diffusion_time=2.0):
    """Apply the heat kernel: p(t) = p(0) @ expm(t * R)."""
    return mu @ expm(diffusion_time * R)


def add_noise(dist, sigma, rng):
    if sigma == 0.0:
        return dist.copy()
    noisy = dist + rng.normal(0, sigma, len(dist))
    noisy = np.clip(noisy, 1e-6, None)
    noisy /= noisy.sum()
    return noisy


def plot_grid(ax, dist, title, rows=ROWS, cols=COLS, marker_node=None,
              marker_color='red', marker_style='*', marker_label=None):
    """Heatmap of a distribution on a grid graph."""
    grid = dist.reshape(rows, cols)
    im = ax.imshow(grid, cmap='YlOrRd', vmin=0, aspect='equal')
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.tick_params(labelsize=7)
    if marker_node is not None:
        r, c = node_to_rc(marker_node)
        ax.plot(c, r, marker_style, color=marker_color, markersize=12,
                label=marker_label, markeredgecolor='black', markeredgewidth=0.5)
    return im


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-weighting', type=str, default='uniform',
                        choices=['original', 'uniform', 'linear'],
                        help='Loss weighting scheme (default: uniform)')
    parser.add_argument('--n-epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=4)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    R = make_grid_graph(ROWS, COLS, weighted=False)
    graph = GraphStructure(R)
    edge_index = rate_matrix_to_edge_index(R)

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"=== Experiment 6: Grid Source Localization ({ROWS}x{COLS}, N={N}) ===\n")

    # ── Generate 50 training source-target pairs ────────────────────────────────
    diffusion_time = 2.0
    n_train = 50
    all_nodes = list(range(N))
    train_peak_nodes = rng.choice(all_nodes, size=n_train, replace=True).tolist()

    peaked_dists = [make_peaked_dist(node, rng=rng) for node in train_peak_nodes]
    diffused_dists = [diffuse(mu, R, diffusion_time) for mu in peaked_dists]

    # Verify diffused distributions are valid
    for mu_d in diffused_dists:
        assert abs(mu_d.sum() - 1.0) < 1e-6

    print(f"Generated {n_train} training pairs (diffusion_time={diffusion_time})")
    print(f"Example: peak at node {train_peak_nodes[0]} {node_to_rc(train_peak_nodes[0])}, "
          f"diffused argmax={np.argmax(diffused_dists[0])} "
          f"{node_to_rc(np.argmax(diffused_dists[0]))}")

    # ── Train backward model: diffused -> peaked ────────────────────────────────
    # Pi_0_back = diffused bumps, Pi_1_back = peaked sources
    back_ckpt = os.path.join(checkpoint_dir, f'meta_model_grid_backward_gnn_{args.loss_weighting}_{args.n_epochs}ep_h{args.hidden_dim}_l{args.n_layers}.pt')
    model = GNNRateMatrixPredictor(edge_index=edge_index, n_nodes=N,
                                   hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    if os.path.exists(back_ckpt):
        model.load_state_dict(torch.load(back_ckpt, map_location='cpu'))
        print(f"Loaded checkpoint from {back_ckpt}")
        losses = None
    else:
        print("\nCreating MetaFlowMatchingDataset (5000 samples)...")
        # Use diagonal coupling: diffused_k <-> peaked_k (known ground-truth pairing)
        meta_coupling = np.eye(n_train) / n_train
        dataset = MetaFlowMatchingDataset(
            graph=graph,
            source_distributions=diffused_dists,   # observed (spread)
            target_distributions=peaked_dists,      # sources (to recover)
            n_samples=5000,
            meta_coupling=meta_coupling,
            seed=42,
        )

        print("Training GNNRateMatrixPredictor (500 epochs)...")
        history = train(model, dataset, n_epochs=args.n_epochs, batch_size=256, lr=1e-3,
                        device=get_device(), loss_weighting=args.loss_weighting)
        losses = history['losses']
        print(f"Initial loss: {losses[0]:.6f}, Final loss: {losses[-1]:.6f}")
        if losses[-1] >= losses[0]:
            print(f"WARNING: Loss did not decrease ({losses[0]:.6f} -> {losses[-1]:.6f}) — check training setup")

        torch.save(model.state_dict(), back_ckpt)
        print(f"Checkpoint saved to {back_ckpt}")

    model.eval()

    # ── Generate 15 held-out test cases ────────────────────────────────────────
    n_test = 15
    test_peak_nodes = rng.choice(all_nodes, size=n_test, replace=True).tolist()
    test_sources = [make_peaked_dist(node, rng=rng) for node in test_peak_nodes]
    test_observations = [diffuse(mu, R, diffusion_time) for mu in test_sources]

    print(f"\nGenerated {n_test} test cases, peaks: "
          f"{[node_to_rc(n) for n in test_peak_nodes]}")

    noise_levels = [0.0, 0.02, 0.05]
    results = {}

    for sigma in noise_levels:
        sigma_results = []
        for i, (mu_obs, mu_true) in enumerate(zip(test_observations, test_sources)):
            mu_noisy = add_noise(mu_obs, sigma, rng)

            # Forward integration of backward model
            _, traj = sample_trajectory(model, mu_noisy, n_steps=200)
            mu_recovered = traj[-1]

            true_peak = test_peak_nodes[i]
            rec_peak = int(np.argmax(mu_recovered))
            tv = total_variation(mu_recovered, mu_true)
            peak_correct = int(rec_peak == true_peak)
            loc_err = manhattan(rec_peak, true_peak)

            sigma_results.append({
                'mu_obs': mu_obs,
                'mu_noisy': mu_noisy,
                'mu_true': mu_true,
                'mu_recovered': mu_recovered,
                'traj': traj,
                'tv': tv,
                'peak_correct': peak_correct,
                'loc_err': loc_err,
                'true_peak': true_peak,
                'rec_peak': rec_peak,
            })
        results[sigma] = sigma_results

    # ── Console output ─────────────────────────────────────────────────────────
    print("\n=== Validation Results ===")
    for sigma in noise_levels:
        sr = results[sigma]
        tvs = [r['tv'] for r in sr]
        peak_acc = np.mean([r['peak_correct'] for r in sr]) * 100
        mean_loc = np.mean([r['loc_err'] for r in sr])
        print(f"\nNoise sigma={sigma:.2f}:")
        print(f"  Mean TV:                {np.mean(tvs):.4f} ± {np.std(tvs):.4f}")
        print(f"  Peak recovery:          {peak_acc:.0f}%")
        print(f"  Mean localization error: {mean_loc:.2f} cells (Manhattan)")
        for r in sr:
            rc_true = node_to_rc(r['true_peak'])
            rc_rec = node_to_rc(r['rec_peak'])
            print(f"    true={rc_true} rec={rc_rec} "
                  f"d={r['loc_err']} TV={r['tv']:.3f} "
                  f"{'OK' if r['peak_correct'] else 'MISS'}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f'Experiment 6: Grid Source Localization ({ROWS}×{COLS}, diffusion_time={diffusion_time})',
                 fontsize=13)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # Panel A: Training loss
    ax_A = fig.add_subplot(gs[0, 0])
    if losses is not None:
        ax_A.plot(losses, color='steelblue', lw=1.5)
        ax_A.set_yscale('log')
    else:
        ax_A.text(0.5, 0.5, 'Loaded from\ncheckpoint', transform=ax_A.transAxes,
                  ha='center', va='center', fontsize=12)
    ax_A.set_xlabel('Epoch')
    ax_A.set_ylabel('MSE Loss')
    ax_A.set_title('Panel A: Training Loss')
    ax_A.grid(True, alpha=0.3)

    # Panel B: Example diffusion — 2x2 mini-grid of heatmaps inside one panel
    ax_B = fig.add_subplot(gs[0, 1])
    ax_B.axis('off')
    ax_B.set_title('Panel B: Example case (sigma=0)', pad=12)

    example = results[0.0][0]
    inner_gs = gs[0, 1].subgridspec(2, 2, hspace=0.6, wspace=0.4)
    titles_b = ['True source', 'Diffused obs', 'Recovered', 'True source (ref)']
    dists_b = [example['mu_true'], example['mu_obs'],
               example['mu_recovered'], example['mu_true']]
    markers_b = [example['true_peak'], None, example['rec_peak'], example['true_peak']]
    marker_colors_b = ['lime', None, 'red', 'lime']

    for idx, (title, dist, mk, mc) in enumerate(
            zip(titles_b, dists_b, markers_b, marker_colors_b)):
        r_idx, c_idx = divmod(idx, 2)
        ax_inner = fig.add_subplot(inner_gs[r_idx, c_idx])
        plot_grid(ax_inner, dist, title, marker_node=mk,
                  marker_color=mc if mc else 'red')

    # Panel C: Spatial accuracy for 3 test cases (sigma=0)
    ax_C = fig.add_subplot(gs[0, 2])
    ax_C.axis('off')
    ax_C.set_title('Panel C: Spatial accuracy, 3 cases (sigma=0)', pad=12)

    inner_gs_c = gs[0, 2].subgridspec(1, 3, wspace=0.5)
    for si in range(3):
        res = results[0.0][si]
        ax_inner = fig.add_subplot(inner_gs_c[0, si])
        # Show recovered distribution as background heatmap
        plot_grid(ax_inner, res['mu_recovered'],
                  f'Case {si+1}\ntrue {node_to_rc(res["true_peak"])}',
                  marker_node=res['true_peak'], marker_color='lime',
                  marker_style='*', marker_label='True')
        # Also mark recovered peak
        r_rec, c_rec = node_to_rc(res['rec_peak'])
        ax_inner.plot(c_rec, r_rec, 'o', color='red', markersize=8,
                      markeredgecolor='black', markeredgewidth=0.5,
                      label='Recovered')
        if si == 0:
            ax_inner.legend(fontsize=6, loc='upper right')

    # Panel D: TV vs noise level
    ax_D = fig.add_subplot(gs[1, 0])
    x_noise = np.arange(len(noise_levels))
    _rng_j = np.random.default_rng(99)
    for ni, sigma in enumerate(noise_levels):
        tvs = [r['tv'] for r in results[sigma]]
        jitter = _rng_j.uniform(-0.1, 0.1, len(tvs))
        ax_D.scatter(np.full(len(tvs), ni) + jitter, tvs, alpha=0.6, s=40, zorder=3)
        ax_D.plot([ni - 0.2, ni + 0.2], [np.mean(tvs)] * 2,
                  color='black', lw=2.5, zorder=4)
    ax_D.set_xticks(x_noise)
    ax_D.set_xticklabels([f'σ={s}' for s in noise_levels])
    ax_D.set_xlabel('Noise level σ')
    ax_D.set_ylabel('TV(recovered, true source)')
    ax_D.set_title('Panel D: TV distance vs noise\n(dots=cases, bar=mean)')
    ax_D.grid(True, alpha=0.3)

    # Panel E: Peak recovery accuracy
    ax_E = fig.add_subplot(gs[1, 1])
    accuracies = [np.mean([r['peak_correct'] for r in results[sigma]]) * 100
                  for sigma in noise_levels]
    bar_colors = ['steelblue', 'goldenrod', 'salmon']
    bars = ax_E.bar(x_noise, accuracies, color=bar_colors, width=0.5)
    ax_E.set_xticks(x_noise)
    ax_E.set_xticklabels([f'σ={s}' for s in noise_levels])
    ax_E.set_xlabel('Noise level σ')
    ax_E.set_ylabel('Peak recovery accuracy (%)')
    ax_E.set_title('Panel E: Peak node recovery accuracy')
    ax_E.set_ylim(0, 120)
    ax_E.axhline(100, color='gray', ls=':', alpha=0.4)
    for bar, acc in zip(bars, accuracies):
        ax_E.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                  f'{acc:.0f}%', ha='center', va='bottom', fontsize=11)
    ax_E.grid(True, alpha=0.3, axis='y')

    # Panel F: Mean localization error (Manhattan)
    ax_F = fig.add_subplot(gs[1, 2])
    mean_loc_errs = [np.mean([r['loc_err'] for r in results[sigma]])
                     for sigma in noise_levels]
    bars_f = ax_F.bar(x_noise, mean_loc_errs, color=bar_colors, width=0.5)
    ax_F.set_xticks(x_noise)
    ax_F.set_xticklabels([f'σ={s}' for s in noise_levels])
    ax_F.set_xlabel('Noise level σ')
    ax_F.set_ylabel('Mean Manhattan distance (cells)')
    ax_F.set_title('Panel F: Mean localization error')
    ax_F.set_ylim(0, max(mean_loc_errs) * 1.4 + 0.5)
    for bar, err in zip(bars_f, mean_loc_errs):
        ax_F.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                  f'{err:.1f}', ha='center', va='bottom', fontsize=11)
    ax_F.grid(True, alpha=0.3, axis='y')

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'ex6_grid_source_localization.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
