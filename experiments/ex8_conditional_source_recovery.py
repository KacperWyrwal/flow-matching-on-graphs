"""
Experiment 8: Conditional Source Recovery via Flow Matching on a 5x5 Grid.

A ConditionalGNNRateMatrixPredictor is conditioned on (mu_obs, tau_diff) and
learns to transport observed (heat-diffused) distributions back to their peaked
sources. Compared to the exact matrix-exponential inverse at well-posed
diffusion times (tau_diff in [0.3, 1.5]), both methods stay accurate, letting
the experiment show difficulty-scaling across a tractable range.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
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


def node_to_rc(node):
    return divmod(node, COLS)


def make_peaked_dist(peak_node, noise_std=0.02, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    dist = np.full(N, 0.2 / (N - 1))
    dist[peak_node] = 0.8
    dist += rng.normal(0, noise_std, N)
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist


def diffuse(mu, R, tau_diff):
    return mu @ expm(tau_diff * R)


def exact_inverse(mu_obs, R, tau_diff):
    """mu_source_exact = mu_obs @ expm(-tau_diff * R), clipped and renormed."""
    result = mu_obs @ expm(-tau_diff * R)
    result = np.clip(result, 0.0, None)
    s = result.sum()
    if s > 1e-12:
        result /= s
    return result


def plot_grid_hm(ax, dist, title='', vmax=None):
    im = ax.imshow(dist.reshape(ROWS, COLS), cmap='YlOrRd', vmin=0,
                   vmax=vmax or dist.max(), aspect='equal')
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


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
    device = get_device()

    R = make_grid_graph(ROWS, COLS, weighted=False)
    graph = GraphStructure(R)
    edge_index = rate_matrix_to_edge_index(R)

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f'meta_model_ex8_cond_gnn_{args.loss_weighting}_{args.n_epochs}ep_h{args.hidden_dim}_l{args.n_layers}.pt')

    print(f"=== Experiment 8: Conditional Source Recovery ({ROWS}×{COLS} grid) ===\n")

    # ── Training data ──────────────────────────────────────────────────────────
    n_train = 200
    tau_diff_range = (0.3, 1.5)

    source_obs_pairs = []
    for _ in range(n_train):
        peak_node = int(rng.integers(0, N))
        tau_diff = float(rng.uniform(*tau_diff_range))
        mu_source = make_peaked_dist(peak_node, rng=rng)
        mu_obs = diffuse(mu_source, R, tau_diff)
        source_obs_pairs.append({'mu_source': mu_source, 'mu_obs': mu_obs, 'tau_diff': tau_diff})

    print(f"Generated {n_train} training pairs, tau_diff in {tau_diff_range}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = ConditionalGNNRateMatrixPredictor(
        edge_index=edge_index, n_nodes=N, context_dim=2, hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"Loaded checkpoint from {ckpt_path}")
        losses = None
    else:
        print("\nBuilding ConditionalMetaFlowMatchingDataset (5000 samples)...")
        dataset = ConditionalMetaFlowMatchingDataset(
            graph=graph, source_obs_pairs=source_obs_pairs, n_samples=5000, seed=42)

        print("Training ConditionalGNNRateMatrixPredictor (500 epochs)...")
        history = train_conditional(model, dataset, n_epochs=args.n_epochs, batch_size=256,
                                    lr=1e-3, device=device,
                                    loss_weighting=args.loss_weighting)
        losses = history['losses']
        print(f"Initial loss: {losses[0]:.6f}, Final loss: {losses[-1]:.6f}")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    model.eval()

    # ── Evaluation ─────────────────────────────────────────────────────────────
    test_tau_diffs = [0.5]*5 + [0.8]*5 + [1.2]*5 + [1.4]*5
    unique_taus = [0.5, 0.8, 1.2, 1.4]
    test_cases = []

    rng_test = np.random.default_rng(99)
    for tau_diff in test_tau_diffs:
        peak_node = int(rng_test.integers(0, N))
        mu_source = make_peaked_dist(peak_node, rng=rng_test)
        mu_obs = diffuse(mu_source, R, tau_diff)
        test_cases.append({'mu_source': mu_source, 'mu_obs': mu_obs,
                           'tau_diff': tau_diff, 'peak_node': peak_node})

    results = []
    for case in test_cases:
        mu_obs, mu_source, tau_diff = case['mu_obs'], case['mu_source'], case['tau_diff']

        context = np.stack([mu_obs, np.full(N, tau_diff)], axis=-1)  # (N, 2)
        _, traj = sample_trajectory_conditional(model, mu_obs, context,
                                                n_steps=200, device=device)
        mu_learned = traj[-1]
        mu_exact = exact_inverse(mu_obs, R, tau_diff)

        has_neg_exact = bool((mu_obs @ expm(-tau_diff * R)).min() < 0)

        results.append({
            'mu_obs': mu_obs, 'mu_source': mu_source, 'mu_learned': mu_learned,
            'mu_exact': mu_exact, 'tau_diff': tau_diff,
            'traj': traj,
            'tv_learned': total_variation(mu_learned, mu_source),
            'tv_exact': total_variation(mu_exact, mu_source),
            'peak_learned': int(np.argmax(mu_learned) == case['peak_node']),
            'peak_exact': int(np.argmax(mu_exact) == case['peak_node']),
            'has_neg_exact': has_neg_exact,
        })

    # ── Console output ─────────────────────────────────────────────────────────
    print("\n=== Evaluation Results ===")
    for tau in unique_taus:
        sr = [r for r in results if r['tau_diff'] == tau]
        tv_l = np.mean([r['tv_learned'] for r in sr])
        tv_e = np.mean([r['tv_exact'] for r in sr])
        pk_l = np.mean([r['peak_learned'] for r in sr]) * 100
        pk_e = np.mean([r['peak_exact'] for r in sr]) * 100
        neg = sum(r['has_neg_exact'] for r in sr)
        print(f"  tau_diff={tau}: TV_learned={tv_l:.4f} TV_exact={tv_e:.4f} "
              f"peak_learned={pk_l:.0f}% peak_exact={pk_e:.0f}%  "
              f"exact_neg_entries={neg}/5")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Experiment 8: Conditional Source Recovery (5×5 grid, τ∈[0.3,1.5])',
                 fontsize=13)
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35)

    # Panel A: loss
    ax_A = fig.add_subplot(gs[0, 0])
    if losses is not None:
        ax_A.plot(losses, color='steelblue', lw=1.5)
        ax_A.set_yscale('log')
    else:
        ax_A.text(0.5, 0.5, 'Loaded from\ncheckpoint',
                  transform=ax_A.transAxes, ha='center', va='center', fontsize=12)
    ax_A.set_xlabel('Epoch'); ax_A.set_ylabel('MSE Loss')
    ax_A.set_title('Panel A: Training Loss'); ax_A.grid(True, alpha=0.3)

    # Panel B: example recovery (first test case at tau=0.8)
    ax_B = fig.add_subplot(gs[0, 1])
    ax_B.axis('off')
    ax_B.set_title('Panel B: Example recovery (τ=0.8)', pad=12)
    ex = next(r for r in results if r['tau_diff'] == 0.8)
    inner_B = gs[0, 1].subgridspec(1, 4, wspace=0.3)
    vmax_b = max(ex['mu_source'].max(), ex['mu_obs'].max(),
                 ex['mu_learned'].max(), ex['mu_exact'].max())
    for si, (dist, title) in enumerate([
            (ex['mu_obs'], 'Observation'),
            (ex['mu_learned'], 'Recovered\n(learned)'),
            (ex['mu_exact'], 'Recovered\n(exact)'),
            (ex['mu_source'], 'True source')]):
        ax_in = fig.add_subplot(inner_B[0, si])
        plot_grid_hm(ax_in, dist, title, vmax=vmax_b)

    # Panel C: TV grouped by tau_diff (learned vs exact)
    ax_C = fig.add_subplot(gs[0, 2])
    x = np.arange(len(unique_taus))
    w = 0.35
    tv_l_means = [np.mean([r['tv_learned'] for r in results if r['tau_diff'] == tau])
                  for tau in unique_taus]
    tv_e_means = [np.mean([r['tv_exact'] for r in results if r['tau_diff'] == tau])
                  for tau in unique_taus]
    ax_C.bar(x - w/2, tv_l_means, w, label='Learned', color='steelblue', alpha=0.85)
    ax_C.bar(x + w/2, tv_e_means, w, label='Exact inverse', color='salmon', alpha=0.85)
    ax_C.set_xticks(x); ax_C.set_xticklabels([f'τ={t}' for t in unique_taus])
    ax_C.set_ylabel('Mean TV to true source')
    ax_C.set_title('Panel C: TV by diffusion time')
    ax_C.legend(fontsize=8); ax_C.grid(True, alpha=0.3, axis='y')

    # Panel D: peak recovery accuracy
    ax_D = fig.add_subplot(gs[1, 0])
    pk_l_means = [np.mean([r['peak_learned'] for r in results if r['tau_diff'] == tau]) * 100
                  for tau in unique_taus]
    pk_e_means = [np.mean([r['peak_exact'] for r in results if r['tau_diff'] == tau]) * 100
                  for tau in unique_taus]
    ax_D.bar(x - w/2, pk_l_means, w, label='Learned', color='steelblue', alpha=0.85)
    ax_D.bar(x + w/2, pk_e_means, w, label='Exact', color='salmon', alpha=0.85)
    ax_D.set_xticks(x); ax_D.set_xticklabels([f'τ={t}' for t in unique_taus])
    ax_D.set_ylabel('Peak recovery (%)'); ax_D.set_ylim(0, 115)
    ax_D.set_title('Panel D: Peak accuracy by diffusion time')
    ax_D.legend(fontsize=8); ax_D.grid(True, alpha=0.3, axis='y')

    # Panel E: TV vs tau_diff (line plot, learned and exact)
    ax_E = fig.add_subplot(gs[1, 1])
    ax_E.plot(unique_taus, tv_l_means, 'o-', color='steelblue', lw=2, ms=8, label='Learned')
    ax_E.plot(unique_taus, tv_e_means, 's--', color='salmon', lw=2, ms=8, label='Exact inverse')
    ax_E.set_xlabel('Diffusion time τ_diff')
    ax_E.set_ylabel('Mean TV (recovered vs true)')
    ax_E.set_title('Panel E: TV vs difficulty\n(higher τ = harder recovery)')
    ax_E.legend(fontsize=8); ax_E.grid(True, alpha=0.3)

    # Panel F: trajectory for tau=0.8 test case
    ax_F = fig.add_subplot(gs[1, 2])
    ax_F.axis('off')
    ax_F.set_title('Panel F: One recovery trajectory (τ=0.8)', pad=12)
    traj = ex['traj']
    n_steps = len(traj)
    snap_idx = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]
    snap_labels = ['t=0', 't=0.25', 't=0.5', 't=0.75', 't=1']
    inner_F = gs[1, 2].subgridspec(1, 5, wspace=0.3)
    vmax_f = max(traj[i].max() for i in snap_idx)
    for si, (idx, lbl) in enumerate(zip(snap_idx, snap_labels)):
        ax_in = fig.add_subplot(inner_F[0, si])
        plot_grid_hm(ax_in, traj[idx], lbl, vmax=vmax_f)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'ex8_conditional_source_recovery.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
