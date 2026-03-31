"""
Experiment 7: Discrete Generative Modeling via Flow Matching on a 3x3 Grid.

Trains a GNNRateMatrixPredictor to transport near-uniform noise to community-
structured distributions on a 3x3 grid graph. At inference, integrating the
learned flow from uniform noise generates new samples that exhibit the 4-mode
community structure of the training data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

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


# ── graph layout ───────────────────────────────────────────────────────────────
# 3x3 grid:
#  0(0,0)  1(0,1)  2(0,2)
#  3(1,0)  4(1,1)  5(1,2)
#  6(2,0)  7(2,1)  8(2,2)

ROWS, COLS, N = 3, 3, 9

COMMUNITIES = {
    'A': [0, 1, 3],   # top-left
    'B': [1, 2, 5],   # top-right
    'C': [3, 6, 7],   # bottom-left
    'D': [5, 7, 8],   # bottom-right
}


def classify_community(dist):
    """Assign a distribution to the community with highest total mass."""
    scores = {k: dist[nodes].sum() for k, nodes in COMMUNITIES.items()}
    return max(scores, key=scores.get)


def entropy(p):
    p = np.asarray(p, dtype=float) + 1e-15
    return float(-np.sum(p * np.log(p)))


# ── data generators ────────────────────────────────────────────────────────────

def make_community_sample(community_key, rng, alpha=5.0, noise=0.01):
    """Sample from community X: Dirichlet on 3 nodes + uniform noise on rest."""
    dist = np.full(N, noise)
    nodes = COMMUNITIES[community_key]
    weights = rng.dirichlet(np.full(len(nodes), alpha))
    for node, w in zip(nodes, weights):
        dist[node] += w
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist


def make_near_uniform(rng, std=0.02):
    dist = np.full(N, 1.0 / N) + rng.normal(0, std, N)
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist


def plot_grid_heatmap(ax, dist, title='', vmax=None):
    grid = dist.reshape(ROWS, COLS)
    im = ax.imshow(grid, cmap='YlOrRd', vmin=0,
                   vmax=vmax if vmax else grid.max(), aspect='equal')
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
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
    device = get_device()

    R = make_grid_graph(ROWS, COLS, weighted=False)
    graph = GraphStructure(R)
    edge_index = rate_matrix_to_edge_index(R)

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"=== Experiment 7: Discrete Generative Modeling (3×3 grid, N={N}) ===\n")

    # ── Generate training data ─────────────────────────────────────────────────
    n_per_community = 50
    community_keys = list(COMMUNITIES.keys())

    data_samples = []
    data_labels = []
    for key in community_keys:
        for _ in range(n_per_community):
            data_samples.append(make_community_sample(key, rng))
            data_labels.append(key)

    rng_shuffle = np.random.default_rng(0)
    idx = rng_shuffle.permutation(len(data_samples))
    data_samples = [data_samples[i] for i in idx]
    data_labels = [data_labels[i] for i in idx]

    n_data = len(data_samples)
    prior_samples = [make_near_uniform(rng) for _ in range(n_data)]

    print(f"Training data: {n_data} community samples ({n_per_community} per community)")
    print(f"Prior: {n_data} near-uniform samples")

    # ── Train generative model ─────────────────────────────────────────────────
    gen_ckpt = os.path.join(checkpoint_dir, f'meta_model_generative_gnn_{args.loss_weighting}_{args.n_epochs}ep_h{args.hidden_dim}_l{args.n_layers}.pt')
    model = GNNRateMatrixPredictor(edge_index=edge_index, n_nodes=N,
                                   hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    if os.path.exists(gen_ckpt):
        model.load_state_dict(torch.load(gen_ckpt, map_location='cpu'))
        print(f"Loaded checkpoint from {gen_ckpt}")
        losses = None
    else:
        print("\nCreating MetaFlowMatchingDataset (10000 samples)...")
        dataset = MetaFlowMatchingDataset(
            graph=graph,
            source_distributions=prior_samples,
            target_distributions=data_samples,
            n_samples=10000,
            seed=42,
        )

        print("Training GNNRateMatrixPredictor (500 epochs)...")
        history = train(model, dataset, n_epochs=args.n_epochs, batch_size=256,
                        lr=1e-3, device=device, loss_weighting=args.loss_weighting)
        losses = history['losses']
        print(f"Initial loss: {losses[0]:.6f}, Final loss: {losses[-1]:.6f}")

        torch.save(model.state_dict(), gen_ckpt)
        print(f"Checkpoint saved to {gen_ckpt}")

    model.eval()

    # ── Generate 100 new samples ───────────────────────────────────────────────
    n_gen = 100
    print(f"\nGenerating {n_gen} samples from learned flow...")

    generated = []
    for _ in range(n_gen):
        mu_start = make_near_uniform(rng)
        _, traj = sample_trajectory(model, mu_start, n_steps=200, device=device)
        generated.append(traj[-1])

    # ── Evaluation ────────────────────────────────────────────────────────────
    gen_labels = [classify_community(g) for g in generated]
    real_labels = data_labels

    gen_community_counts = {k: gen_labels.count(k) for k in community_keys}
    real_community_counts = {k: real_labels.count(k) for k in community_keys}

    gen_entropies = [entropy(g) for g in generated]
    real_entropies = [entropy(d) for d in data_samples]

    mode_coverage = sum(1 for k in community_keys
                        if gen_community_counts[k] / n_gen > 0.10)

    print("\n=== Experiment 7: Generative Modeling Results ===")
    if losses is not None:
        print(f"Training: initial loss = {losses[0]:.6f}, final loss = {losses[-1]:.6f}")
    print(f"Generated {n_gen} samples:")
    dist_str = ', '.join(f'{k}={gen_community_counts[k]/n_gen*100:.0f}%'
                         for k in community_keys)
    print(f"  Community distribution: {dist_str}")
    print(f"  Mode coverage: {mode_coverage}/4 communities with >10% representation")
    print(f"  Mean entropy (real):      {np.mean(real_entropies):.4f}")
    print(f"  Mean entropy (generated): {np.mean(gen_entropies):.4f}")
    ratio = np.mean(gen_entropies) / np.mean(real_entropies)
    print(f"  Entropy ratio (generated/real): {ratio:.4f} (want ~1.0)")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Experiment 7: Discrete Generative Modeling on 3×3 Grid', fontsize=13)
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35)

    # Panel A: training loss
    ax_A = fig.add_subplot(gs[0, 0])
    if losses is not None:
        ax_A.plot(losses, color='steelblue', lw=1.5)
        ax_A.set_yscale('log')
    else:
        ax_A.text(0.5, 0.5, 'Loaded from\ncheckpoint',
                  transform=ax_A.transAxes, ha='center', va='center', fontsize=12)
    ax_A.set_xlabel('Epoch')
    ax_A.set_ylabel('MSE Loss')
    ax_A.set_title('Panel A: Training Loss')
    ax_A.grid(True, alpha=0.3)

    # Panel B: 8 real data examples (2 per community), 2x4 sub-grid
    ax_B = fig.add_subplot(gs[0, 1])
    ax_B.axis('off')
    ax_B.set_title('Panel B: Real data (2 per community)', pad=12)
    inner_B = gs[0, 1].subgridspec(2, 4, hspace=0.6, wspace=0.3)
    real_examples = []
    for key in community_keys:
        idxs = [i for i, l in enumerate(data_labels) if l == key][:2]
        real_examples.extend([(data_samples[i], key) for i in idxs])
    for idx_b, (dist, label) in enumerate(real_examples):
        r, c = divmod(idx_b, 4)
        ax_inner = fig.add_subplot(inner_B[r, c])
        plot_grid_heatmap(ax_inner, dist, title=f'Com {label}')

    # Panel C: 8 generated samples (2 per community where possible), 2x4 sub-grid
    ax_C = fig.add_subplot(gs[0, 2])
    ax_C.axis('off')
    ax_C.set_title('Panel C: Generated samples', pad=12)
    inner_C = gs[0, 2].subgridspec(2, 4, hspace=0.6, wspace=0.3)
    gen_examples = []
    for key in community_keys:
        idxs = [i for i, l in enumerate(gen_labels) if l == key][:2]
        gen_examples.extend([(generated[i], key) for i in idxs])
    # pad to 8 if any community has 0 samples
    while len(gen_examples) < 8:
        gen_examples.append((generated[len(gen_examples) % n_gen], '?'))
    for idx_c, (dist, label) in enumerate(gen_examples[:8]):
        r, c = divmod(idx_c, 4)
        ax_inner = fig.add_subplot(inner_C[r, c])
        plot_grid_heatmap(ax_inner, dist, title=f'Com {label}')

    # Panel D: community distribution bar chart
    ax_D = fig.add_subplot(gs[1, 0])
    x = np.arange(len(community_keys))
    w = 0.35
    real_fracs = [real_community_counts[k] / n_data for k in community_keys]
    gen_fracs = [gen_community_counts[k] / n_gen for k in community_keys]
    ax_D.bar(x - w/2, real_fracs, w, label='Real', color='steelblue', alpha=0.85)
    ax_D.bar(x + w/2, gen_fracs, w, label='Generated', color='salmon', alpha=0.85)
    ax_D.axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Uniform (25%)')
    ax_D.set_xticks(x)
    ax_D.set_xticklabels([f'Com {k}' for k in community_keys])
    ax_D.set_ylabel('Fraction of samples')
    ax_D.set_title('Panel D: Community distribution')
    ax_D.legend(fontsize=8)
    ax_D.grid(True, alpha=0.3, axis='y')

    # Panel E: entropy histograms
    ax_E = fig.add_subplot(gs[1, 1])
    bins = np.linspace(min(min(real_entropies), min(gen_entropies)) - 0.05,
                       max(max(real_entropies), max(gen_entropies)) + 0.05, 20)
    ax_E.hist(real_entropies, bins=bins, alpha=0.6, label='Real', color='steelblue')
    ax_E.hist(gen_entropies, bins=bins, alpha=0.6, label='Generated', color='salmon')
    ax_E.set_xlabel('Entropy H(p)')
    ax_E.set_ylabel('Count')
    ax_E.set_title('Panel E: Entropy histogram\n(generated should overlap real)')
    ax_E.legend(fontsize=8)
    ax_E.grid(True, alpha=0.3)

    # Panel F: trajectory of one generated sample (5 snapshots)
    ax_F = fig.add_subplot(gs[1, 2])
    ax_F.axis('off')
    ax_F.set_title('Panel F: One generated trajectory (t=0→1)', pad=12)
    inner_F = gs[1, 2].subgridspec(1, 5, wspace=0.3)
    mu_traj_start = make_near_uniform(np.random.default_rng(7))
    _, traj_show = sample_trajectory(model, mu_traj_start, n_steps=200, device=device)
    n_traj = len(traj_show)
    snap_idx = [0, n_traj // 4, n_traj // 2, 3 * n_traj // 4, n_traj - 1]
    snap_t = ['t=0', 't=0.25', 't=0.5', 't=0.75', 't=1']
    vmax_traj = max(traj_show[i].max() for i in snap_idx)
    for si, (idx_t, tlab) in enumerate(zip(snap_idx, snap_t)):
        ax_inner = fig.add_subplot(inner_F[0, si])
        plot_grid_heatmap(ax_inner, traj_show[idx_t], title=tlab, vmax=vmax_traj)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'ex7_generative_modeling.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
