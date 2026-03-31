"""
Experiment 4: Meta-Level Generalization Test.

Trains on peaked distributions at nodes 0, 1, 2 and tests on nodes 3, 4, 5
(never seen during training) to evaluate whether the model has learned the
structure of the rate matrix field or just memorized node identities.
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
    compute_ot_coupling,
    marginal_distribution,
    marginal_rate_matrix,
    evolve_distribution,
    total_variation,
    make_cycle_graph,
)
from meta_fm import MetaFlowMatchingDataset, RateMatrixPredictor, train, sample_trajectory, get_device


def entropy(p):
    p = np.asarray(p) + 1e-15
    return float(-np.sum(p * np.log(p)))


def make_peaked_dist(n, peak_node, noise_std=0.03, rng=None):
    """Place 0.6 at peak_node, distribute 0.4 uniformly among the rest, add noise."""
    if rng is None:
        rng = np.random.default_rng(42)
    dist = np.full(n, 0.4 / (n - 1))
    dist[peak_node] = 0.6
    dist += rng.normal(0, noise_std, n)
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist


def make_near_uniform(n, rng=None):
    """Near-uniform: start from 1/n, add Gaussian noise std=0.05, clip, renorm."""
    if rng is None:
        rng = np.random.default_rng(42)
    dist = np.full(n, 1.0 / n) + rng.normal(0, 0.05, n)
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-weighting', type=str, default='uniform',
                        choices=['original', 'uniform', 'linear'],
                        help='Loss weighting scheme (default: uniform)')
    parser.add_argument('--n-epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    N = 6
    R = make_cycle_graph(N, weighted=False)
    graph = GraphStructure(R)
    print(f"Cycle graph N={N}")

    # --- Generate training distributions (peaked at nodes 0, 1, 2) ---
    train_nodes = [0, 1, 2]
    n_per_node = 17  # ~50 total
    source_train = []
    for node in train_nodes:
        for _ in range(n_per_node):
            source_train.append(make_peaked_dist(N, node, rng=rng))
    # pad to exactly 50
    while len(source_train) < 50:
        source_train.append(make_peaked_dist(N, rng.choice(train_nodes), rng=rng))
    source_train = source_train[:50]

    target_train = [make_near_uniform(N, rng=rng) for _ in range(50)]

    print(f"Training: {len(source_train)} sources (nodes 0,1,2), {len(target_train)} targets")

    # --- Build training dataset ---
    print("\nCreating MetaFlowMatchingDataset (5000 samples)...")
    dataset = MetaFlowMatchingDataset(
        graph=graph,
        source_distributions=source_train,
        target_distributions=target_train,
        n_samples=5000,
        seed=42,
    )

    # --- Train model ---
    print("Training RateMatrixPredictor (500 epochs)...")
    model = RateMatrixPredictor(n_nodes=N, hidden_dim=128, n_layers=3)
    history = train(model=model, dataset=dataset, n_epochs=args.n_epochs, batch_size=256,
                    lr=1e-3, device=get_device(), loss_weighting=args.loss_weighting)
    losses = history['losses']
    print(f"Initial loss: {losses[0]:.6f}, Final loss: {losses[-1]:.6f}")

    # --- Generate test distributions ---
    # In-distribution: peaked at nodes 0, 1, 2 (5 samples)
    test_id_nodes = [0, 1, 2, 0, 1]
    test_id_sources = [make_peaked_dist(N, node, rng=rng) for node in test_id_nodes]
    test_id_targets = [make_near_uniform(N, rng=rng) for _ in range(5)]

    # Out-of-distribution: peaked at nodes 3, 4, 5 (5 samples, never seen)
    test_ood_nodes = [3, 4, 5, 3, 4]
    test_ood_sources = [make_peaked_dist(N, node, rng=rng) for node in test_ood_nodes]
    test_ood_targets = [make_near_uniform(N, rng=rng) for _ in range(5)]

    print(f"\nTest in-distribution sources at nodes: {test_id_nodes}")
    print(f"Test out-of-distribution sources at nodes: {test_ood_nodes}")

    def evaluate_test_set(sources, targets, label):
        """Run learned and exact flows, return per-sample results."""
        results = []
        for mu_s, nu_t in zip(sources, targets):
            pi = compute_ot_coupling(mu_s, nu_t, graph_struct=graph)

            def rate_fn(t, _pi=pi):
                return marginal_rate_matrix(graph, _pi, t)

            _, traj_exact = evolve_distribution(mu_s, rate_fn, (0.0, 0.999), n_steps=100)
            times_l, traj_learned = sample_trajectory(model, mu_s, n_steps=100)
            times_e = np.linspace(0.0, 0.999, 100)

            H_exact = [entropy(traj_exact[k]) for k in range(len(traj_exact))]
            H_learned = [entropy(traj_learned[k]) for k in range(len(traj_learned))]

            tv_learned = total_variation(traj_learned[-1], nu_t)
            tv_exact = total_variation(traj_exact[-1], nu_t)

            results.append({
                'times_exact': times_e,
                'times_learned': times_l,
                'H_exact': H_exact,
                'H_learned': H_learned,
                'tv_learned': tv_learned,
                'tv_exact': tv_exact,
                'traj_learned': traj_learned,
                'traj_exact': traj_exact,
            })
        return results

    print("\nEvaluating in-distribution test cases...")
    id_results = evaluate_test_set(test_id_sources, test_id_targets, "in-distribution")

    print("Evaluating out-of-distribution test cases...")
    ood_results = evaluate_test_set(test_ood_sources, test_ood_targets, "out-of-distribution")

    # --- Console output ---
    print("\n=== Validation Results ===")
    print("\nIn-distribution test cases (nodes 0,1,2):")
    for i, (res, node) in enumerate(zip(id_results, test_id_nodes)):
        print(f"  [{i+1}] peak node={node}, TV_learned={res['tv_learned']:.4f}, "
              f"TV_exact={res['tv_exact']:.4f}, "
              f"final H_learned={res['H_learned'][-1]:.4f}")

    print("\nOut-of-distribution test cases (nodes 3,4,5):")
    for i, (res, node) in enumerate(zip(ood_results, test_ood_nodes)):
        print(f"  [{i+1}] peak node={node}, TV_learned={res['tv_learned']:.4f}, "
              f"TV_exact={res['tv_exact']:.4f}, "
              f"final H_learned={res['H_learned'][-1]:.4f}")

    mean_tv_id = np.mean([r['tv_learned'] for r in id_results])
    mean_tv_ood = np.mean([r['tv_learned'] for r in ood_results])
    ratio = mean_tv_ood / mean_tv_id if mean_tv_id > 1e-10 else float('inf')
    print(f"\nMean TV in-distribution:  {mean_tv_id:.4f}")
    print(f"Mean TV out-of-distribution: {mean_tv_ood:.4f}")
    print(f"OOD/ID ratio (ideal ~1.0): {ratio:.3f}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Experiment 4: Meta-Level Generalization (Cycle N=6)', fontsize=13)

    # Panel A: Training loss
    ax_A = axes[0, 0]
    ax_A.plot(losses, color='steelblue', lw=1.5)
    ax_A.set_xlabel('Epoch')
    ax_A.set_ylabel('MSE Loss')
    ax_A.set_title('Panel A: Training Loss')
    ax_A.set_yscale('log')
    ax_A.grid(True, alpha=0.3)

    # Panel B: Entropy trajectories — in-distribution
    ax_B = axes[0, 1]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, (res, node) in enumerate(zip(id_results[:3], test_id_nodes[:3])):
        c = colors[i]
        ax_B.plot(res['times_exact'], res['H_exact'], '--', color=c, alpha=0.8,
                  label=f'Exact (node {node})')
        ax_B.plot(res['times_learned'], res['H_learned'], '-', color=c, alpha=0.8,
                  label=f'Learned (node {node})')
    ax_B.set_xlabel('Time t')
    ax_B.set_ylabel('Entropy H(p_t)')
    ax_B.set_title('Panel B: In-distribution entropy\n(solid=learned, dashed=exact)')
    ax_B.legend(fontsize=7)
    ax_B.grid(True, alpha=0.3)

    # Panel C: Entropy trajectories — out-of-distribution
    ax_C = axes[1, 0]
    for i, (res, node) in enumerate(zip(ood_results[:3], test_ood_nodes[:3])):
        c = colors[i]
        ax_C.plot(res['times_exact'], res['H_exact'], '--', color=c, alpha=0.8,
                  label=f'Exact (node {node})')
        ax_C.plot(res['times_learned'], res['H_learned'], '-', color=c, alpha=0.8,
                  label=f'Learned (node {node})')
    ax_C.set_xlabel('Time t')
    ax_C.set_ylabel('Entropy H(p_t)')
    ax_C.set_title('Panel C: OOD entropy\n(solid=learned, dashed=exact)')
    ax_C.legend(fontsize=7)
    ax_C.grid(True, alpha=0.3)

    # Panel D: TV bar chart — ID vs OOD
    ax_D = axes[1, 1]
    all_labels = [f'ID node {n}' for n in test_id_nodes] + [f'OOD node {n}' for n in test_ood_nodes]
    all_tvs = [r['tv_learned'] for r in id_results] + [r['tv_learned'] for r in ood_results]
    bar_colors = ['steelblue'] * 5 + ['salmon'] * 5
    x = np.arange(len(all_labels))
    bars = ax_D.bar(x, all_tvs, color=bar_colors)
    ax_D.set_xticks(x)
    ax_D.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
    ax_D.set_ylabel('TV distance to target at t=1')
    ax_D.set_title('Panel D: TV at t=1\n(blue=in-dist, red=OOD)')
    ax_D.axvline(x=4.5, color='gray', linestyle='--', alpha=0.5)
    ax_D.grid(True, alpha=0.3, axis='y')
    # Add legend patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='In-distribution'),
                       Patch(facecolor='salmon', label='OOD')]
    ax_D.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ex4_generalization.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
