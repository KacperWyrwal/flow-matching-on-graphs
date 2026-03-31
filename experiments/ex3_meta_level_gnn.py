"""
Experiment 3 (GNN): Meta-level flow matching on a cycle graph (N=6).

Re-runs Experiment 3 with GNNRateMatrixPredictor instead of the MLP.
Saves results as ex3_meta_level_gnn.png and checkpoint as
checkpoints/meta_model_gnn.pt.
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
from meta_fm import (
    MetaFlowMatchingDataset,
    GNNRateMatrixPredictor,
    rate_matrix_to_edge_index,
    train,
    sample_trajectory,
    get_device,
)


def make_peaked_dist(n, peak_node, eps=0.02):
    dist = np.ones(n) * eps / (n - 1)
    dist[peak_node] = 1.0 - eps
    dist /= dist.sum()
    return dist


def make_near_uniform(n, seed=0):
    rng = np.random.default_rng(seed)
    dist = np.ones(n) / n + rng.normal(0, 0.05 / n, n)
    dist = np.clip(dist, 1e-3, None)
    dist /= dist.sum()
    return dist


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

    np.random.seed(42)
    torch.manual_seed(42)

    N = 6
    R = make_cycle_graph(N, weighted=False)
    graph = GraphStructure(R)
    print(f"Cycle graph N={N}")

    n_dist = 50
    source_distributions = [make_peaked_dist(N, i % N) for i in range(n_dist)]
    target_distributions = [make_near_uniform(N, i) for i in range(n_dist)]

    print(f"Generated {n_dist} source and {n_dist} target distributions")

    print("\nCreating MetaFlowMatchingDataset (5000 samples)...")
    dataset = MetaFlowMatchingDataset(
        graph=graph,
        source_distributions=source_distributions,
        target_distributions=target_distributions,
        n_samples=5000,
        seed=42,
    )

    print("Training GNNRateMatrixPredictor (500 epochs)...")
    edge_index = rate_matrix_to_edge_index(R)
    model = GNNRateMatrixPredictor(edge_index=edge_index, n_nodes=N, hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    history = train(
        model=model,
        dataset=dataset,
        n_epochs=args.n_epochs,
        batch_size=256,
        lr=1e-3,
        device=get_device(),
        loss_weighting=args.loss_weighting,
    )
    losses = history['losses']
    print(f"\nInitial loss: {losses[0]:.6f}, Final loss: {losses[-1]:.6f}")
    print(f"Loss reduction: {losses[0]/losses[-1]:.1f}x")

    # Test on 3 held-out distributions
    test_sources = [make_peaked_dist(N, 0), make_peaked_dist(N, 2), make_peaked_dist(N, 4)]
    test_targets = [make_near_uniform(N, 10), make_near_uniform(N, 20), make_near_uniform(N, 30)]

    test_couplings = [compute_ot_coupling(mu_s, nu_t, graph_struct=graph)
                      for mu_s, nu_t in zip(test_sources, test_targets)]

    print("\n--- Testing on 3 held-out distributions ---")
    tv_results = {'learned': [], 'exact': []}

    for test_idx, (mu_s, nu_t, pi) in enumerate(zip(test_sources, test_targets, test_couplings)):
        def rate_fn(t, _pi=pi):
            return marginal_rate_matrix(graph, _pi, t)

        _, traj_exact = evolve_distribution(mu_s, rate_fn, (0.0, 0.999), n_steps=100)
        _, traj_learned = sample_trajectory(model, mu_s, n_steps=100)

        tv_learned = total_variation(traj_learned[-1], nu_t)
        tv_exact = total_variation(traj_exact[-1], nu_t)
        tv_results['learned'].append(tv_learned)
        tv_results['exact'].append(tv_exact)
        print(f"Test {test_idx+1}: TV_learned={tv_learned:.4f}, TV_exact={tv_exact:.4f}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Experiment 3 (GNN): Meta-Level Flow Matching on Cycle Graph (N=6)', fontsize=13)

    ax_A = axes[0]
    ax_A.plot(losses, color='blue', lw=1.5)
    ax_A.set_xlabel('Epoch')
    ax_A.set_ylabel('MSE Loss')
    ax_A.set_title('Panel A: Training Loss')
    ax_A.set_yscale('log')
    ax_A.grid(True, alpha=0.3)

    ax_B = axes[1]
    colors = ['blue', 'red', 'green']
    for test_idx, (mu_s, nu_t, pi) in enumerate(zip(test_sources, test_targets, test_couplings)):
        def rate_fn_b(t, _pi=pi):
            return marginal_rate_matrix(graph, _pi, t)

        times_e, traj_exact = evolve_distribution(mu_s, rate_fn_b, (0.0, 0.999), n_steps=100)
        times_l, traj_learned = sample_trajectory(model, mu_s, n_steps=100)

        def H(p):
            p = p + 1e-15
            return -np.sum(p * np.log(p))

        ax_B.plot(times_e, [H(traj_exact[k]) for k in range(len(times_e))],
                  '--', color=colors[test_idx], alpha=0.7, label=f'Exact #{test_idx+1}')
        ax_B.plot(times_l, [H(traj_learned[k]) for k in range(len(times_l))],
                  '-', color=colors[test_idx], alpha=0.7, label=f'Learned #{test_idx+1}')

    ax_B.set_xlabel('Time t')
    ax_B.set_ylabel('Entropy H(p_t)')
    ax_B.set_title('Panel B: Entropy along trajectories\n(solid=learned, dashed=exact)')
    ax_B.legend(fontsize=7, ncol=2)
    ax_B.grid(True, alpha=0.3)

    ax_C = axes[2]
    x = np.arange(3)
    width = 0.35
    ax_C.bar(x - width/2, tv_results['exact'], width, label='Exact', color='steelblue')
    ax_C.bar(x + width/2, tv_results['learned'], width, label='Learned', color='salmon')
    ax_C.set_xlabel('Test case')
    ax_C.set_ylabel('TV distance to target')
    ax_C.set_title('Panel C: TV comparison learned vs exact')
    ax_C.set_xticks(x)
    ax_C.set_xticklabels([f'Test {i+1}' for i in range(3)])
    ax_C.legend()
    ax_C.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ex3_meta_level_gnn.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()

    # Save GNN checkpoint
    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_path = os.path.join(checkpoint_dir, f'meta_model_gnn_{args.loss_weighting}_{args.n_epochs}ep_h{args.hidden_dim}_l{args.n_layers}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"GNN model saved to {model_path}")

    graph_path = os.path.join(checkpoint_dir, 'graph_rate_matrix.npy')
    np.save(graph_path, R)

    test_data_path = os.path.join(checkpoint_dir, f'ex3_test_data_gnn_{args.loss_weighting}_{args.n_epochs}ep_h{args.hidden_dim}_l{args.n_layers}.npz')
    np.savez(test_data_path,
             test_sources=np.array(test_sources),
             test_targets=np.array(test_targets))
    print(f"Test data saved to {test_data_path}")


if __name__ == '__main__':
    main()
