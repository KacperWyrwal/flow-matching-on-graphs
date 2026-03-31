"""
Experiment 5: Source Localization via Dedicated Backward Model.

Trains a separate GNNRateMatrixPredictor with source/target roles swapped
relative to Experiment 3. This learns to transport near-uniform (observed)
distributions to peaked (source) distributions using stable forward integration.

Compared to the naive approach of integrating the forward model backward in time,
which is numerically unstable near the t=1 singularity (1/(1-t) divergence), the
dedicated backward model keeps all integration in the forward direction.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch

from graph_ot_fm import (
    GraphStructure,
    compute_ot_coupling,
    marginal_distribution,
    marginal_rate_matrix,
    evolve_distribution,
    total_variation,
    kl_divergence,
    make_cycle_graph,
)
from meta_fm import (
    MetaFlowMatchingDataset,
    GNNRateMatrixPredictor,
    rate_matrix_to_edge_index,
    train,
    sample_trajectory,
    backward_trajectory,
    get_device,
)


# ── distribution generators ────────────────────────────────────────────────────

def make_peaked_dist(n, peak_node, noise_std=0.03, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    dist = np.full(n, 0.4 / (n - 1))
    dist[peak_node] = 0.6
    dist += rng.normal(0, noise_std, n)
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist


def make_near_uniform(n, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    dist = np.full(n, 1.0 / n) + rng.normal(0, 0.05, n)
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist


def add_noise(dist, sigma, rng):
    if sigma == 0.0:
        return dist.copy()
    noisy = dist + rng.normal(0, sigma, len(dist))
    noisy = np.clip(noisy, 1e-6, None)
    noisy /= noisy.sum()
    return noisy


# ── Method 1: naive backward integration of forward model ─────────────────────

def run_method1(graph, R, edge_index, N, checkpoint_dir, rng_m1, peak_nodes_test,
                loss_weighting='uniform', hidden_dim=64, n_layers=4):
    """
    Attempt source localization by integrating the forward model backward.
    Returns list of dicts with tv, peak_correct per test case.
    Falls back to quick on-the-fly training if no forward checkpoint exists.
    """
    fwd_path = os.path.join(checkpoint_dir, f'meta_model_gnn_{loss_weighting}_500ep_h{hidden_dim}_l{n_layers}.pt')
    fwd_model = GNNRateMatrixPredictor(edge_index=edge_index, n_nodes=N,
                                       hidden_dim=hidden_dim, n_layers=n_layers)

    if os.path.exists(fwd_path):
        fwd_model.load_state_dict(torch.load(fwd_path, map_location='cpu'))
    else:
        # Train quickly for comparison (200 epochs, not 500)
        source_dists = [make_peaked_dist(N, i % N, rng=rng_m1) for i in range(50)]
        target_dists = [make_near_uniform(N, rng=rng_m1) for _ in range(50)]
        dataset = MetaFlowMatchingDataset(
            graph=graph, source_distributions=source_dists,
            target_distributions=target_dists, n_samples=2000, seed=42)
        train(fwd_model, dataset, n_epochs=200, batch_size=256, lr=1e-3,
              loss_weighting=loss_weighting)

    fwd_model.eval()

    results = []
    for node in peak_nodes_test:
        mu0_true = make_peaked_dist(N, node, rng=rng_m1)
        target = make_near_uniform(N, rng=rng_m1)
        pi = compute_ot_coupling(mu0_true, target, graph_struct=graph)
        mu1_exact = marginal_distribution(graph, pi, 0.999)

        _, traj_bwd = backward_trajectory(fwd_model, mu1_exact, n_steps=200)
        mu0_recovered = traj_bwd[-1]

        results.append({
            'tv': total_variation(mu0_recovered, mu0_true),
            'peak_correct': int(np.argmax(mu0_recovered) == np.argmax(mu0_true)),
            'true_peak': node,
            'recovered_peak': int(np.argmax(mu0_recovered)),
        })
    return results


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

    N = 6
    R = make_cycle_graph(N, weighted=False)
    graph = GraphStructure(R)
    edge_index = rate_matrix_to_edge_index(R)

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Train dedicated backward model ─────────────────────────────────────────
    # Source/target are SWAPPED vs Experiment 3:
    #   Pi_0_back = near-uniform distributions (observed)
    #   Pi_1_back = peaked distributions (sources to recover)

    print("=== Experiment 5: Source Localization via Dedicated Backward Model ===")
    print(f"Graph: cycle N={N}\n")

    n_dist = 50
    back_sources = [make_near_uniform(N, rng=rng) for _ in range(n_dist)]  # near-uniform
    back_targets = [make_peaked_dist(N, i % N, rng=rng) for i in range(n_dist)]  # peaked

    print(f"Backward training set: {n_dist} near-uniform sources -> {n_dist} peaked targets")

    back_path = os.path.join(checkpoint_dir, f'meta_model_backward_gnn_{args.loss_weighting}_{args.n_epochs}ep_h{args.hidden_dim}_l{args.n_layers}.pt')
    back_model = GNNRateMatrixPredictor(edge_index=edge_index, n_nodes=N,
                                        hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    if os.path.exists(back_path):
        back_model.load_state_dict(torch.load(back_path, map_location='cpu'))
        print(f"Loaded backward model from {back_path}")
        losses = None
    else:
        print("\nCreating backward MetaFlowMatchingDataset (5000 samples)...")
        back_dataset = MetaFlowMatchingDataset(
            graph=graph,
            source_distributions=back_sources,
            target_distributions=back_targets,
            n_samples=5000,
            seed=42,
        )

        print("Training backward GNNRateMatrixPredictor (500 epochs)...")
        history = train(back_model, back_dataset,
                        n_epochs=args.n_epochs, batch_size=256, lr=1e-3,
                        device=get_device(), loss_weighting=args.loss_weighting)
        losses = history['losses']
        print(f"Initial loss: {losses[0]:.6f}, Final loss: {losses[-1]:.6f}")

        torch.save(back_model.state_dict(), back_path)
        print(f"Backward model saved to {back_path}")

    back_model.eval()

    # ── Generate 10 held-out test pairs ────────────────────────────────────────
    # Round-trip: exact forward solver produces the observation, backward model recovers source.

    test_peak_nodes = [0, 1, 2, 3, 4, 5, 0, 2, 4, 3]
    test_sources_true = [make_peaked_dist(N, node, rng=rng) for node in test_peak_nodes]

    # Exact forward flow: peaked -> near-uniform
    test_target_partners = [make_near_uniform(N, rng=rng) for _ in range(len(test_sources_true))]
    fwd_couplings = [compute_ot_coupling(mu_s, nu_t, graph_struct=graph)
                     for mu_s, nu_t in zip(test_sources_true, test_target_partners)]
    observations = [marginal_distribution(graph, pi, 0.999) for pi in fwd_couplings]

    print(f"\nGenerated {len(test_sources_true)} held-out test cases, peaks: {test_peak_nodes}")

    noise_levels = [0.0, 0.02, 0.05]
    results = {}

    for sigma in noise_levels:
        sigma_results = []
        for i, (mu_obs, mu_true) in enumerate(zip(observations, test_sources_true)):
            mu_noisy = add_noise(mu_obs, sigma, rng)

            # Forward integration of backward model: dp/dt = p @ R_backward(p, t)
            times_fwd, traj_fwd = sample_trajectory(back_model, mu_noisy, n_steps=200)
            mu_recovered = traj_fwd[-1]

            tv = total_variation(mu_recovered, mu_true)
            kl = kl_divergence(mu_true, mu_recovered)
            peak_correct = int(np.argmax(mu_recovered) == np.argmax(mu_true))

            sigma_results.append({
                'mu_obs': mu_obs,
                'mu_noisy': mu_noisy,
                'mu_true': mu_true,
                'mu_recovered': mu_recovered,
                'times_fwd': times_fwd,
                'traj_fwd': traj_fwd,
                'tv': tv,
                'kl': kl,
                'peak_correct': peak_correct,
                'true_peak': test_peak_nodes[i],
                'recovered_peak': int(np.argmax(mu_recovered)),
            })
        results[sigma] = sigma_results

    # ── Method 1: naive backward integration (for comparison) ──────────────────
    print("\nRunning Method 1 (naive backward integration) for comparison...")
    rng_m1 = np.random.default_rng(43)
    m1_results = run_method1(graph, R, edge_index, N, checkpoint_dir,
                             rng_m1, test_peak_nodes,
                             loss_weighting=args.loss_weighting,
                             hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    m1_mean_tv = np.mean([r['tv'] for r in m1_results])
    m1_peak_acc = np.mean([r['peak_correct'] for r in m1_results]) * 100

    # ── Console output ─────────────────────────────────────────────────────────
    print("\n=== Source Localization Results ===")
    print(f"\nMethod 1 (backward integration of forward model):")
    print(f"  Mean TV at sigma=0:      {m1_mean_tv:.4f}  [{'FAILED' if m1_mean_tv > 0.3 else 'OK'}]")
    print(f"  Peak recovery at sigma=0: {m1_peak_acc:.0f}%")

    print(f"\nMethod 2 (dedicated backward model — forward integration):")
    for sigma in noise_levels:
        sr = results[sigma]
        tvs = [r['tv'] for r in sr]
        peak_acc = np.mean([r['peak_correct'] for r in sr]) * 100
        print(f"\n  Noise sigma={sigma:.2f}:")
        print(f"    Mean TV:              {np.mean(tvs):.4f} ± {np.std(tvs):.4f}")
        print(f"    Peak recovery:        {peak_acc:.0f}%")
        for r in sr:
            print(f"    true={r['true_peak']}, recovered={r['recovered_peak']}, "
                  f"TV={r['tv']:.4f}, {'YES' if r['peak_correct'] else 'NO'}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Experiment 5: Source Localization via Dedicated Backward Model (Cycle N=6)',
                 fontsize=13)

    # Panel A: forward trajectory of backward model (sigma=0, one case)
    ax_A = axes[0, 0]
    example = results[0.0][0]
    traj = example['traj_fwd']
    n_steps = len(traj)
    snap_idx = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
    snap_labels = ['t=0.0', 't=0.25', 't=0.5', 't=0.75', 't=1.0']
    x_pos = np.arange(N)
    bar_w = 0.15
    cmap = plt.cm.viridis
    for pi, (idx, label) in enumerate(zip(snap_idx, snap_labels)):
        color = cmap(pi / (len(snap_idx) - 1))
        ax_A.bar(x_pos + pi * bar_w, traj[idx], bar_w, label=label, color=color, alpha=0.85)
    ax_A.set_xlabel('Node')
    ax_A.set_ylabel('Probability')
    ax_A.set_title(f'Panel A: Backward model forward trajectory (sigma=0)\n'
                   f'True peak: node {example["true_peak"]}')
    ax_A.set_xticks(x_pos + bar_w * 2)
    ax_A.set_xticklabels([str(i) for i in range(N)])
    ax_A.legend(fontsize=7, ncol=2)
    ax_A.grid(True, alpha=0.3, axis='y')

    # Panel B: true vs recovered for 3 cases at sigma=0
    ax_B = axes[0, 1]
    colors_show = ['tab:blue', 'tab:orange', 'tab:green']
    for si in range(3):
        res = results[0.0][si]
        offset = si * (N + 1)
        xs = np.arange(N) + offset
        ax_B.bar(xs - 0.175, res['mu_true'], 0.35,
                 color=colors_show[si], alpha=0.9)
        ax_B.bar(xs + 0.175, res['mu_recovered'], 0.35,
                 color=colors_show[si], alpha=0.4, hatch='//')
    tick_pos = [si * (N + 1) + (N - 1) / 2 for si in range(3)]
    ax_B.set_xticks(tick_pos)
    ax_B.set_xticklabels([f'Case {i+1}\n(peak {results[0.0][i]["true_peak"]})' for i in range(3)],
                         fontsize=8)
    ax_B.set_ylabel('Probability')
    ax_B.set_title('Panel B: True vs recovered source (sigma=0)\n(solid=true, hatched=recovered)')
    ax_B.legend(handles=[Patch(facecolor='gray', label='True'),
                          Patch(facecolor='gray', alpha=0.4, hatch='//', label='Recovered')],
                fontsize=8)
    ax_B.grid(True, alpha=0.3, axis='y')

    # Panel C: TV vs noise level
    ax_C = axes[1, 0]
    x_noise = np.arange(len(noise_levels))
    _rng_j = np.random.default_rng(99)
    for ni, sigma in enumerate(noise_levels):
        tvs = [r['tv'] for r in results[sigma]]
        jitter = _rng_j.uniform(-0.08, 0.08, len(tvs))
        ax_C.scatter(np.full(len(tvs), ni) + jitter, tvs, alpha=0.7, s=50, zorder=3)
        ax_C.plot([ni - 0.2, ni + 0.2], [np.mean(tvs)] * 2, color='black', lw=2, zorder=4)
    # Mark Method 1 baseline
    ax_C.axhline(y=m1_mean_tv, color='red', linestyle='--', alpha=0.6,
                 label=f'Method 1 (backward int.) σ=0: {m1_mean_tv:.2f}')
    ax_C.set_xticks(x_noise)
    ax_C.set_xticklabels([f'σ={s}' for s in noise_levels])
    ax_C.set_xlabel('Noise level σ')
    ax_C.set_ylabel('TV(recovered, true source)')
    ax_C.set_title('Panel C: TV distance vs noise level\n(red dashed = failed Method 1 baseline)')
    ax_C.legend(fontsize=8)
    ax_C.grid(True, alpha=0.3)

    # Panel D: peak recovery accuracy vs noise level
    ax_D = axes[1, 1]
    accuracies = [np.mean([r['peak_correct'] for r in results[sigma]]) * 100
                  for sigma in noise_levels]
    bar_colors_d = ['steelblue', 'goldenrod', 'salmon']
    bars = ax_D.bar(x_noise, accuracies, color=bar_colors_d, width=0.5)
    # Method 1 baseline
    ax_D.axhline(y=m1_peak_acc, color='red', linestyle='--', alpha=0.6,
                 label=f'Method 1 (backward int.): {m1_peak_acc:.0f}%')
    ax_D.set_xticks(x_noise)
    ax_D.set_xticklabels([f'σ={s}' for s in noise_levels])
    ax_D.set_xlabel('Noise level σ')
    ax_D.set_ylabel('Peak recovery accuracy (%)')
    ax_D.set_title('Panel D: Peak node recovery accuracy vs noise\n(red dashed = Method 1 baseline)')
    ax_D.set_ylim(0, 120)
    ax_D.axhline(y=100, color='gray', linestyle=':', alpha=0.4)
    for bar, acc in zip(bars, accuracies):
        ax_D.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                  f'{acc:.0f}%', ha='center', va='bottom', fontsize=11)
    ax_D.legend(fontsize=8)
    ax_D.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'ex5_source_localization.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
