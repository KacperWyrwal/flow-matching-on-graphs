"""
Experiment 1: Flow matching on a cycle graph (N=8).

Demonstrates OT flow matching between two distributions on a cycle.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from graph_ot_fm import (
    GraphStructure,
    compute_cost_matrix,
    compute_ot_coupling,
    marginal_distribution,
    marginal_rate_matrix,
    evolve_distribution,
    total_variation,
    make_cycle_graph,
)
from experiments.plotting import plot_distribution_bars, plot_heatmap


def main():
    # --- Setup ---
    N = 8
    R = make_cycle_graph(N, weighted=False)
    graph = GraphStructure(R)

    # Source: peaked at node 0
    mu0 = np.array([0.70, 0.10, 0.05, 0.05, 0.02, 0.02, 0.03, 0.03])
    mu0 /= mu0.sum()

    # Target: peaked at node 4
    mu1 = np.array([0.02, 0.02, 0.03, 0.03, 0.70, 0.10, 0.05, 0.05])
    mu1 /= mu1.sum()

    # --- Compute OT coupling ---
    print("Computing cost matrix...")
    cost = compute_cost_matrix(graph)
    print("Cost matrix:\n", cost)

    print("\nComputing OT coupling...")
    coupling = compute_ot_coupling(mu0, mu1, graph_struct=graph)
    print("OT coupling (nonzero entries):")
    for (i, j) in np.argwhere(coupling > 1e-10):
        print(f"  pi({i},{j}) = {coupling[i,j]:.4f}")

    # OT cost (Wasserstein distance)
    W = np.sum(coupling * cost)
    print(f"\nWasserstein distance W(mu0, mu1) = {W:.4f}")

    # --- Validation: check marginals ---
    print("\n--- Validation checks ---")
    print(f"pi row sums (should be mu0): {coupling.sum(axis=1).round(4)}")
    print(f"mu0:                         {mu0.round(4)}")
    print(f"pi col sums (should be mu1): {coupling.sum(axis=0).round(4)}")
    print(f"mu1:                         {mu1.round(4)}")

    # --- Evolve distribution ---
    t_eval = [0.0, 0.25, 0.5, 0.75, 0.99]
    snapshots = {}
    for t in t_eval:
        snapshots[t] = marginal_distribution(graph, coupling, t)

    print("\n--- Marginal distributions ---")
    for t in t_eval:
        p = snapshots[t]
        tv_to_mu0 = total_variation(p, mu0)
        tv_to_mu1 = total_variation(p, mu1)
        print(f"t={t:.2f}: sum={p.sum():.4f}, TV(p,mu0)={tv_to_mu0:.4f}, TV(p,mu1)={tv_to_mu1:.4f}")

    # --- Evolve using ODE ---
    def rate_fn(t):
        return marginal_rate_matrix(graph, coupling, t)

    times, trajectory = evolve_distribution(mu0, rate_fn, (0.0, 0.999), n_steps=100)

    # Compute TV distance along trajectory
    tv_to_mu0 = [total_variation(trajectory[k], mu0) for k in range(len(times))]
    tv_to_mu1 = [total_variation(trajectory[k], mu1) for k in range(len(times))]

    # Final TV validation
    final_tv = total_variation(trajectory[-1], mu1)
    print(f"\nFinal TV distance to target: {final_tv:.4f} (should be < 0.1)")

    # --- Plots ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Experiment 1: OT Flow Matching on Cycle Graph (N=8)', fontsize=14)

    # Panel A: bar plots at multiple time points
    ax_A = axes[0]
    time_points = t_eval
    n_tp = len(time_points)
    width = 0.15
    x = np.arange(N)
    colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, n_tp))
    for k, t in enumerate(time_points):
        ax_A.bar(x + k * width, snapshots[t], width=width, label=f't={t}', color=colors[k], alpha=0.85)
    ax_A.set_xlabel('Node')
    ax_A.set_ylabel('Probability')
    ax_A.set_title('Panel A: Distribution snapshots at various times')
    ax_A.set_xticks(x + width * (n_tp - 1) / 2)
    ax_A.set_xticklabels(range(N))
    ax_A.legend(fontsize=8)

    # Panel B: heatmap of p_t(x) over time
    ax_B = axes[1]
    heatmap_data = trajectory.T  # (N, n_times)
    im = plot_heatmap(ax_B, heatmap_data, xlabel='Time step', ylabel='Node',
                      title='Panel B: p_t(x) heatmap over time')
    plt.colorbar(im, ax=ax_B, label='Probability')
    ax_B.set_yticks(range(N))

    # Panel C: TV distance over time
    ax_C = axes[2]
    ax_C.plot(times, tv_to_mu0, label='TV(p_t, mu0)', color='blue')
    ax_C.plot(times, tv_to_mu1, label='TV(p_t, mu1)', color='red')
    ax_C.set_xlabel('Time t')
    ax_C.set_ylabel('Total Variation Distance')
    ax_C.set_title('Panel C: TV distance validation')
    ax_C.legend()
    ax_C.grid(True, alpha=0.3)
    ax_C.set_ylim(0, 1)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ex1_cycle_graph.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
