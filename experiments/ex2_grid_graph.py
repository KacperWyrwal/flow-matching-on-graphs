"""
Experiment 2: Flow matching on a 4x4 grid graph (N=16).

Demonstrates OT flow matching on a 2D grid.
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
    make_grid_graph,
)
from experiments.plotting import plot_distribution_grid, plot_heatmap


def make_peaked_dist(n, peak_node, eps=0.01):
    """Create a distribution peaked at peak_node with small uniform background."""
    dist = np.ones(n) * eps / (n - 1)
    dist[peak_node] = 1.0 - eps
    dist /= dist.sum()
    return dist


def main():
    # --- Setup ---
    rows, cols = 4, 4
    N = rows * cols  # 16
    R = make_grid_graph(rows, cols, weighted=False)
    graph = GraphStructure(R)

    print(f"Grid graph {rows}x{cols}, N={N}")
    print(f"Max distance: {int(np.max(graph.dist[np.isfinite(graph.dist)]))}")

    # --- Demo 1: top-left to bottom-right ---
    print("\n--- Demo 1: Node 0 (top-left) -> Node 15 (bottom-right) ---")
    mu0 = make_peaked_dist(N, 0)
    mu1 = make_peaked_dist(N, 15)

    cost = compute_cost_matrix(graph)
    coupling = compute_ot_coupling(mu0, mu1, graph_struct=graph)
    W = np.sum(coupling * cost)
    print(f"Wasserstein distance: {W:.4f}")

    def rate_fn_1(t):
        return marginal_rate_matrix(graph, coupling, t)

    times, trajectory = evolve_distribution(mu0, rate_fn_1, (0.0, 0.999), n_steps=100)

    # Snapshots for visualization
    t_points = [0.0, 0.25, 0.5, 0.75, 0.99]
    snapshots = {t: marginal_distribution(graph, coupling, t) for t in t_points}

    final_tv = total_variation(trajectory[-1], mu1)
    print(f"Final TV distance to target: {final_tv:.4f}")

    # --- Demo 2: uniform -> center ---
    print("\n--- Demo 2: Uniform -> center node (5) ---")
    center_node = 5  # row 1, col 1
    mu0_b = np.ones(N) / N
    mu1_b = make_peaked_dist(N, center_node)

    coupling_b = compute_ot_coupling(mu0_b, mu1_b, graph_struct=graph)
    W_b = np.sum(coupling_b * cost)
    print(f"Wasserstein distance (demo 2): {W_b:.4f}")

    snapshots_b = {t: marginal_distribution(graph, coupling_b, t) for t in t_points}

    def rate_fn_2(t):
        return marginal_rate_matrix(graph, coupling_b, t)

    times_b, trajectory_b = evolve_distribution(mu0_b, rate_fn_2, (0.0, 0.999), n_steps=100)
    final_tv_b = total_variation(trajectory_b[-1], mu1_b)
    print(f"Final TV distance to target (demo 2): {final_tv_b:.4f}")

    # --- Geodesic flow verification ---
    print("\n--- Geodesic flow verification ---")
    # Check that mass flows along shortest paths
    # Node 0 to node 15: geodesic goes through diagonal
    d_01_15 = graph.dist[0, 15]
    print(f"d(0, 15) = {d_01_15}")
    # Sample some geodesic nodes
    for a in range(N):
        d_0a = graph.dist[0, a]
        d_a15 = graph.dist[a, 15]
        if abs(d_0a + d_a15 - d_01_15) < 1e-9 and d_0a > 0:
            r, c = a // cols, a % cols
            print(f"  Node {a} ({r},{c}) is on geodesic from 0 to 15")

    # --- Plots ---
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('Experiment 2: OT Flow Matching on 4x4 Grid Graph', fontsize=14)

    # Panel A: Grid visualizations at 5 time snapshots (Demo 1)
    n_tp = len(t_points)
    for k, t in enumerate(t_points):
        ax = fig.add_subplot(4, n_tp, k + 1)
        plot_distribution_grid(ax, snapshots[t], rows, cols, title=f't={t}')
        if k == 0:
            ax.set_ylabel('Panel A\n(0→15)')

    # Panel A Demo 2: Grid visualizations
    for k, t in enumerate(t_points):
        ax = fig.add_subplot(4, n_tp, n_tp + k + 1)
        plot_distribution_grid(ax, snapshots_b[t], rows, cols, title=f't={t}')
        if k == 0:
            ax.set_ylabel('Panel A\n(unif→center)')

    # Panel B: Heatmap of p_t(x) over time (Demo 1)
    ax_B = fig.add_subplot(4, 1, 3)
    heatmap_data = trajectory.T  # (N, n_times)
    im = plot_heatmap(ax_B, heatmap_data, xlabel='Time step', ylabel='Node',
                      title='Panel B: p_t(x) heatmap over time (Demo 1: 0→15)')
    plt.colorbar(im, ax=ax_B, label='Prob')

    # Panel C: TV verification
    ax_C = fig.add_subplot(4, 1, 4)
    tv_to_mu1 = [total_variation(trajectory[k], mu1) for k in range(len(times))]
    tv_to_mu1_b = [total_variation(trajectory_b[k], mu1_b) for k in range(len(times_b))]
    ax_C.plot(times, tv_to_mu1, label='TV(p_t, target) - Demo1: 0→15', color='blue')
    ax_C.plot(times_b, tv_to_mu1_b, label='TV(p_t, target) - Demo2: unif→center', color='red')
    ax_C.set_xlabel('Time t')
    ax_C.set_ylabel('TV Distance')
    ax_C.set_title('Panel C: Geodesic flow verification via TV distance')
    ax_C.legend()
    ax_C.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ex2_grid_graph.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
