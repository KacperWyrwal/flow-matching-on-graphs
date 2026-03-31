"""
Experiment 17 Data Visualization: Graph zoo, transport examples, and size range.

Standalone script that generates 3 figures without training:
  1. ex17_graph_zoo.png      -- all training+test graphs
  2. ex17_transport_examples.png -- 6 graphs x 5 cols (OT path at t=0,0.25,0.5,0.75,1.0)
  3. ex17_size_range.png     -- smallest and largest graphs with distributions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from graph_ot_fm import (
    GraphStructure,
    GeodesicCache,
)
from graph_ot_fm.ot_solver import compute_ot_coupling
from graph_ot_fm.flow import marginal_distribution_fast

from experiments.ex17_ot_generalization import (
    generate_training_graphs,
    generate_test_graphs_topology,
    generate_test_graphs_size,
    generate_distribution_pair,
    draw_graph_with_dist,
)


def _degree_array(R):
    """Compute degree of each node from rate matrix."""
    adj = (np.abs(R) > 0).astype(float)
    np.fill_diagonal(adj, 0)
    return adj.sum(axis=1)


def plot_graph_zoo(train_graphs, topo_graphs, size_graphs, out_path):
    """Plot all graphs in a grid, color-coded by split."""
    all_entries = []
    for name, R, pos in train_graphs:
        all_entries.append((name, R, pos, 'train'))
    for name, R, pos in topo_graphs:
        all_entries.append((name, R, pos, 'topo'))
    for name, R, pos in size_graphs:
        all_entries.append((name, R, pos, 'size'))

    n_total = len(all_entries)
    ncols = 6
    nrows = (n_total + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten() if n_total > 1 else [axes]

    split_colors = {'train': 'tab:blue', 'topo': 'tab:red', 'size': 'tab:orange'}

    for idx, (name, R, pos, split) in enumerate(all_entries):
        ax = axes[idx]
        N = R.shape[0]
        deg = _degree_array(R)

        # Normalize positions
        pos_n = pos.copy()
        pos_n = pos_n - pos_n.min(0)
        rng_p = pos_n.max(0) - pos_n.min(0) + 1e-8
        pos_n = pos_n / rng_p.max()

        # Draw edges
        for i in range(N):
            for j in range(i + 1, N):
                if abs(R[i, j]) > 1e-10:
                    ax.plot([pos_n[i, 0], pos_n[j, 0]],
                            [pos_n[i, 1], pos_n[j, 1]],
                            '-', color=split_colors[split], alpha=0.2,
                            linewidth=0.5)

        # Draw nodes colored by degree
        sc = ax.scatter(pos_n[:, 0], pos_n[:, 1], c=deg, cmap='viridis',
                        s=20, zorder=5, edgecolors=split_colors[split],
                        linewidths=0.8)
        ax.set_title(f'{name} (N={N})', fontsize=7,
                     color=split_colors[split])
        ax.axis('off')
        ax.set_aspect('equal')

    # Hide unused axes
    for idx in range(n_total, len(axes)):
        axes[idx].axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='tab:blue', label='Training'),
        Patch(facecolor='tab:red', label='OOD-topo'),
        Patch(facecolor='tab:orange', label='OOD-size'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def plot_transport_examples(train_graphs, topo_graphs, size_graphs, out_path):
    """6 graphs x 5 cols showing exact OT path at t=0, 0.25, 0.5, 0.75, 1.0."""
    rng = np.random.default_rng(1234)

    # Pick 2 training, 2 OOD-topo, 2 OOD-size
    picks = []
    for graphs, label in [(train_graphs, 'train'), (topo_graphs, 'topo'),
                          (size_graphs, 'size')]:
        indices = rng.choice(len(graphs), size=min(2, len(graphs)),
                             replace=False)
        for i in indices:
            picks.append((graphs[i], label))

    t_cols = [0.0, 0.25, 0.5, 0.75, 1.0]
    n_rows = len(picks)
    n_cols = len(t_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.8))
    if n_rows == 1:
        axes = axes[None, :]

    for row, ((name, R, pos), label) in enumerate(picks):
        N = R.shape[0]
        mu_src, mu_tgt = generate_distribution_pair(N, pos, rng)

        # Compute exact OT interpolation at all t values
        try:
            graph_struct = GraphStructure(R)
            geo_cache = GeodesicCache(graph_struct)
            coupling = compute_ot_coupling(mu_src, mu_tgt, graph_struct=graph_struct)
            geo_cache.precompute_for_coupling(coupling)
            interp = {}
            for t in t_cols:
                if t <= 0.0:
                    interp[t] = mu_src.copy()
                elif t >= 0.999:
                    interp[t] = mu_tgt.copy()
                else:
                    interp[t] = marginal_distribution_fast(
                        geo_cache, coupling, min(t, 0.999))
        except Exception as e:
            print(f"  OT failed for {name}: {e}", flush=True)
            interp = {}
            for t in t_cols:
                mu_lin = (1 - t) * mu_src + t * mu_tgt
                mu_lin /= mu_lin.sum() + 1e-15
                interp[t] = mu_lin

        node_size = max(10, min(40, 1500 // max(N, 1)))
        for col_idx, t in enumerate(t_cols):
            ax = axes[row, col_idx]
            draw_graph_with_dist(ax, R, pos, mu=interp[t],
                                 title=(f't={t:.2f}' if row == 0 else ''),
                                 cmap='hot', node_size=node_size)
            if col_idx == 0:
                ax.set_ylabel(f'{name}\n(N={N}, {label})', fontsize=7)

    fig.suptitle('Exact OT Transport on Diverse Graphs', fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def plot_size_range(size_graphs, out_path):
    """Show smallest and largest test graphs with source/target distributions."""
    rng = np.random.default_rng(5678)

    # Find smallest and largest
    sizes_list = [(g[1].shape[0], g) for g in size_graphs]
    sizes_list.sort(key=lambda x: x[0])

    smallest = sizes_list[0][1]
    largest = sizes_list[-1][1]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for row, (name, R, pos) in enumerate([smallest, largest]):
        N = R.shape[0]
        mu_src, mu_tgt = generate_distribution_pair(N, pos, rng)

        for col, (mu, label) in enumerate([
            (mu_src, 'Source'),
            (mu_tgt, 'Target'),
        ]):
            ax = axes[row, col]
            node_size = max(10, min(50, 2000 // N))
            draw_graph_with_dist(ax, R, pos, mu=mu,
                                 title=f'{name} (N={N}) - {label}',
                                 cmap='hot', node_size=node_size)

    plt.suptitle('Size Range: Smallest vs Largest Test Graphs', fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=== Experiment 17: Data Visualization ===", flush=True)

    print("\nGenerating training graphs...", flush=True)
    train_graphs = generate_training_graphs(seed=42)
    print(f"  {len(train_graphs)} training graphs", flush=True)

    print("Generating test graphs (topology)...", flush=True)
    topo_graphs = generate_test_graphs_topology(seed=9000)
    print(f"  {len(topo_graphs)} OOD-topo graphs", flush=True)

    print("Generating test graphs (size)...", flush=True)
    size_graphs = generate_test_graphs_size(seed=9500)
    print(f"  {len(size_graphs)} OOD-size graphs", flush=True)

    print("\nPlotting graph zoo...", flush=True)
    plot_graph_zoo(
        train_graphs, topo_graphs, size_graphs,
        os.path.join(out_dir, 'ex17_graph_zoo.png'),
    )

    print("Plotting transport examples...", flush=True)
    plot_transport_examples(
        train_graphs, topo_graphs, size_graphs,
        os.path.join(out_dir, 'ex17_transport_examples.png'),
    )

    print("Plotting size range...", flush=True)
    plot_size_range(
        size_graphs,
        os.path.join(out_dir, 'ex17_size_range.png'),
    )

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
