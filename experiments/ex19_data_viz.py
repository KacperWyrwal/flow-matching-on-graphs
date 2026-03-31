"""
Experiment 19 Data Visualization: Asymmetric Markov chains and stationary distributions.

Standalone script that generates 3 figures without training:
  1. ex19_asymmetry_examples.png  -- 4 graphs showing asymmetry, degree-prop, true pi, difference
  2. ex19_asymmetry_spectrum.png  -- cycle_20 at 5 asymmetry levels
  3. ex19_flow_to_stationary.png  -- OT flow from uniform to pi on barbell_10
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
    make_grid_graph,
    make_cycle_graph,
    make_barabasi_albert_graph,
    make_barbell_graph,
    draw_graph_with_dist,
    compute_exact_interpolation,
)
from experiments.ex19_stationary import (
    make_asymmetric_rate_matrix,
    compute_stationary,
    baseline_degree_proportional,
)


# -- Figure 1: Asymmetry Examples ---------------------------------------------

def _draw_graph_with_arrows(ax, R, pos, title='', node_size=30):
    """Draw graph with directed arrows showing asymmetry.

    Thicker arrows in the direction of higher rate.
    """
    N = R.shape[0]
    R_off = R.copy()
    np.fill_diagonal(R_off, 0)

    # Draw edges with arrows indicating asymmetry direction
    max_rate = np.abs(R_off).max() + 1e-10
    for i in range(N):
        for j in range(i + 1, N):
            r_ij = R_off[i, j]
            r_ji = R_off[j, i]
            if r_ij < 1e-10 and r_ji < 1e-10:
                continue

            # Draw thin background line
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    'k-', alpha=0.1, linewidth=0.3)

            # Arrow from higher-rate direction
            mid_x = (pos[i, 0] + pos[j, 0]) / 2
            mid_y = (pos[i, 1] + pos[j, 1]) / 2

            if r_ij > r_ji:
                # i -> j is dominant
                strength = (r_ij - r_ji) / max_rate
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
            else:
                # j -> i is dominant
                strength = (r_ji - r_ij) / max_rate
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]

            if strength > 0.01:
                scale = 0.25
                ax.annotate('', xy=(mid_x + dx * scale, mid_y + dy * scale),
                            xytext=(mid_x - dx * scale, mid_y - dy * scale),
                            arrowprops=dict(arrowstyle='->', color='red',
                                            alpha=min(0.8, strength * 2),
                                            lw=max(0.5, strength * 3)))

    # Draw nodes
    ax.scatter(pos[:, 0], pos[:, 1], c='steelblue', s=node_size,
               zorder=5, edgecolors='k', linewidths=0.3)
    ax.set_title(title, fontsize=8)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_asymmetry_examples(out_path):
    """4 rows x 4 cols: graph arrows, degree-prop, true pi, difference."""
    rng = np.random.default_rng(42)
    asymmetry = 2.0

    # Generate 4 base graphs
    base_graphs = []
    R, pos = make_grid_graph(4, 4)
    base_graphs.append(('grid_4x4', R, pos))

    R, pos = make_cycle_graph(20)
    base_graphs.append(('cycle_20', R, pos))

    R, pos = make_barabasi_albert_graph(30, m=3, rng=rng)
    base_graphs.append(('ba_30', R, pos))

    R, pos = make_barbell_graph(10, bridge_len=3, rng=rng)
    base_graphs.append(('barbell_10', R, pos))

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))

    col_titles = ['Asymmetric Edges', 'Degree-prop Estimate',
                  'True Stationary pi', '|Degree-prop - True pi|']

    for row, (name, R_base, pos) in enumerate(base_graphs):
        R_asym, _ = make_asymmetric_rate_matrix(
            R_base, asymmetry=asymmetry, rng=rng)
        N = R_asym.shape[0]
        pi_true = compute_stationary(R_asym)
        pi_deg = baseline_degree_proportional(R_asym)
        diff = np.abs(pi_deg - pi_true)
        node_size = max(10, min(40, 1200 // max(N, 1)))

        # Col 0: Graph with asymmetry arrows
        ax = axes[row, 0]
        _draw_graph_with_arrows(ax, R_asym, pos,
                                title=(col_titles[0] if row == 0 else ''),
                                node_size=node_size)
        ax.set_ylabel(f'{name}\n(N={N})', fontsize=8)

        # Col 1: Degree-proportional estimate
        ax = axes[row, 1]
        draw_graph_with_dist(ax, R_asym, pos, mu=pi_deg,
                             title=(col_titles[1] if row == 0 else ''),
                             cmap='hot', node_size=node_size)

        # Col 2: True stationary distribution
        ax = axes[row, 2]
        draw_graph_with_dist(ax, R_asym, pos, mu=pi_true,
                             title=(col_titles[2] if row == 0 else ''),
                             cmap='hot', node_size=node_size)

        # Col 3: Difference
        ax = axes[row, 3]
        draw_graph_with_dist(ax, R_asym, pos, mu=diff,
                             title=(col_titles[3] if row == 0 else ''),
                             cmap='Reds', node_size=node_size)

    fig.suptitle(f'Asymmetric Markov Chains (asymmetry={asymmetry})',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


# -- Figure 2: Asymmetry Spectrum ---------------------------------------------

def plot_asymmetry_spectrum(out_path):
    """1 row x 5 cols: cycle_20 at asymmetry levels 0.0, 0.5, 1.0, 2.0, 3.0."""
    rng_base = np.random.default_rng(123)
    R_base, pos = make_cycle_graph(20)
    N = R_base.shape[0]

    asym_levels = [0.0, 0.5, 1.0, 2.0, 3.0]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))

    for col, asym in enumerate(asym_levels):
        ax = axes[col]
        # Use same base random seed so asymmetry weights are consistent
        rng = np.random.default_rng(123)
        if asym == 0.0:
            R_asym = R_base.copy()
        else:
            R_asym, _ = make_asymmetric_rate_matrix(
                R_base, asymmetry=asym, rng=rng)
        pi = compute_stationary(R_asym)
        draw_graph_with_dist(ax, R_asym, pos, mu=pi,
                             title=f'asymmetry={asym}',
                             cmap='hot', node_size=40)

    fig.suptitle('Stationary Distribution vs Asymmetry Level (cycle_20)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


# -- Figure 3: Flow to Stationary ---------------------------------------------

def plot_flow_to_stationary(out_path):
    """1 row x 5 cols: exact OT flow from uniform to pi on barbell_10."""
    rng = np.random.default_rng(77)
    R_base, pos = make_barbell_graph(10, bridge_len=3, rng=rng)
    rng2 = np.random.default_rng(88)
    R_asym, _ = make_asymmetric_rate_matrix(R_base, asymmetry=2.0, rng=rng2)
    N = R_asym.shape[0]
    pi = compute_stationary(R_asym)
    mu_uniform = np.ones(N, dtype=np.float32) / N

    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Compute exact OT interpolation: uniform -> pi
    interp = compute_exact_interpolation(mu_uniform, pi, R_asym, t_values)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    node_size = max(10, min(35, 800 // max(N, 1)))

    for col, t in enumerate(t_values):
        ax = axes[col]
        mu_t = interp.get(t, mu_uniform)
        draw_graph_with_dist(ax, R_asym, pos, mu=mu_t,
                             title=f't={t:.2f}',
                             cmap='hot', node_size=node_size)

    fig.suptitle('OT Flow: Uniform -> Stationary (barbell_10, asymmetry=2.0)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


# -- Main ---------------------------------------------------------------------

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=== Experiment 19: Data Visualization ===", flush=True)

    print("\nPlotting asymmetry examples...", flush=True)
    plot_asymmetry_examples(
        os.path.join(out_dir, 'ex19_asymmetry_examples.png'))

    print("Plotting asymmetry spectrum...", flush=True)
    plot_asymmetry_spectrum(
        os.path.join(out_dir, 'ex19_asymmetry_spectrum.png'))

    print("Plotting flow to stationary...", flush=True)
    plot_flow_to_stationary(
        os.path.join(out_dir, 'ex19_flow_to_stationary.png'))

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
