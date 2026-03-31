"""
Experiment 18 Data Visualization: Diffusion source recovery difficulty and tau effects.

Standalone script that generates 2 figures without training:
  1. ex18_difficulty_spectrum.png  -- difficulty histogram + easy/medium/hard examples
  2. ex18_tau_effect.png           -- one graph, one source, observations at 7 tau values
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from graph_ot_fm import GraphStructure
from meta_fm.model import rate_matrix_to_edge_index

from experiments.ex17_ot_generalization import (
    generate_training_graphs,
    draw_graph_with_dist,
)
from experiments.ex18_source_recovery import (
    generate_source_distribution,
    generate_observation,
    compute_difficulty,
)


# -- Figure 1: Difficulty Spectrum --------------------------------------------

def plot_difficulty_spectrum(out_path):
    """Top row: histogram of difficulty values from training data.
    Bottom 3 rows x 3 cols: easy/medium/hard examples, each showing
    (source, observation) on graph."""
    rng = np.random.default_rng(42)

    print("  Generating training graphs...", flush=True)
    train_graphs = generate_training_graphs(seed=42)

    # Sample 200 pairs
    n_pairs = 200
    pairs = []
    tau_range = (0.3, 1.5)
    for _ in range(n_pairs):
        g_idx = int(rng.integers(len(train_graphs)))
        name, R, pos = train_graphs[g_idx]
        N = R.shape[0]
        mu_source = generate_source_distribution(N, pos, rng, R=R)
        tau = float(rng.uniform(tau_range[0], tau_range[1]))
        obs = generate_observation(mu_source, R, tau)
        difficulty = compute_difficulty(mu_source, obs)
        pairs.append({
            'name': name, 'R': R, 'pos': pos, 'N': N,
            'mu_source': mu_source, 'obs': obs, 'tau': tau,
            'difficulty': difficulty,
        })

    difficulties = np.array([p['difficulty'] for p in pairs])

    # Sort by difficulty
    sorted_pairs = sorted(pairs, key=lambda p: p['difficulty'])

    # Pick easy (bottom 10%), medium (around 50%), hard (top 10%)
    n = len(sorted_pairs)
    easy_indices = [int(n * 0.05), int(n * 0.08), int(n * 0.12)]
    med_indices = [int(n * 0.45), int(n * 0.50), int(n * 0.55)]
    hard_indices = [int(n * 0.85), int(n * 0.90), int(n * 0.95)]

    example_rows = [
        ('Easy', easy_indices),
        ('Medium', med_indices),
        ('Hard', hard_indices),
    ]

    fig = plt.figure(figsize=(14, 16))

    # Top row: histogram spanning full width
    ax_hist = fig.add_axes([0.08, 0.78, 0.85, 0.18])
    ax_hist.hist(difficulties, bins=30, color='steelblue', edgecolor='white',
                 alpha=0.8)
    ax_hist.set_xlabel('Difficulty (TV: source vs observation)', fontsize=10)
    ax_hist.set_ylabel('Count', fontsize=10)
    ax_hist.set_title('Difficulty Distribution (200 training pairs)', fontsize=12)
    ax_hist.grid(True, alpha=0.3, axis='y')

    # Mark easy/medium/hard regions
    for label, indices, color in [('Easy', easy_indices, 'green'),
                                   ('Medium', med_indices, 'orange'),
                                   ('Hard', hard_indices, 'red')]:
        d_vals = [sorted_pairs[i]['difficulty'] for i in indices]
        ax_hist.axvspan(min(d_vals) - 0.02, max(d_vals) + 0.02,
                        alpha=0.15, color=color, label=label)
    ax_hist.legend(fontsize=8)

    # Bottom 3 rows x 3 cols (each showing source + observation side by side)
    # Actually: 3 rows x 6 cols (source, obs for each of 3 examples)
    # Simpler: 3 rows x 3 cols, each cell shows source on top half, obs on bottom
    # Let's do 3 rows x 6 cols: pairs of (source, obs)
    n_example_rows = 3
    n_examples_per_row = 3
    bottom_height = 0.72
    row_height = bottom_height / n_example_rows
    col_width = 0.85 / (n_examples_per_row * 2)

    for row_idx, (row_label, indices) in enumerate(example_rows):
        for ex_idx, pair_idx in enumerate(indices):
            p = sorted_pairs[pair_idx]
            R = p['R']
            pos = p['pos']
            N = p['N']
            node_size = max(8, min(30, 1000 // max(N, 1)))

            # Source
            x0 = 0.08 + ex_idx * 2 * col_width
            y0 = 0.02 + (n_example_rows - 1 - row_idx) * row_height
            ax_src = fig.add_axes([x0, y0, col_width - 0.01, row_height - 0.02])
            draw_graph_with_dist(ax_src, R, pos, mu=p['mu_source'],
                                 title=('Source' if row_idx == 0 and ex_idx == 0 else ''),
                                 cmap='hot', node_size=node_size)
            if ex_idx == 0:
                ax_src.set_ylabel(f'{row_label}\n(diff={p["difficulty"]:.2f})',
                                  fontsize=7)

            # Observation
            ax_obs = fig.add_axes([x0 + col_width, y0, col_width - 0.01,
                                   row_height - 0.02])
            draw_graph_with_dist(ax_obs, R, pos, mu=p['obs'],
                                 title=('Observation' if row_idx == 0 and ex_idx == 0 else ''),
                                 cmap='hot', node_size=node_size)
            ax_obs.set_title(f'tau={p["tau"]:.2f}', fontsize=6)

    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


# -- Figure 2: Tau Effect ----------------------------------------------------

def plot_tau_effect(out_path):
    """One graph (grid_5x5 from training), one source.
    1 row x 7 cols. tau = 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0.
    Each panel: graph colored by observation.
    Title: tau=X.X, diff=X.XX."""
    rng = np.random.default_rng(123)

    print("  Generating training graphs...", flush=True)
    train_graphs = generate_training_graphs(seed=42)

    # Find grid_5x5
    target_graph = None
    for name, R, pos in train_graphs:
        if name == 'grid_5x5':
            target_graph = (name, R, pos)
            break
    if target_graph is None:
        # Fallback to first graph
        target_graph = train_graphs[0]

    name, R, pos = target_graph
    N = R.shape[0]

    # Generate a fixed source
    mu_source = generate_source_distribution(N, pos, rng, R=R)

    tau_values = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
    n_cols = len(tau_values)

    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.8, 3.5))
    node_size = max(15, min(50, 1500 // max(N, 1)))

    for col, tau in enumerate(tau_values):
        ax = axes[col]
        obs = generate_observation(mu_source, R, tau)
        diff = compute_difficulty(mu_source, obs)
        draw_graph_with_dist(ax, R, pos, mu=obs,
                             title=f'tau={tau:.1f}, diff={diff:.2f}',
                             cmap='hot', node_size=node_size)

    fig.suptitle(f'Observation vs Diffusion Time ({name}, N={N})',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


# -- Main ---------------------------------------------------------------------

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=== Experiment 18: Data Visualization ===", flush=True)

    print("\nPlotting difficulty spectrum...", flush=True)
    plot_difficulty_spectrum(
        os.path.join(out_dir, 'ex18_difficulty_spectrum.png'))

    print("Plotting tau effect...", flush=True)
    plot_tau_effect(
        os.path.join(out_dir, 'ex18_tau_effect.png'))

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
