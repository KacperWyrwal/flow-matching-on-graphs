"""
Experiment 10: Topology Generalization.

Trains a FlexibleConditionalGNNRateMatrixPredictor on a mix of 13 graph
topologies and tests whether it generalizes to unseen graph structures.
This is the key experiment for the "foundation model for distributions on
graphs" vision.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from scipy.linalg import expm

from graph_ot_fm import (
    total_variation,
    make_grid_graph, make_cycle_graph, make_path_graph,
    make_star_graph, make_complete_bipartite_graph,
    make_barbell_graph, make_petersen_graph,
)
from meta_fm import (
    FlexibleConditionalGNNRateMatrixPredictor,
    TopologyGeneralizationDataset,
    train_flexible_conditional,
    sample_trajectory_flexible,
    get_device,
)


# ── Graph definitions ───────────────────────────────────────────────────────

TRAINING_GRAPHS = [
    ('grid_3x3',      make_grid_graph(3, 3, weighted=False)),
    ('grid_4x4',      make_grid_graph(4, 4, weighted=False)),
    ('grid_5x5',      make_grid_graph(5, 5, weighted=False)),
    ('cycle_8',       make_cycle_graph(8)),
    ('cycle_10',      make_cycle_graph(10)),
    ('cycle_12',      make_cycle_graph(12)),
    ('path_6',        make_path_graph(6)),
    ('path_8',        make_path_graph(8)),
    ('path_10',       make_path_graph(10)),
    ('star_7',        make_star_graph(7)),
    ('star_9',        make_star_graph(9)),
    ('bipartite_3_3', make_complete_bipartite_graph(3, 3)),
    ('bipartite_4_4', make_complete_bipartite_graph(4, 4)),
]  # 13 topologies

TEST_GRAPHS = [
    ('grid_3x5',  make_grid_graph(3, 5, weighted=False)),   # 15 nodes, rectangular
    ('cycle_15',  make_cycle_graph(15)),                    # larger than any training cycle
    ('barbell',   make_barbell_graph(4, 3)),                # 11 nodes, qualitatively new
    ('petersen',  make_petersen_graph()),                   # 10 nodes, unusual structure
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_multipeak_dist(N, n_peaks, rng):
    """Returns (dist, peak_nodes)."""
    peak_nodes = rng.choice(N, size=n_peaks, replace=False).tolist()
    weights = rng.dirichlet(np.full(n_peaks, 2.0))
    dist = np.ones(N) * 0.2 / N
    for node, w in zip(peak_nodes, weights):
        dist[node] += 0.8 * w
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist, peak_nodes


def diffuse(mu, R, tau_diff):
    result = mu @ expm(tau_diff * R)
    result = np.clip(result, 1e-12, None)
    result /= result.sum()
    return result


def exact_inverse(mu_obs, R, tau_diff):
    result = mu_obs @ expm(-tau_diff * R)
    result = np.clip(result, 0.0, None)
    s = result.sum()
    return result / s if s > 1e-12 else mu_obs.copy()


def peak_recovery_topk(recovered, true_peaks):
    """Fraction of true peaks in top-k positions by mass."""
    k = len(true_peaks)
    top_k = set(np.argsort(recovered)[-k:].tolist())
    return len(top_k & set(true_peaks)) / k


def evaluate_graph(model, name, R, edge_index, n_test=20, tau_diff=0.8,
                   rng=None, device=None):
    """Evaluate model on one graph, return list of result dicts."""
    N = R.shape[0]
    if rng is None:
        rng = np.random.default_rng(42)

    results = []
    for _ in range(n_test):
        n_peaks = int(rng.integers(1, 4))
        mu_source, peak_nodes = make_multipeak_dist(N, n_peaks, rng)
        mu_obs = diffuse(mu_source, R, tau_diff)

        context = np.stack([mu_obs, np.full(N, tau_diff)], axis=-1)  # (N, 2)
        _, traj = sample_trajectory_flexible(
            model, mu_obs, context, edge_index, n_steps=200, device=device)
        mu_learned = traj[-1]
        mu_exact = exact_inverse(mu_obs, R, tau_diff)

        results.append({
            'mu_obs':    mu_obs,
            'mu_source': mu_source,
            'mu_learned': mu_learned,
            'mu_exact':  mu_exact,
            'peak_nodes': peak_nodes,
            'traj':      traj,
            'tv_learned': total_variation(mu_learned, mu_source),
            'tv_exact':   total_variation(mu_exact, mu_source),
            'peak_topk_learned': peak_recovery_topk(mu_learned, peak_nodes),
            'peak_topk_exact':   peak_recovery_topk(mu_exact, peak_nodes),
        })
    return results


def draw_graph_dist(ax, R, dist, title='', node_size=200):
    """Draw graph with node colors proportional to distribution mass."""
    try:
        import networkx as nx
        N = len(dist)
        G = nx.from_numpy_array(np.maximum(R, 0))
        G.remove_edges_from(nx.selfloop_edges(G))

        if N == 15 and R.shape == (15, 15):
            # Try to detect grid_3x5
            try:
                pos = {i: (i % 5, -(i // 5)) for i in range(N)}
            except Exception:
                pos = nx.spring_layout(G, seed=42)
        elif all(R[i, (i + 1) % N] > 0 for i in range(N)):
            pos = nx.circular_layout(G)
        elif N == 10 and max(np.sum(R > 0, axis=1)) == 3:
            # Petersen graph — shell layout
            pos = nx.shell_layout(G, nlist=[list(range(5)), list(range(5, 10))])
        else:
            pos = nx.spring_layout(G, seed=42)

        vmax = max(dist.max(), 0.01)
        node_colors = [dist[i] for i in range(N)]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               cmap='YlOrRd', vmin=0, vmax=vmax,
                               node_size=node_size)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray',
                               alpha=0.5, width=0.8)
    except ImportError:
        # Fallback: bar chart
        ax.bar(range(len(dist)), dist, color='steelblue', alpha=0.7)

    ax.set_title(title, fontsize=7)
    ax.axis('off')


def barbell_positions(clique_size=4, path_length=3):
    """Manual layout for barbell graph."""
    pos = {}
    # Clique 1: circle centered at (-2, 0) radius 0.5
    for i in range(clique_size):
        angle = 2 * np.pi * i / clique_size
        pos[i] = (-2.0 + 0.5 * np.cos(angle), 0.5 * np.sin(angle))
    # Path: evenly spaced between -1.2 and 1.2
    for j in range(path_length):
        x = -1.2 + 2.4 * j / max(path_length - 1, 1)
        pos[clique_size + j] = (x, 0.0)
    # Clique 2: circle centered at (2, 0) radius 0.5
    c2 = clique_size + path_length
    for i in range(clique_size):
        angle = 2 * np.pi * i / clique_size
        pos[c2 + i] = (2.0 + 0.5 * np.cos(angle), 0.5 * np.sin(angle))
    return pos


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-weighting', type=str, default='uniform',
                        choices=['original', 'uniform', 'linear'])
    parser.add_argument('--n-epochs', type=int, default=1000)
    parser.add_argument('--n-samples-per-graph', type=int, default=1000)
    parser.add_argument('--n-pairs-per-graph', type=int, default=50)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=4)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    device = get_device()

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        checkpoint_dir,
        f'meta_model_ex10_topology_{args.loss_weighting}_{args.n_epochs}ep_h{args.hidden_dim}_l{args.n_layers}.pt')

    print(f"=== Experiment 10: Topology Generalization ===")
    print(f"Training on {len(TRAINING_GRAPHS)} graph topologies")
    print(f"Testing on  {len(TEST_GRAPHS)} held-out topologies\n")

    # ── Model ──────────────────────────────────────────────────────────────
    model = FlexibleConditionalGNNRateMatrixPredictor(
        context_dim=2, hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"Loaded checkpoint from {ckpt_path}")
        losses = None
    else:
        total_samples = args.n_samples_per_graph * len(TRAINING_GRAPHS)
        print(f"Building TopologyGeneralizationDataset "
              f"({total_samples} total samples)...")
        dataset = TopologyGeneralizationDataset(
            graphs=TRAINING_GRAPHS,
            n_samples_per_graph=args.n_samples_per_graph,
            n_pairs_per_graph=args.n_pairs_per_graph,
            tau_diff_range=(0.3, 1.5),
            seed=42,
        )
        print(f"Dataset built: {len(dataset)} samples")

        print(f"Training FlexibleConditionalGNNRateMatrixPredictor "
              f"({args.n_epochs} epochs)...")
        history = train_flexible_conditional(
            model, dataset,
            n_epochs=args.n_epochs,
            batch_size=256,
            lr=1e-3,
            device=device,
            loss_weighting=args.loss_weighting,
        )
        losses = history['losses']
        print(f"Initial loss: {losses[0]:.6f}, Final loss: {losses[-1]:.6f}")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    model.eval()

    # ── Evaluation ─────────────────────────────────────────────────────────
    rng_eval = np.random.default_rng(99)

    # Evaluate training graphs (in-distribution, 20 test cases each)
    train_results = {}
    from meta_fm import rate_matrix_to_edge_index
    for name, R in TRAINING_GRAPHS:
        edge_index = rate_matrix_to_edge_index(R)
        res = evaluate_graph(model, name, R, edge_index, n_test=20,
                             tau_diff=0.8, rng=rng_eval, device=device)
        train_results[name] = res

    # Evaluate test graphs (out-of-distribution, 20 test cases each)
    test_results = {}
    for name, R in TEST_GRAPHS:
        edge_index = rate_matrix_to_edge_index(R)
        res = evaluate_graph(model, name, R, edge_index, n_test=20,
                             tau_diff=0.8, rng=rng_eval, device=device)
        test_results[name] = res

    # ── Console output ──────────────────────────────────────────────────────
    print("\n=== Experiment 10: Topology Generalization Results ===\n")
    print("Training topologies (in-distribution):")
    train_tvs = []
    for name, _ in TRAINING_GRAPHS:
        res = train_results[name]
        tv_l = np.mean([r['tv_learned'] for r in res])
        pk_l = np.mean([r['peak_topk_learned'] for r in res]) * 100
        train_tvs.append(tv_l)
        print(f"  {name:16s}: TV={tv_l:.4f}, peak_recovery={pk_l:.0f}%")

    print("\nTest topologies (out-of-distribution):")
    test_tvs = []
    for name, _ in TEST_GRAPHS:
        res = test_results[name]
        tv_l = np.mean([r['tv_learned'] for r in res])
        pk_l = np.mean([r['peak_topk_learned'] for r in res]) * 100
        test_tvs.append(tv_l)
        print(f"  {name:16s}: TV={tv_l:.4f}, peak_recovery={pk_l:.0f}%")

    mean_train_tv = np.mean(train_tvs)
    mean_test_tv = np.mean(test_tvs)
    ratio = mean_test_tv / mean_train_tv if mean_train_tv > 1e-12 else float('inf')
    print(f"\nMean TV (training): {mean_train_tv:.4f}")
    print(f"Mean TV (test):     {mean_test_tv:.4f}")
    print(f"Ratio:              {ratio:.4f} (want close to 1.0)")

    # ── Plots ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        'Experiment 10: Topology Generalization\n'
        '(trained on 13 topologies, tested on 4 held-out topologies)',
        fontsize=12)
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.35)

    train_colors = 'steelblue'
    test_colors = 'salmon'

    # Panel A: training loss
    ax_A = fig.add_subplot(gs[0, 0])
    if losses is not None:
        ax_A.plot(losses, color='steelblue', lw=1.5)
        ax_A.set_yscale('log')
    else:
        ax_A.text(0.5, 0.5, 'Loaded from\ncheckpoint',
                  transform=ax_A.transAxes, ha='center', va='center', fontsize=12)
    ax_A.set_xlabel('Epoch')
    ax_A.set_ylabel('Loss')
    ax_A.set_title('Panel A: Training Loss')
    ax_A.grid(True, alpha=0.3)

    # Panel B: TV by topology (train vs test)
    ax_B = fig.add_subplot(gs[0, 1])
    all_names = [n for n, _ in TRAINING_GRAPHS] + [n for n, _ in TEST_GRAPHS]
    all_tvs = train_tvs + test_tvs
    all_colors = [train_colors] * len(TRAINING_GRAPHS) + [test_colors] * len(TEST_GRAPHS)
    x_pos = np.arange(len(all_names))
    bars = ax_B.bar(x_pos, all_tvs, color=all_colors, alpha=0.85, width=0.7)
    ax_B.set_xticks(x_pos)
    ax_B.set_xticklabels(all_names, rotation=45, ha='right', fontsize=6)
    ax_B.set_ylabel('Mean TV (recovered vs true)')
    ax_B.set_title('Panel B: TV by topology')
    ax_B.grid(True, alpha=0.3, axis='y')
    from matplotlib.patches import Patch
    ax_B.legend(handles=[Patch(color=train_colors, label='Train'),
                          Patch(color=test_colors, label='Test')], fontsize=8)

    # Panel C: Example recoveries on test graphs (graph drawings)
    ax_C = fig.add_subplot(gs[0, 2])
    ax_C.axis('off')
    ax_C.set_title('Panel C: Example recoveries on test graphs', pad=12)
    inner_C = gs[0, 2].subgridspec(4, 2, hspace=0.8, wspace=0.3)
    row_labels = [n for n, _ in TEST_GRAPHS]
    col_labels = ['Observation', 'Recovered']
    for row_i, (name, R) in enumerate(TEST_GRAPHS):
        res = test_results[name]
        ex = res[0]  # first test case
        for col_i, (dist, col_lbl) in enumerate([
                (ex['mu_obs'], col_labels[0]),
                (ex['mu_learned'], col_labels[1])]):
            ax_in = fig.add_subplot(inner_C[row_i, col_i])
            edge_index = rate_matrix_to_edge_index(R)
            draw_graph_dist(ax_in, R, dist,
                            title=f'{name}\n{col_lbl}' if col_i == 0 else col_lbl,
                            node_size=80)

    # Panel D: Peak recovery by topology (train vs test)
    ax_D = fig.add_subplot(gs[1, 0])
    train_peaks = [np.mean([r['peak_topk_learned'] for r in train_results[n]]) * 100
                   for n, _ in TRAINING_GRAPHS]
    test_peaks = [np.mean([r['peak_topk_learned'] for r in test_results[n]]) * 100
                  for n, _ in TEST_GRAPHS]
    all_peaks = train_peaks + test_peaks
    ax_D.bar(x_pos, all_peaks, color=all_colors, alpha=0.85, width=0.7)
    ax_D.set_xticks(x_pos)
    ax_D.set_xticklabels(all_names, rotation=45, ha='right', fontsize=6)
    ax_D.set_ylabel('Peak recovery top-k (%)')
    ax_D.set_ylim(0, 115)
    ax_D.set_title('Panel D: Peak recovery by topology')
    ax_D.legend(handles=[Patch(color=train_colors, label='Train'),
                          Patch(color=test_colors, label='Test')], fontsize=8)
    ax_D.grid(True, alpha=0.3, axis='y')

    # Panel E: TV vs graph size (scatter)
    ax_E = fig.add_subplot(gs[1, 1])
    train_sizes = [R.shape[0] for _, R in TRAINING_GRAPHS]
    test_sizes = [R.shape[0] for _, R in TEST_GRAPHS]
    ax_E.scatter(train_sizes, train_tvs, color=train_colors, alpha=0.8,
                 s=80, zorder=3, label='Train')
    ax_E.scatter(test_sizes, test_tvs, color=test_colors, alpha=0.8,
                 s=100, marker='*', zorder=4, label='Test')
    for size, tv, name in (
            list(zip(train_sizes, train_tvs, [n for n, _ in TRAINING_GRAPHS])) +
            list(zip(test_sizes, test_tvs, [n for n, _ in TEST_GRAPHS]))):
        ax_E.annotate(name, (size, tv), fontsize=5, ha='left',
                      xytext=(3, 0), textcoords='offset points')
    ax_E.set_xlabel('Number of nodes')
    ax_E.set_ylabel('Mean TV (learned vs true)')
    ax_E.set_title('Panel E: TV vs graph size')
    ax_E.legend(fontsize=8)
    ax_E.grid(True, alpha=0.3)

    # Panel F: Recovery trajectory on barbell graph
    ax_F = fig.add_subplot(gs[1, 2])
    ax_F.axis('off')
    ax_F.set_title('Panel F: Recovery trajectory on barbell graph', pad=12)
    barbell_name, barbell_R = TEST_GRAPHS[2]  # ('barbell', ...)
    barbell_res = test_results[barbell_name]
    ex_bb = barbell_res[0]
    traj_bb = ex_bb['traj']
    n_traj = len(traj_bb)
    snap_idx = [0, n_traj // 4, n_traj // 2, 3 * n_traj // 4, n_traj - 1]
    snap_labels = ['t=0', 't=0.25', 't=0.5', 't=0.75', 't=1']
    inner_F = gs[1, 2].subgridspec(1, 5, wspace=0.3)
    barbell_edge_index = rate_matrix_to_edge_index(barbell_R)
    for si, (idx_t, tlbl) in enumerate(zip(snap_idx, snap_labels)):
        ax_in = fig.add_subplot(inner_F[0, si])
        draw_graph_dist(ax_in, barbell_R, traj_bb[idx_t], title=tlbl, node_size=60)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'ex10_topology_generalization.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
