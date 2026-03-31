"""
Experiment 11: Combined Generalization.

Combines Ex8.2 (multi-peak recovery) and Ex10 (topology generalization):
recover multi-peak sources with variable diffusion times on unseen topologies.

Trains FlexibleConditionalGNNRateMatrixPredictor on 13 topologies,
evaluates on 4 held-out topologies across tau_diff ∈ {0.5, 1.0, 1.4}
and n_peaks ∈ {1, 2, 3}.
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
    DirectGNNPredictor,
    TopologyGeneralizationDataset,
    train_flexible_conditional,
    train_direct_gnn,
    sample_trajectory_flexible,
    get_device,
)
from meta_fm.model import rate_matrix_to_edge_index


# ── Graph definitions (same as Ex10) ─────────────────────────────────────────

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
]

TEST_GRAPHS = [
    ('grid_3x5',  make_grid_graph(3, 5, weighted=False)),
    ('cycle_15',  make_cycle_graph(15)),
    ('barbell',   make_barbell_graph(4, 3)),
    ('petersen',  make_petersen_graph()),
]

TAU_DIFFS   = [0.5, 1.0, 1.4]
N_PEAKS_ALL = [1, 2, 3]
N_PER_CELL  = 5  # test cases per (topology, tau_diff, n_peaks)


# ── Helper functions ──────────────────────────────────────────────────────────

def make_multipeak_dist(N, n_peaks, rng):
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
    """Fraction of true peaks in top-k of recovered distribution."""
    k = len(true_peaks)
    top_k = set(np.argsort(recovered)[-k:].tolist())
    return len(top_k & set(true_peaks)) / k


def peak_location_topk(recovered, true_peaks, R):
    """Fraction of true peaks within graph distance ≤ 1 of a top-k recovered node."""
    k = len(true_peaks)
    top_k_set = set(np.argsort(recovered)[-k:].tolist())
    adj = R > 0  # off-diagonal positive entries = edges
    found = 0
    for p in true_peaks:
        neighborhood = set(np.where(adj[p])[0].tolist()) | {p}
        if top_k_set & neighborhood:
            found += 1
    return found / k


def argmax_peaks(mu_obs, n_peaks):
    """Return top-n_peaks nodes by mass in mu_obs."""
    return set(np.argsort(mu_obs)[-n_peaks:].tolist())


def draw_graph_dist(ax, R, dist, title='', node_size=150, peak_nodes=None):
    try:
        import networkx as nx
        N = len(dist)
        G = nx.from_numpy_array(np.maximum(R, 0))
        G.remove_edges_from(nx.selfloop_edges(G))

        if R.shape[0] == 15 and all(R[i % 15, (i + 1) % 15] == 0 for i in range(15)):
            # grid_3x5
            pos = {i: (i % 5, -(i // 5)) for i in range(15)}
        elif N <= 20 and all(R[i, (i + 1) % N] > 0 for i in range(N)):
            pos = nx.circular_layout(G)
        elif N == 10 and all(sum(1 for j in range(N) if R[i, j] > 0) == 3 for i in range(N)):
            pos = nx.shell_layout(G, nlist=[list(range(5)), list(range(5, 10))])
        else:
            pos = nx.spring_layout(G, seed=42)

        vmax = max(dist.max(), 0.01)
        node_colors = [dist[i] for i in range(N)]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               cmap='YlOrRd', vmin=0, vmax=vmax,
                               node_size=node_size)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray',
                               alpha=0.4, width=0.6)
        if peak_nodes is not None:
            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=peak_nodes,
                                   node_color='none', edgecolors='blue',
                                   linewidths=1.5, node_size=node_size * 1.4)
    except ImportError:
        ax.bar(range(len(dist)), dist, color='steelblue', alpha=0.7)

    ax.set_title(title, fontsize=6)
    ax.axis('off')


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_cell(model, R, edge_index, tau_diff, n_peaks, n_cases, rng, device,
                  direct_model=None):
    """Evaluate one (tau_diff, n_peaks) cell. Returns list of result dicts."""
    N = R.shape[0]
    results = []
    for _ in range(n_cases):
        mu_source, peak_nodes = make_multipeak_dist(N, n_peaks, rng)
        mu_obs = diffuse(mu_source, R, tau_diff)
        context = np.stack([mu_obs, np.full(N, tau_diff)], axis=-1)

        _, traj = sample_trajectory_flexible(
            model, mu_obs, context, edge_index, n_steps=200, device=device)
        mu_learned = traj[-1]
        mu_exact = exact_inverse(mu_obs, R, tau_diff)
        top_argmax = argmax_peaks(mu_obs, n_peaks)

        # DirectGNN prediction
        if direct_model is not None:
            try:
                ctx_t = torch.tensor(context, dtype=torch.float32, device=device)
                ei_dev = edge_index.to(device)
                with torch.no_grad():
                    mu_direct = direct_model(ctx_t, ei_dev).cpu().numpy()
            except Exception:
                mu_direct = mu_obs.copy()
        else:
            mu_direct = mu_obs.copy()

        results.append({
            'mu_source':  mu_source,
            'mu_obs':     mu_obs,
            'mu_learned': mu_learned,
            'mu_direct':  mu_direct,
            'mu_exact':   mu_exact,
            'peak_nodes': peak_nodes,
            'traj':       traj,
            'tv_learned': total_variation(mu_learned, mu_source),
            'tv_direct':  total_variation(mu_direct, mu_source),
            'tv_exact':   total_variation(mu_exact, mu_source),
            'peak_topk_learned':  peak_recovery_topk(mu_learned, peak_nodes),
            'peak_topk_direct':   peak_recovery_topk(mu_direct, peak_nodes),
            'peak_topk_exact':    peak_recovery_topk(mu_exact, peak_nodes),
            'peak_topk_argmax':   len(top_argmax & set(peak_nodes)) / n_peaks,
            'peak_loc_learned':   peak_location_topk(mu_learned, peak_nodes, R),
            'peak_loc_direct':    peak_location_topk(mu_direct, peak_nodes, R),
            'peak_loc_argmax':    peak_location_topk(
                                      np.array([1.0 if i in top_argmax else 0.0
                                                for i in range(N)]),
                                      peak_nodes, R),
        })
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-weighting', type=str, default='uniform',
                        choices=['original', 'uniform', 'linear'])
    parser.add_argument('--n-epochs', type=int, default=1000)
    parser.add_argument('--n-samples-per-graph', type=int, default=1000)
    parser.add_argument('--n-pairs-per-graph', type=int, default=75)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--direct-epochs', type=int, default=500)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    device = get_device()

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        checkpoint_dir,
        f'meta_model_ex11_combined_{args.loss_weighting}_{args.n_epochs}ep'
        f'_h{args.hidden_dim}_l{args.n_layers}.pt')

    print("=== Experiment 11: Combined Generalization ===")
    print(f"Training on {len(TRAINING_GRAPHS)} topologies, "
          f"testing on {len(TEST_GRAPHS)} held-out topologies\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FlexibleConditionalGNNRateMatrixPredictor(
        context_dim=2, hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"Loaded checkpoint from {ckpt_path}")
        losses = None
    else:
        total_samples = args.n_samples_per_graph * len(TRAINING_GRAPHS)
        print(f"Building TopologyGeneralizationDataset "
              f"({total_samples} total samples, "
              f"{args.n_pairs_per_graph} pairs/graph)...")
        dataset = TopologyGeneralizationDataset(
            graphs=TRAINING_GRAPHS,
            n_samples_per_graph=args.n_samples_per_graph,
            n_pairs_per_graph=args.n_pairs_per_graph,
            tau_diff_range=(0.3, 1.5),
            seed=42,
        )
        print(f"Dataset built: {len(dataset)} samples")

        print(f"Training FlexibleConditionalGNNRateMatrixPredictor "
              f"({args.n_epochs} epochs, lr={args.lr})...")
        history = train_flexible_conditional(
            model, dataset,
            n_epochs=args.n_epochs,
            batch_size=256,
            lr=args.lr,
            device=device,
            loss_weighting=args.loss_weighting,
        )
        losses = history['losses']
        print(f"Initial loss: {losses[0]:.6f}, Final loss: {losses[-1]:.6f}")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    model.eval()

    # ── DirectGNN baseline ───────────────────────────────────────────────────
    direct_model = DirectGNNPredictor(
        context_dim=2,  # [obs(a), tau_broadcast]
        hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    direct_ckpt = os.path.join(
        checkpoint_dir,
        f'direct_model_ex11_{args.direct_epochs}ep'
        f'_h{args.hidden_dim}_l{args.n_layers}.pt')

    if os.path.exists(direct_ckpt):
        direct_model.load_state_dict(torch.load(direct_ckpt, map_location='cpu'))
        print(f"Loaded DirectGNN from {direct_ckpt}")
    else:
        # Build training pairs from training graphs
        print("Building DirectGNN training pairs...")
        direct_rng = np.random.default_rng(123)
        direct_pairs = []
        for name, R in TRAINING_GRAPHS:
            N = R.shape[0]
            ei = rate_matrix_to_edge_index(R)
            for _ in range(args.n_pairs_per_graph):
                n_pk = int(direct_rng.integers(1, 4))
                mu_src, _ = make_multipeak_dist(N, n_pk, direct_rng)
                tau = float(direct_rng.uniform(0.3, 1.5))
                mu_obs = diffuse(mu_src, R, tau)
                ctx = np.stack([mu_obs, np.full(N, tau)], axis=-1).astype(np.float32)
                direct_pairs.append((ctx, mu_src.astype(np.float32), ei))

        print(f"Training DirectGNN ({args.direct_epochs} epochs, "
              f"{len(direct_pairs)} pairs)...")
        train_direct_gnn(
            direct_model, direct_pairs,
            n_epochs=args.direct_epochs, lr=args.lr,
            device=device, seed=42, ema_decay=0.999)
        torch.save(direct_model.state_dict(), direct_ckpt)
        print(f"Saved DirectGNN to {direct_ckpt}")

    direct_model.eval()

    # ── Structured evaluation: 4 × 3 × 3 × 5 = 180 test cases ───────────────
    rng_eval = np.random.default_rng(99)

    # results[graph_name][tau_diff][n_peaks] = list of result dicts
    results = {}
    for name, R in TEST_GRAPHS:
        results[name] = {}
        ei = rate_matrix_to_edge_index(R)
        for tau_diff in TAU_DIFFS:
            results[name][tau_diff] = {}
            for n_peaks in N_PEAKS_ALL:
                results[name][tau_diff][n_peaks] = evaluate_cell(
                    model, R, ei, tau_diff, n_peaks,
                    N_PER_CELL, rng_eval, device,
                    direct_model=direct_model)

    # ── Console output ────────────────────────────────────────────────────────
    def flat_results(filter_fn=None):
        out = []
        for name, R in TEST_GRAPHS:
            for tau_diff in TAU_DIFFS:
                for n_peaks in N_PEAKS_ALL:
                    for r in results[name][tau_diff][n_peaks]:
                        if filter_fn is None or filter_fn(name, tau_diff, n_peaks):
                            out.append(r)
        return out

    all_r = flat_results()
    print("\n=== Experiment 11: Combined Generalization ===\n")
    print(f"Overall ({len(all_r)} test cases):")
    print(f"  Mean TV (learned):        {np.mean([r['tv_learned'] for r in all_r]):.4f}")
    print(f"  Mean TV (DirectGNN):      {np.mean([r['tv_direct'] for r in all_r]):.4f}")
    print(f"  Mean TV (exact):          {np.mean([r['tv_exact'] for r in all_r]):.4f}")
    print(f"  Peak recovery (learned):  "
          f"{np.mean([r['peak_topk_learned'] for r in all_r])*100:.1f}%")
    print(f"  Peak recovery (DirectGNN):"
          f" {np.mean([r['peak_topk_direct'] for r in all_r])*100:.1f}%")
    print(f"  Peak recovery (argmax):   "
          f"{np.mean([r['peak_topk_argmax'] for r in all_r])*100:.1f}%")

    print("\nBy test topology:")
    for name, _ in TEST_GRAPHS:
        r_g = flat_results(lambda n, td, np_: n == name)
        tv_l = np.mean([r['tv_learned'] for r in r_g])
        tv_d = np.mean([r['tv_direct'] for r in r_g])
        pk   = np.mean([r['peak_topk_learned'] for r in r_g]) * 100
        pk_d = np.mean([r['peak_topk_direct'] for r in r_g]) * 100
        print(f"  {name:10s}: TV(FM)={tv_l:.4f}, TV(GNN)={tv_d:.4f}, "
              f"peak(FM)={pk:.0f}%, peak(GNN)={pk_d:.0f}%")

    print("\nBy tau_diff:")
    for tau_diff in TAU_DIFFS:
        r_t = flat_results(lambda n, td, np_: td == tau_diff)
        tv_l = np.mean([r['tv_learned'] for r in r_t])
        tv_d = np.mean([r['tv_direct'] for r in r_t])
        pk_l = np.mean([r['peak_topk_learned'] for r in r_t]) * 100
        pk_d = np.mean([r['peak_topk_direct'] for r in r_t]) * 100
        pk_a = np.mean([r['peak_topk_argmax'] for r in r_t]) * 100
        print(f"  tau={tau_diff}: TV(FM)={tv_l:.4f}, TV(GNN)={tv_d:.4f}, "
              f"peak(FM)={pk_l:.0f}%, peak(GNN)={pk_d:.0f}%, peak(argmax)={pk_a:.0f}%")

    print("\nBy n_peaks:")
    for n_peaks in N_PEAKS_ALL:
        r_p = flat_results(lambda n, td, np_: np_ == n_peaks)
        tv_l = np.mean([r['tv_learned'] for r in r_p])
        tv_d = np.mean([r['tv_direct'] for r in r_p])
        pk_l = np.mean([r['peak_topk_learned'] for r in r_p]) * 100
        pk_d = np.mean([r['peak_topk_direct'] for r in r_p]) * 100
        print(f"  {n_peaks} peak(s): TV(FM)={tv_l:.4f}, TV(GNN)={tv_d:.4f}, "
              f"peak(FM)={pk_l:.0f}%, peak(GNN)={pk_d:.0f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        'Experiment 11: Combined Generalization\n'
        '(multi-peak sources, variable τ_diff, unseen topologies)',
        fontsize=12)
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.38)

    # ── Panel A: Training loss ────────────────────────────────────────────────
    ax_A = fig.add_subplot(gs[0, 0])
    if losses is not None:
        ax_A.plot(losses, color='steelblue', lw=1.5)
        ax_A.set_yscale('log')
        ax_A.set_xlabel('Epoch')
        ax_A.set_ylabel('Loss')
    else:
        ax_A.text(0.5, 0.5, 'Loaded from\ncheckpoint',
                  transform=ax_A.transAxes, ha='center', va='center', fontsize=12)
    ax_A.set_title('A: Training Loss')
    ax_A.grid(True, alpha=0.3)

    # ── Panel B: TV heatmap (topologies × tau_diff) ───────────────────────────
    ax_B = fig.add_subplot(gs[0, 1])
    test_names = [n for n, _ in TEST_GRAPHS]
    tv_matrix = np.array([
        [np.mean([r['tv_learned']
                  for np_ in N_PEAKS_ALL
                  for r in results[name][tau_diff][np_]])
         for tau_diff in TAU_DIFFS]
        for name in test_names
    ])
    im_B = ax_B.imshow(tv_matrix, aspect='auto', cmap='YlOrRd')
    ax_B.set_xticks(range(len(TAU_DIFFS)))
    ax_B.set_xticklabels([f'τ={t}' for t in TAU_DIFFS])
    ax_B.set_yticks(range(len(test_names)))
    ax_B.set_yticklabels(test_names, fontsize=8)
    for i in range(len(test_names)):
        for j in range(len(TAU_DIFFS)):
            ax_B.text(j, i, f'{tv_matrix[i, j]:.3f}', ha='center', va='center',
                      fontsize=7, color='black' if tv_matrix[i, j] < tv_matrix.max() * 0.7 else 'white')
    fig.colorbar(im_B, ax=ax_B, fraction=0.046, pad=0.04)
    # Marginal means
    ax_B.set_xlabel(
        'τ means: ' + '  '.join(f'{t}→{tv_matrix[:,j].mean():.3f}' for j, t in enumerate(TAU_DIFFS)),
        fontsize=6)
    ax_B.set_title('B: TV Heatmap (topology × τ_diff)')

    # ── Panel C: TV by n_peaks (bar chart, learned vs exact) ─────────────────
    ax_C = fig.add_subplot(gs[0, 2])
    x = np.arange(len(N_PEAKS_ALL))
    width = 0.25
    tv_by_peaks_learned = [
        np.mean([r['tv_learned']
                 for name, _ in TEST_GRAPHS
                 for tau_diff in TAU_DIFFS
                 for r in results[name][tau_diff][np_]])
        for np_ in N_PEAKS_ALL]
    tv_by_peaks_direct = [
        np.mean([r['tv_direct']
                 for name, _ in TEST_GRAPHS
                 for tau_diff in TAU_DIFFS
                 for r in results[name][tau_diff][np_]])
        for np_ in N_PEAKS_ALL]
    tv_by_peaks_exact = [
        np.mean([r['tv_exact']
                 for name, _ in TEST_GRAPHS
                 for tau_diff in TAU_DIFFS
                 for r in results[name][tau_diff][np_]])
        for np_ in N_PEAKS_ALL]
    ax_C.bar(x - width, tv_by_peaks_learned, width, label='FM', color='steelblue', alpha=0.85)
    ax_C.bar(x, tv_by_peaks_direct, width, label='DirectGNN', color='tab:red', alpha=0.85)
    ax_C.bar(x + width, tv_by_peaks_exact, width, label='Exact inverse', color='salmon', alpha=0.85)
    ax_C.set_xticks(x)
    ax_C.set_xticklabels([f'{np_} peak(s)' for np_ in N_PEAKS_ALL])
    ax_C.set_ylabel('Mean TV')
    ax_C.set_title('C: TV by Number of Peaks')
    ax_C.legend(fontsize=8)
    ax_C.grid(True, alpha=0.3, axis='y')

    # ── Panel D: Peak recovery heatmap (topologies × tau_diff) ───────────────
    ax_D = fig.add_subplot(gs[1, 0])
    pk_matrix = np.array([
        [np.mean([r['peak_topk_learned']
                  for np_ in N_PEAKS_ALL
                  for r in results[name][tau_diff][np_]]) * 100
         for tau_diff in TAU_DIFFS]
        for name in test_names
    ])
    im_D = ax_D.imshow(pk_matrix, aspect='auto', cmap='YlGn', vmin=0, vmax=100)
    ax_D.set_xticks(range(len(TAU_DIFFS)))
    ax_D.set_xticklabels([f'τ={t}' for t in TAU_DIFFS])
    ax_D.set_yticks(range(len(test_names)))
    ax_D.set_yticklabels(test_names, fontsize=8)
    for i in range(len(test_names)):
        for j in range(len(TAU_DIFFS)):
            ax_D.text(j, i, f'{pk_matrix[i, j]:.0f}%', ha='center', va='center',
                      fontsize=7, color='black' if pk_matrix[i, j] > 40 else 'white')
    fig.colorbar(im_D, ax=ax_D, fraction=0.046, pad=0.04)
    ax_D.set_title('D: Peak Recovery Heatmap (topology × τ_diff)')

    # ── Panel E: Learned vs argmax peak location recovery vs tau_diff ─────────
    ax_E = fig.add_subplot(gs[1, 1])
    loc_learned_by_tau = [
        np.mean([r['peak_loc_learned']
                 for name, _ in TEST_GRAPHS
                 for np_ in N_PEAKS_ALL
                 for r in results[name][tau_diff][np_]]) * 100
        for tau_diff in TAU_DIFFS]
    loc_direct_by_tau = [
        np.mean([r['peak_loc_direct']
                 for name, _ in TEST_GRAPHS
                 for np_ in N_PEAKS_ALL
                 for r in results[name][tau_diff][np_]]) * 100
        for tau_diff in TAU_DIFFS]
    loc_argmax_by_tau = [
        np.mean([r['peak_loc_argmax']
                 for name, _ in TEST_GRAPHS
                 for np_ in N_PEAKS_ALL
                 for r in results[name][tau_diff][np_]]) * 100
        for tau_diff in TAU_DIFFS]
    ax_E.plot(TAU_DIFFS, loc_learned_by_tau, 'o-', color='steelblue',
              lw=2, ms=7, label='FM')
    ax_E.plot(TAU_DIFFS, loc_direct_by_tau, '^-', color='tab:red',
              lw=2, ms=7, label='DirectGNN')
    ax_E.plot(TAU_DIFFS, loc_argmax_by_tau, 's--', color='tomato',
              lw=2, ms=7, label='Argmax baseline')
    ax_E.set_xlabel('τ_diff')
    ax_E.set_ylabel('Peak location recovery (%, dist ≤ 1)')
    ax_E.set_ylim(0, 105)
    ax_E.set_xticks(TAU_DIFFS)
    ax_E.set_title('E: Learned vs Argmax Peak Location Recovery')
    ax_E.legend(fontsize=9)
    ax_E.grid(True, alpha=0.3)

    # ── Panel F: Trajectory gallery (1-peak, 2-peak, 3-peak examples) ─────────
    ax_F = fig.add_subplot(gs[1, 2])
    ax_F.axis('off')
    ax_F.set_title('F: Trajectory Gallery (t = 0 → 0.5 → 1)', pad=12)
    inner_F = gs[1, 2].subgridspec(3, 3, hspace=0.8, wspace=0.15)

    # Pick one example per n_peaks from different test graphs
    gallery_specs = [
        (0, 'grid_3x5',  TEST_GRAPHS[0][1], 1, 0.8),   # 1 peak, grid_3x5
        (1, 'barbell',   TEST_GRAPHS[2][1], 2, 1.0),   # 2 peaks, barbell
        (2, 'petersen',  TEST_GRAPHS[3][1], 3, 1.2),   # 3 peaks, petersen
    ]
    rng_gal = np.random.default_rng(7)
    for row_i, (_, gname, R_g, n_pk, tau_g) in enumerate(gallery_specs):
        N_g = R_g.shape[0]
        ei_g = rate_matrix_to_edge_index(R_g)
        mu_src, peak_ns = make_multipeak_dist(N_g, n_pk, rng_gal)
        mu_ob = diffuse(mu_src, R_g, tau_g)
        ctx_g = np.stack([mu_ob, np.full(N_g, tau_g)], axis=-1)
        times, traj = sample_trajectory_flexible(
            model, mu_ob, ctx_g, ei_g, n_steps=200, device=device)

        step_half = len(times) // 2
        snapshots = [
            (traj[0],         't=0 (obs)'),
            (traj[step_half], 't=0.5'),
            (traj[-1],        't=1 (rec)'),
        ]
        for col_i, (snap, label) in enumerate(snapshots):
            ax_in = fig.add_subplot(inner_F[row_i, col_i])
            pk_show = peak_ns if col_i == 2 else None  # show true peaks on final
            title = f'{gname}\n{n_pk}pk, τ={tau_g}\n{label}' if col_i == 0 else label
            draw_graph_dist(ax_in, R_g, snap, title=title, node_size=60,
                            peak_nodes=pk_show)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'ex11_combined_generalization.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
