"""
Experiment 11d: Dirichlet-Start Variant of Ex11.

Isolates the performance cost of Dirichlet-start posterior sampling vs
observation-start point estimation. Runs Ex11's task (multi-peak recovery on
unseen topologies) with Dirichlet random starts and FiLM conditioning,
then compares three-way: FM obs-start (Ex11), FM Dirichlet-start (this),
and DirectGNN (from Ex11).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from scipy.linalg import expm
from scipy.stats import pearsonr

from graph_ot_fm import (
    GraphStructure,
    GeodesicCache,
    compute_cost_matrix,
    total_variation,
    make_grid_graph, make_cycle_graph, make_path_graph,
    make_star_graph, make_complete_bipartite_graph,
    make_barbell_graph, make_petersen_graph,
)
from graph_ot_fm.ot_solver import compute_ot_coupling
from graph_ot_fm.flow import marginal_distribution_fast, marginal_rate_matrix_fast

from meta_fm import (
    FiLMConditionalGNNRateMatrixPredictor,
    train_film_conditional,
    get_device,
)
from meta_fm.model import rate_matrix_to_edge_index
from meta_fm.sample import sample_posterior_film

from experiments.ex11_combined_generalization import (
    TRAINING_GRAPHS, TEST_GRAPHS, TAU_DIFFS, N_PEAKS_ALL, N_PER_CELL,
    make_multipeak_dist, diffuse, exact_inverse,
    peak_recovery_topk, peak_location_topk, draw_graph_dist,
)


# ── Ex11 reference numbers (from previous run) ──────────────────────────────

EX11_FM_OBS = {
    'overall':   {'tv': 0.054, 'peak': 98.0},
    'grid_3x5':  {'tv': 0.035, 'peak': 97.0},
    'cycle_15':  {'tv': 0.013, 'peak': 100.0},
    'barbell':   {'tv': 0.118, 'peak': 97.0},
    'petersen':  {'tv': 0.048, 'peak': 99.0},
    'tau=0.5':   {'tv': 0.022, 'peak': 99.0},
    'tau=1.0':   {'tv': 0.042, 'peak': 99.0},
    'tau=1.4':   {'tv': 0.098, 'peak': 97.0},
}

EX11_DIRECT_GNN = {
    'overall':   {'tv': 0.038, 'peak': 99.0},
    'grid_3x5':  {'tv': 0.017, 'peak': 100.0},
    'cycle_15':  {'tv': 0.025, 'peak': 100.0},
    'barbell':   {'tv': 0.097, 'peak': 96.0},
    'petersen':  {'tv': 0.013, 'peak': 99.0},
    'tau=0.5':   {'tv': 0.020, 'peak': 99.0},
    'tau=1.0':   {'tv': 0.030, 'peak': 100.0},
    'tau=1.4':   {'tv': 0.064, 'peak': 97.0},
}


# ── Dataset ──────────────────────────────────────────────────────────────────

class DirichletStartDataset(torch.utils.data.Dataset):
    """
    Flow matching dataset with Dirichlet random starts.

    For each (graph, source, tau) pair, generates n_starts_per_pair
    Dirichlet random starts. OT coupling goes from random start → source,
    conditioned on observation.

    Returns 7-tuple for train_film_conditional:
        (mu_tau, tau, node_ctx, global_ctx, u_tilde, edge_index, N)
    """

    def __init__(self, graphs, n_pairs_per_graph=75,
                 n_samples_per_graph=1000, n_starts_per_pair=5,
                 dirichlet_alpha=1.0, tau_diff_range=(0.3, 1.5), seed=42):
        rng = np.random.default_rng(seed)
        all_items = []
        self.all_pairs = []

        total_samples = n_samples_per_graph * len(graphs)
        n_per = max(1, total_samples // (
            len(graphs) * n_pairs_per_graph * n_starts_per_pair))

        for name, R in graphs:
            N = R.shape[0]
            graph_struct = GraphStructure(R)
            geo_cache = GeodesicCache(graph_struct)
            edge_index = rate_matrix_to_edge_index(R)

            for pair_idx in range(n_pairs_per_graph):
                n_peaks = int(rng.integers(1, 4))
                mu_source, peak_nodes = make_multipeak_dist(N, n_peaks, rng)
                tau_val = float(rng.uniform(*tau_diff_range))
                mu_obs = diffuse(mu_source, R, tau_val)

                # Node context: observation (N, 1)
                node_ctx = mu_obs[:, None].astype(np.float32)
                # Global context: tau (1,)
                global_ctx = np.array([tau_val], dtype=np.float32)

                for start_idx in range(n_starts_per_pair):
                    # Dirichlet random start
                    mu_start = rng.dirichlet(np.full(N, dirichlet_alpha))

                    # OT coupling from random start to source
                    coupling = compute_ot_coupling(
                        mu_start, mu_source, graph_struct=graph_struct)
                    geo_cache.precompute_for_coupling(coupling)

                    self.all_pairs.append({
                        'name': name,
                        'N': N,
                        'R': R,
                        'edge_index': edge_index,
                        'mu_source': mu_source,
                        'mu_obs': mu_obs,
                        'mu_start': mu_start,
                        'tau': tau_val,
                        'peak_nodes': peak_nodes,
                    })

                    # Sample flow matching training tuples
                    for _ in range(n_per):
                        tau_flow = float(rng.uniform(0.0, 0.999))
                        mu_tau = marginal_distribution_fast(
                            geo_cache, coupling, tau_flow)
                        R_target = marginal_rate_matrix_fast(
                            geo_cache, coupling, tau_flow)
                        u_tilde = R_target * (1.0 - tau_flow)

                        all_items.append((
                            torch.tensor(mu_tau, dtype=torch.float32),
                            torch.tensor([tau_flow], dtype=torch.float32),
                            torch.tensor(node_ctx, dtype=torch.float32),
                            torch.tensor(global_ctx, dtype=torch.float32),
                            torch.tensor(u_tilde, dtype=torch.float32),
                            edge_index, N,
                        ))

        idx = rng.permutation(len(all_items))
        n_actual = min(total_samples, len(all_items))
        self.samples = [all_items[i] for i in idx[:n_actual]]
        print(f"  Dataset: {len(self.samples)} samples from "
              f"{len(self.all_pairs)} pairs on {len(graphs)} graphs",
              flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_cell_dirichlet(model, R, edge_index, tau_diff, n_peaks,
                            n_cases, rng, device, posterior_k=20,
                            dirichlet_alpha=1.0, n_ode_steps=100):
    """Evaluate one (tau_diff, n_peaks) cell with posterior sampling."""
    N = R.shape[0]
    results = []

    for _ in range(n_cases):
        mu_source, peak_nodes = make_multipeak_dist(N, n_peaks, rng)
        mu_obs = diffuse(mu_source, R, tau_diff)

        # Node context: observation
        node_ctx = mu_obs[:, None].astype(np.float32)
        global_ctx = np.array([tau_diff], dtype=np.float32)

        # Posterior sampling: K Dirichlet starts
        mu_starts = rng.dirichlet(
            np.full(N, dirichlet_alpha), size=posterior_k)

        try:
            fm_samples = sample_posterior_film(
                model, mu_starts, node_ctx, global_ctx, edge_index,
                n_steps=n_ode_steps, device=device)  # (K, N)
        except Exception as e:
            print(f"  Posterior sampling failed: {e}", flush=True)
            fm_samples = np.tile(mu_obs, (posterior_k, 1))

        # Posterior mean
        fm_mean = fm_samples.mean(axis=0)
        fm_mean = np.clip(fm_mean, 0, None)
        fm_mean /= fm_mean.sum() + 1e-15

        # Posterior std per node
        fm_std = fm_samples.std(axis=0)  # (N,)

        # Error per node
        error = np.abs(fm_mean - mu_source)  # (N,)

        # Calibration: Pearson r between std and |error|
        if fm_std.std() > 1e-10 and error.std() > 1e-10:
            r_cal, _ = pearsonr(fm_std, error)
        else:
            r_cal = 0.0

        # Diversity: mean pairwise TV between posterior samples
        diversity_tvs = []
        for i in range(min(posterior_k, 10)):
            for j in range(i + 1, min(posterior_k, 10)):
                diversity_tvs.append(
                    total_variation(fm_samples[i], fm_samples[j]))
        diversity = float(np.mean(diversity_tvs)) if diversity_tvs else 0.0

        # Exact inverse baseline
        mu_exact = exact_inverse(mu_obs, R, tau_diff)

        results.append({
            'mu_source': mu_source,
            'mu_obs': mu_obs,
            'mu_learned': fm_mean,
            'mu_exact': mu_exact,
            'peak_nodes': peak_nodes,
            'fm_std': fm_std,
            'error': error,
            'tv_learned': total_variation(fm_mean, mu_source),
            'tv_exact': total_variation(mu_exact, mu_source),
            'peak_topk_learned': peak_recovery_topk(fm_mean, peak_nodes),
            'peak_topk_exact': peak_recovery_topk(mu_exact, peak_nodes),
            'calibration_r': r_cal,
            'diversity': diversity,
        })

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ex11d: Dirichlet-start variant of Ex11')
    parser.add_argument('--n-epochs', type=int, default=1000)
    parser.add_argument('--n-samples-per-graph', type=int, default=1000)
    parser.add_argument('--n-pairs-per-graph', type=int, default=75)
    parser.add_argument('--n-starts-per-pair', type=int, default=5)
    parser.add_argument('--dirichlet-alpha', type=float, default=1.0)
    parser.add_argument('--posterior-k', type=int, default=20)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    device = get_device()
    print(f"Device: {device}", flush=True)

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        checkpoint_dir,
        f'ex11d_dirichlet_film_{args.n_epochs}ep'
        f'_h{args.hidden_dim}_l{args.n_layers}.pt')

    print("\n=== Experiment 11d: Dirichlet-Start Variant ===", flush=True)
    print(f"Training on {len(TRAINING_GRAPHS)} topologies, "
          f"testing on {len(TEST_GRAPHS)} held-out topologies\n", flush=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = FiLMConditionalGNNRateMatrixPredictor(
        node_context_dim=1,  # observation per node
        global_dim=1,        # tau
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    )

    if args.resume and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"Loaded checkpoint from {ckpt_path}", flush=True)
        losses = None
    else:
        print("Building DirichletStartDataset...", flush=True)
        dataset = DirichletStartDataset(
            graphs=TRAINING_GRAPHS,
            n_pairs_per_graph=args.n_pairs_per_graph,
            n_samples_per_graph=args.n_samples_per_graph,
            n_starts_per_pair=args.n_starts_per_pair,
            dirichlet_alpha=args.dirichlet_alpha,
            tau_diff_range=(0.3, 1.5),
            seed=42,
        )

        print(f"Training FiLMConditionalGNNRateMatrixPredictor "
              f"({args.n_epochs} epochs, lr={args.lr})...", flush=True)
        history = train_film_conditional(
            model, dataset,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            loss_weighting='uniform',
            loss_type='rate_kl',
            ema_decay=0.999,
        )
        losses = history['losses']
        print(f"Initial loss: {losses[0]:.6f}, Final loss: {losses[-1]:.6f}",
              flush=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved to {ckpt_path}", flush=True)

    model.eval()

    # ── Evaluation ───────────────────────────────────────────────────────────
    print("\nEvaluating...", flush=True)
    rng_eval = np.random.default_rng(99)

    results = {}
    for name, R in TEST_GRAPHS:
        results[name] = {}
        ei = rate_matrix_to_edge_index(R)
        for tau_diff in TAU_DIFFS:
            results[name][tau_diff] = {}
            for n_peaks in N_PEAKS_ALL:
                results[name][tau_diff][n_peaks] = evaluate_cell_dirichlet(
                    model, R, ei, tau_diff, n_peaks,
                    N_PER_CELL, rng_eval, device,
                    posterior_k=args.posterior_k,
                    dirichlet_alpha=args.dirichlet_alpha)

    # ── Collect flat results ─────────────────────────────────────────────────
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

    # ── Three-way comparison table ───────────────────────────────────────────
    print("\n=== Three-Way Comparison ===\n", flush=True)
    header = (f"{'':15s} {'FM obs-start':>14s}  {'FM Dirichlet':>14s}     "
              f"{'DirectGNN':>14s}")
    print(header, flush=True)
    subheader = (f"{'':15s} {'TV':>6s} {'Peak%':>6s}  "
                 f"{'TV':>6s} {'Peak%':>6s} {'r':>5s}  "
                 f"{'TV':>6s} {'Peak%':>6s}")
    print(subheader, flush=True)
    print("-" * 75, flush=True)

    def print_row(label, obs_key, dirichlet_results):
        obs = EX11_FM_OBS.get(obs_key, {'tv': float('nan'), 'peak': float('nan')})
        gnn = EX11_DIRECT_GNN.get(obs_key, {'tv': float('nan'), 'peak': float('nan')})
        tv_d = np.mean([r['tv_learned'] for r in dirichlet_results])
        pk_d = np.mean([r['peak_topk_learned'] for r in dirichlet_results]) * 100
        cal_d = np.mean([r['calibration_r'] for r in dirichlet_results])
        print(f"{label:15s} {obs['tv']:6.3f} {obs['peak']:5.0f}%  "
              f"{tv_d:6.3f} {pk_d:5.0f}% {cal_d:5.2f}  "
              f"{gnn['tv']:6.3f} {gnn['peak']:5.0f}%", flush=True)

    # By topology
    for name, _ in TEST_GRAPHS:
        r_g = flat_results(lambda n, td, np_: n == name)
        print_row(name, name, r_g)

    # Overall
    print("-" * 75, flush=True)
    print_row("Overall", 'overall', all_r)

    # By tau
    print("", flush=True)
    for tau_diff in TAU_DIFFS:
        r_t = flat_results(lambda n, td, np_: td == tau_diff)
        print_row(f"tau={tau_diff}", f'tau={tau_diff}', r_t)

    # Calibration and diversity summary
    mean_cal = np.mean([r['calibration_r'] for r in all_r])
    mean_div = np.mean([r['diversity'] for r in all_r])
    print(f"\nCalibration (Pearson r between posterior std and |error|): "
          f"{mean_cal:.3f}", flush=True)
    print(f"Posterior diversity (mean pairwise TV): {mean_div:.4f}", flush=True)

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ex11d: Dirichlet-Start vs Obs-Start vs DirectGNN', fontsize=12)

    # Panel A: TV comparison bar chart by topology
    ax = axes[0, 0]
    test_names = [n for n, _ in TEST_GRAPHS]
    x = np.arange(len(test_names))
    width = 0.25

    tv_obs = [EX11_FM_OBS.get(n, {'tv': 0})['tv'] for n in test_names]
    tv_dir = [np.mean([r['tv_learned']
               for r in flat_results(lambda nm, td, np_: nm == n)])
              for n in test_names]
    tv_gnn = [EX11_DIRECT_GNN.get(n, {'tv': 0})['tv'] for n in test_names]

    ax.bar(x - width, tv_obs, width, label='FM obs-start', color='steelblue', alpha=0.85)
    ax.bar(x, tv_dir, width, label='FM Dirichlet', color='tab:orange', alpha=0.85)
    ax.bar(x + width, tv_gnn, width, label='DirectGNN', color='tab:red', alpha=0.85)
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Exact inverse')
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, fontsize=8)
    ax.set_ylabel('Mean TV')
    ax.set_title('(A) TV by Test Topology')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: TV by tau
    ax = axes[0, 1]
    tau_obs = [EX11_FM_OBS.get(f'tau={t}', {'tv': 0})['tv'] for t in TAU_DIFFS]
    tau_dir = [np.mean([r['tv_learned']
                for r in flat_results(lambda n, td, np_: td == t)])
               for t in TAU_DIFFS]
    tau_gnn = [EX11_DIRECT_GNN.get(f'tau={t}', {'tv': 0})['tv'] for t in TAU_DIFFS]

    ax.plot(TAU_DIFFS, tau_obs, 'o-', color='steelblue', lw=2, label='FM obs-start')
    ax.plot(TAU_DIFFS, tau_dir, 's-', color='tab:orange', lw=2, label='FM Dirichlet')
    ax.plot(TAU_DIFFS, tau_gnn, '^-', color='tab:red', lw=2, label='DirectGNN')
    ax.set_xlabel('τ_diff')
    ax.set_ylabel('Mean TV')
    ax.set_xticks(TAU_DIFFS)
    ax.set_title('(B) TV by Diffusion Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Peak recovery by topology
    ax = axes[1, 0]
    pk_obs = [EX11_FM_OBS.get(n, {'peak': 0})['peak'] for n in test_names]
    pk_dir = [np.mean([r['peak_topk_learned']
               for r in flat_results(lambda nm, td, np_: nm == n)]) * 100
              for n in test_names]
    pk_gnn = [EX11_DIRECT_GNN.get(n, {'peak': 0})['peak'] for n in test_names]

    ax.bar(x - width, pk_obs, width, label='FM obs-start', color='steelblue', alpha=0.85)
    ax.bar(x, pk_dir, width, label='FM Dirichlet', color='tab:orange', alpha=0.85)
    ax.bar(x + width, pk_gnn, width, label='DirectGNN', color='tab:red', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(test_names, fontsize=8)
    ax.set_ylabel('Peak Recovery (%)')
    ax.set_ylim(0, 105)
    ax.set_title('(C) Peak Recovery by Topology')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel D: Calibration scatter (Dirichlet only)
    ax = axes[1, 1]
    all_stds = np.concatenate([r['fm_std'] for r in all_r])
    all_errors = np.concatenate([r['error'] for r in all_r])
    r_overall, _ = pearsonr(all_stds, all_errors)

    ax.scatter(all_stds, all_errors, s=2, alpha=0.15, c='tab:orange')
    # Fit line
    if all_stds.std() > 1e-10:
        m, b = np.polyfit(all_stds, all_errors, 1)
        x_fit = np.linspace(all_stds.min(), all_stds.max(), 100)
        ax.plot(x_fit, m * x_fit + b, 'k-', lw=1.5, alpha=0.7)
    ax.set_xlabel('Posterior Std (per node)')
    ax.set_ylabel('|Error| (per node)')
    ax.set_title(f'(D) Calibration: r = {r_overall:.3f}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'ex11d_dirichlet_variant.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
