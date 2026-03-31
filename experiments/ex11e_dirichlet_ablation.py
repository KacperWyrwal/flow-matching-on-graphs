"""
Experiment 11e: Dirichlet Start Ablation.

Ablates over starting distribution for posterior sampling:
  Arm 1 (uniform):      Dirichlet(alpha, ..., alpha)
  Arm 2 (obs-centered): Dirichlet(alpha * N * obs)

Trains a separate FiLM model per (arm, alpha), evaluates on Ex11 test cases,
and produces a summary table + 4-panel figure showing the accuracy-uncertainty
tradeoff.
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
from scipy.stats import pearsonr

from graph_ot_fm import (
    GraphStructure,
    GeodesicCache,
    total_variation,
)
from graph_ot_fm.ot_solver import compute_ot_coupling
from graph_ot_fm.flow import marginal_distribution_fast, marginal_rate_matrix_fast

from meta_fm import (
    FiLMConditionalGNNRateMatrixPredictor,
    DirectGNNPredictor,
    train_film_conditional,
    train_direct_gnn,
    get_device,
)
from meta_fm.model import rate_matrix_to_edge_index
from meta_fm.sample import sample_posterior_film

from experiments.ex11_combined_generalization import (
    TRAINING_GRAPHS, TEST_GRAPHS, TAU_DIFFS, N_PEAKS_ALL, N_PER_CELL,
    make_multipeak_dist, diffuse, exact_inverse,
    peak_recovery_topk,
)


# ── Reference numbers ────────────────────────────────────────────────────────

REF_FM_OBS = {'tv': 0.054, 'peak': 98.0}
REF_DIRECT = {'tv': 0.038, 'peak': 99.0}


# ── Start distribution sampling ─────────────────────────────────────────────

def sample_start(obs, N, alpha, arm, rng):
    """Sample a starting distribution for posterior sampling."""
    if arm == 'uniform':
        return rng.dirichlet(np.full(N, alpha))
    elif arm == 'obs_centered':
        obs_safe = np.clip(obs, 1e-4, None)
        obs_safe /= obs_safe.sum()
        return rng.dirichlet(alpha * N * obs_safe)
    else:
        raise ValueError(f"Unknown arm: {arm}")


# ── Dataset ──────────────────────────────────────────────────────────────────

class AblationDataset(torch.utils.data.Dataset):
    """Dataset for one (arm, alpha) configuration."""

    def __init__(self, graphs, arm, alpha, n_pairs_per_graph=75,
                 n_samples_per_graph=1000, n_starts_per_pair=5,
                 tau_diff_range=(0.3, 1.5), seed=42):
        rng = np.random.default_rng(seed)
        all_items = []

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
                mu_source, _ = make_multipeak_dist(N, n_peaks, rng)
                tau_val = float(rng.uniform(*tau_diff_range))
                mu_obs = diffuse(mu_source, R, tau_val)

                node_ctx = mu_obs[:, None].astype(np.float32)
                global_ctx = np.array([tau_val], dtype=np.float32)

                for _ in range(n_starts_per_pair):
                    mu_start = sample_start(mu_obs, N, alpha, arm, rng)

                    coupling = compute_ot_coupling(
                        mu_start, mu_source, graph_struct=graph_struct)
                    geo_cache.precompute_for_coupling(coupling)

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_config(model, arm, alpha, device, posterior_k=20,
                    n_ode_steps=100, seed=99):
    """Evaluate one (arm, alpha) config on all Ex11 test cases."""
    rng = np.random.default_rng(seed)
    all_tv, all_peak, all_cal_r, all_div = [], [], [], []
    all_stds, all_errors = [], []

    for name, R in TEST_GRAPHS:
        N = R.shape[0]
        ei = rate_matrix_to_edge_index(R)

        for tau_diff in TAU_DIFFS:
            for n_peaks in N_PEAKS_ALL:
                for _ in range(N_PER_CELL):
                    mu_source, peak_nodes = make_multipeak_dist(N, n_peaks, rng)
                    mu_obs = diffuse(mu_source, R, tau_diff)

                    node_ctx = mu_obs[:, None].astype(np.float32)
                    global_ctx = np.array([tau_diff], dtype=np.float32)

                    mu_starts = np.array([
                        sample_start(mu_obs, N, alpha, arm, rng)
                        for _ in range(posterior_k)])

                    try:
                        fm_samples = sample_posterior_film(
                            model, mu_starts, node_ctx, global_ctx, ei,
                            n_steps=n_ode_steps, device=device)
                    except Exception:
                        fm_samples = np.tile(mu_obs, (posterior_k, 1))

                    fm_mean = fm_samples.mean(axis=0)
                    fm_mean = np.clip(fm_mean, 0, None)
                    fm_mean /= fm_mean.sum() + 1e-15

                    fm_std = fm_samples.std(axis=0)
                    error = np.abs(fm_mean - mu_source)

                    all_tv.append(total_variation(fm_mean, mu_source))
                    all_peak.append(peak_recovery_topk(fm_mean, peak_nodes))

                    if fm_std.std() > 1e-10 and error.std() > 1e-10:
                        r_cal, _ = pearsonr(fm_std, error)
                    else:
                        r_cal = 0.0
                    all_cal_r.append(r_cal)

                    # Diversity
                    div_tvs = []
                    for i in range(min(posterior_k, 10)):
                        for j in range(i + 1, min(posterior_k, 10)):
                            div_tvs.append(
                                total_variation(fm_samples[i], fm_samples[j]))
                    all_div.append(float(np.mean(div_tvs)) if div_tvs else 0.0)

                    all_stds.append(fm_std)
                    all_errors.append(error)

    return {
        'tv': float(np.mean(all_tv)),
        'peak': float(np.mean(all_peak)) * 100,
        'cal_r': float(np.mean(all_cal_r)),
        'diversity': float(np.mean(all_div)),
        'all_stds': np.concatenate(all_stds),
        'all_errors': np.concatenate(all_errors),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ex11e: Dirichlet start ablation')
    parser.add_argument('--mode', type=str, default='quick',
                        choices=['quick', 'full'],
                        help='quick=100ep/lr=5e-3, full=1000ep/lr=5e-4')
    parser.add_argument('--arm', type=str, default=None,
                        choices=['uniform', 'obs_centered'],
                        help='Run a single arm (omit to run all)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Run a single alpha (omit to run all)')
    parser.add_argument('--n-epochs', type=int, default=None,
                        help='Override epoch count')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--posterior-k', type=int, default=20)
    parser.add_argument('--train-direct', action='store_true',
                        help='Train and evaluate a DirectGNN baseline')
    parser.add_argument('--direct-epochs', type=int, default=500)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if args.mode == 'quick':
        n_epochs, lr = 100, 5e-3
        suffix = '100ep_quick'
    else:
        n_epochs, lr = 1000, 5e-4
        suffix = '1000ep'

    # Allow explicit overrides
    if args.n_epochs is not None:
        n_epochs = args.n_epochs
        suffix = f'{n_epochs}ep' if n_epochs != 100 else suffix
    if args.lr is not None:
        lr = args.lr

    device = get_device()
    print(f"Device: {device}", flush=True)
    print(f"\n=== Experiment 11e: Dirichlet Start Ablation "
          f"({n_epochs}ep, lr={lr}) ===\n", flush=True)

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Configurations ───────────────────────────────────────────────────────
    if args.arm is not None and args.alpha is not None:
        # Single config mode
        configs = [(args.arm, args.alpha)]
    else:
        configs = []
        for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
            configs.append(('uniform', alpha))
        for alpha in [0.5, 1.0, 5.0, 10.0, 50.0]:
            configs.append(('obs_centered', alpha))

    # ── Train + evaluate each config ─────────────────────────────────────────
    all_results = {}

    for arm, alpha in configs:
        label = f"{arm}_a{alpha}"
        ckpt_path = os.path.join(
            checkpoint_dir, f'ex11e_{arm}_{alpha}_{suffix}.pt')

        print(f"\n--- {arm} alpha={alpha} ---", flush=True)

        model = FiLMConditionalGNNRateMatrixPredictor(
            node_context_dim=1, global_dim=1,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers)

        if args.resume and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
            print(f"  Loaded {ckpt_path}", flush=True)
        elif args.resume and not os.path.exists(ckpt_path):
            print(f"  No checkpoint found, skipping (use without --resume to train)",
                  flush=True)
            continue
        else:
            print(f"  Building dataset...", flush=True)
            dataset = AblationDataset(
                graphs=TRAINING_GRAPHS, arm=arm, alpha=alpha,
                n_pairs_per_graph=75, n_samples_per_graph=1000,
                n_starts_per_pair=5, seed=42)
            print(f"  {len(dataset)} samples. Training {n_epochs} epochs...",
                  flush=True)

            history = train_film_conditional(
                model, dataset,
                n_epochs=n_epochs, batch_size=args.batch_size,
                lr=lr, device=device,
                loss_weighting='uniform', loss_type='rate_kl',
                ema_decay=0.999)
            print(f"  Final loss: {history['losses'][-1]:.6f}", flush=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved {ckpt_path}", flush=True)

        model.eval()
        print(f"  Evaluating...", flush=True)
        res = evaluate_config(
            model, arm, alpha, device,
            posterior_k=args.posterior_k, seed=99)
        all_results[(arm, alpha)] = res
        print(f"  TV={res['tv']:.4f}, Peak={res['peak']:.0f}%, "
              f"Cal-r={res['cal_r']:.3f}, Div={res['diversity']:.4f}",
              flush=True)

    # ── DirectGNN baseline (optional) ────────────────────────────────────────
    direct_results = None
    if args.train_direct:
        direct_ckpt = os.path.join(
            checkpoint_dir, f'ex11e_direct_{args.direct_epochs}ep'
            f'_h{args.hidden_dim}_l{args.n_layers}.pt')

        direct_model = DirectGNNPredictor(
            context_dim=2,  # [obs(a), tau_broadcast]
            hidden_dim=args.hidden_dim, n_layers=args.n_layers)

        if args.resume and os.path.exists(direct_ckpt):
            direct_model.load_state_dict(
                torch.load(direct_ckpt, map_location='cpu'))
            print(f"\nLoaded DirectGNN from {direct_ckpt}", flush=True)
        else:
            print(f"\n--- Training DirectGNN ({args.direct_epochs} epochs) ---",
                  flush=True)
            direct_rng = np.random.default_rng(123)
            direct_pairs = []
            for name, R in TRAINING_GRAPHS:
                N = R.shape[0]
                ei = rate_matrix_to_edge_index(R)
                for _ in range(75):
                    n_pk = int(direct_rng.integers(1, 4))
                    mu_src, _ = make_multipeak_dist(N, n_pk, direct_rng)
                    tau = float(direct_rng.uniform(0.3, 1.5))
                    mu_obs = diffuse(mu_src, R, tau)
                    ctx = np.stack([mu_obs, np.full(N, tau)],
                                   axis=-1).astype(np.float32)
                    direct_pairs.append((ctx, mu_src.astype(np.float32), ei))

            print(f"  {len(direct_pairs)} training pairs", flush=True)
            train_direct_gnn(
                direct_model, direct_pairs,
                n_epochs=args.direct_epochs, lr=lr,
                device=device, seed=42, ema_decay=0.999,
                checkpoint_path=direct_ckpt)
            torch.save(direct_model.state_dict(), direct_ckpt)
            print(f"  Saved {direct_ckpt}", flush=True)

        # Evaluate DirectGNN
        direct_model.to(device)
        direct_model.eval()
        print("  Evaluating DirectGNN...", flush=True)
        rng_d = np.random.default_rng(99)
        d_tv, d_peak = [], []
        for name, R in TEST_GRAPHS:
            N = R.shape[0]
            ei = rate_matrix_to_edge_index(R)
            for tau_diff in TAU_DIFFS:
                for n_peaks in N_PEAKS_ALL:
                    for _ in range(N_PER_CELL):
                        mu_source, peak_nodes = make_multipeak_dist(
                            N, n_peaks, rng_d)
                        mu_obs = diffuse(mu_source, R, tau_diff)
                        ctx = np.stack([mu_obs, np.full(N, tau_diff)],
                                       axis=-1).astype(np.float32)
                        ctx_t = torch.tensor(ctx, dtype=torch.float32,
                                             device=device)
                        ei_d = ei.to(device)
                        with torch.no_grad():
                            mu_pred = direct_model(ctx_t, ei_d).cpu().numpy()
                        d_tv.append(total_variation(mu_pred, mu_source))
                        d_peak.append(peak_recovery_topk(mu_pred, peak_nodes))
        direct_results = {
            'tv': float(np.mean(d_tv)),
            'peak': float(np.mean(d_peak)) * 100,
        }
        print(f"  DirectGNN: TV={direct_results['tv']:.4f}, "
              f"Peak={direct_results['peak']:.0f}%", flush=True)

    # ── Console table ────────────────────────────────────────────────────────
    evaluated = [(arm, alpha) for arm, alpha in configs if (arm, alpha) in all_results]

    print(f"\n{'Arm':15s} {'alpha':>6s}  {'TV':>7s}  {'Peak%':>6s}  "
          f"{'Cal-r':>6s}  {'Diversity':>9s}", flush=True)
    print("-" * 60, flush=True)
    for arm, alpha in evaluated:
        r = all_results[(arm, alpha)]
        arm_label = 'Uniform' if arm == 'uniform' else 'Obs-centered'
        print(f"{arm_label:15s} {alpha:6.1f}  {r['tv']:7.4f}  {r['peak']:5.0f}%  "
              f"{r['cal_r']:6.3f}  {r['diversity']:9.4f}", flush=True)
    print("-" * 60, flush=True)
    print(f"{'FM obs-start':15s} {'-':>6s}  {REF_FM_OBS['tv']:7.3f}  "
          f"{REF_FM_OBS['peak']:5.0f}%  {'  -':>6s}  {'    -':>9s}  (Ex11 ref)",
          flush=True)
    ref_d = direct_results if direct_results else REF_DIRECT
    d_label = f"({args.direct_epochs}ep)" if direct_results else "(Ex11 ref)"
    print(f"{'DirectGNN':15s} {'-':>6s}  {ref_d['tv']:7.3f}  "
          f"{ref_d['peak']:5.0f}%  {'  -':>6s}  {'    -':>9s}  {d_label}",
          flush=True)

    if not evaluated:
        print("\nNo configs evaluated — nothing to plot.", flush=True)
        return

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f'Ex11e: Dirichlet Start Ablation ({n_epochs}ep)', fontsize=12)

    uniform_alphas = [a for arm, a in evaluated if arm == 'uniform']
    obs_alphas = [a for arm, a in evaluated if arm == 'obs_centered']
    uniform_res = [all_results[('uniform', a)] for a in uniform_alphas]
    obs_res = [all_results[('obs_centered', a)] for a in obs_alphas]

    # Panel A: TV vs alpha
    ax = axes[0, 0]
    if uniform_res:
        ax.plot(uniform_alphas, [r['tv'] for r in uniform_res],
                'o-', color='tab:blue', lw=2, label='Uniform Dirichlet')
    if obs_res:
        ax.plot(obs_alphas, [r['tv'] for r in obs_res],
                's-', color='tab:orange', lw=2, label='Obs-centered Dirichlet')
    ax.axhline(y=REF_FM_OBS['tv'], color='steelblue', ls='--', alpha=0.7,
               label=f"FM obs-start ({REF_FM_OBS['tv']:.3f})")
    ax.axhline(y=ref_d['tv'], color='tab:red', ls='--', alpha=0.7,
               label=f"DirectGNN ({ref_d['tv']:.3f})")
    if len(uniform_alphas + obs_alphas) > 1:
        ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_ylabel('Mean TV')
    ax.set_title('(A) Accuracy vs α')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel B: Calibration r vs alpha
    ax = axes[0, 1]
    if uniform_res:
        ax.plot(uniform_alphas, [r['cal_r'] for r in uniform_res],
                'o-', color='tab:blue', lw=2, label='Uniform')
    if obs_res:
        ax.plot(obs_alphas, [r['cal_r'] for r in obs_res],
                's-', color='tab:orange', lw=2, label='Obs-centered')
    if len(uniform_alphas + obs_alphas) > 1:
        ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_ylabel('Calibration r')
    ax.set_title('(B) Calibration vs α')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: TV vs calibration r (Pareto frontier)
    ax = axes[1, 0]
    for arm, alpha in evaluated:
        r = all_results[(arm, alpha)]
        color = 'tab:blue' if arm == 'uniform' else 'tab:orange'
        marker = 'o' if arm == 'uniform' else 's'
        ax.scatter(r['cal_r'], r['tv'], c=color, marker=marker, s=60, zorder=5)
        ax.annotate(f'α={alpha}', (r['cal_r'], r['tv']),
                    fontsize=6, textcoords='offset points', xytext=(4, 4))
    ax.axhline(y=REF_FM_OBS['tv'], color='steelblue', ls='--', alpha=0.5)
    ax.axhline(y=ref_d['tv'], color='tab:red', ls='--', alpha=0.5)
    ax.scatter([], [], c='tab:blue', marker='o', label='Uniform')
    ax.scatter([], [], c='tab:orange', marker='s', label='Obs-centered')
    ax.set_xlabel('Calibration r')
    ax.set_ylabel('Mean TV')
    ax.set_title('(C) Accuracy-Uncertainty Tradeoff')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D: Diversity vs alpha
    ax = axes[1, 1]
    if uniform_res:
        ax.plot(uniform_alphas, [r['diversity'] for r in uniform_res],
                'o-', color='tab:blue', lw=2, label='Uniform')
    if obs_res:
        ax.plot(obs_alphas, [r['diversity'] for r in obs_res],
                's-', color='tab:orange', lw=2, label='Obs-centered')
    if len(uniform_alphas + obs_alphas) > 1:
        ax.set_xscale('log')
    ax.set_xlabel('α')
    ax.set_ylabel('Posterior Diversity (mean pairwise TV)')
    ax.set_title('(D) Diversity vs α')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'ex11e_dirichlet_ablation.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out_path}", flush=True)
    print("Done.", flush=True)


if __name__ == '__main__':
    main()
