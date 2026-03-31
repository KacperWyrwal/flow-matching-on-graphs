"""
Experiment 21: Kawasaki Ising on 2D Lattice.

Learns to sample from the Ising model on an L×L torus with fixed magnetization
using Kawasaki dynamics. Demonstrates constraint enforcement by construction,
locality, and phase transition from disorder to ordered domains.
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
from scipy.stats import ks_2samp
from scipy.ndimage import label as ndimage_label

from config_fm import ConfigurationRatePredictor, train_configuration_fm
from config_fm.sample import generate_samples
from config_fm.spaces import KawasakiSpace, DFMSpace
from config_fm.spaces.kawasaki_mcmc import (
    kawasaki_mcmc, generate_kawasaki_pool, ising_energy_lattice,
)


# ── Evaluation helpers ───────────────────────────────────────────────────────

def compute_validity(samples, k):
    return float(np.mean(samples.sum(axis=1) == k))


def compute_energy_stats(samples, L, J=1.0):
    energies = np.array([ising_energy_lattice(x, L, J) for x in samples])
    return float(energies.mean()), float(energies.std()), energies


def compute_pairwise_correlations(samples):
    return (samples.T @ samples) / len(samples)


def compute_spatial_correlation(samples, L, max_r=None):
    """Spatial correlation C(r) = <sigma_i sigma_{i+r}> - <sigma>^2 (vectorized).

    Uses np.roll for fast batch shifting. Returns: (rs, C_r) arrays.
    """
    if max_r is None:
        max_r = L
    grids = samples.reshape(-1, L, L)
    mean_sigma = grids.mean()

    C_r = np.zeros(max_r + 1)
    counts = np.zeros(max_r + 1)

    for dy in range(-(L // 2), L // 2 + 1):
        for dx in range(-(L // 2), L // 2 + 1):
            r = min(abs(dx), L - abs(dx)) + min(abs(dy), L - abs(dy))
            if r > max_r:
                continue
            shifted = np.roll(np.roll(grids, dy, axis=1), dx, axis=2)
            C_r[r] += (grids * shifted).mean() - mean_sigma ** 2
            counts[r] += 1

    valid = counts > 0
    C_r[valid] /= counts[valid]
    rs = np.arange(max_r + 1)
    return rs[valid], C_r[valid]


def compute_domain_sizes(samples, L):
    """Compute connected component sizes across samples.

    Uses 4-connectivity on the L×L grid. Returns list of all component sizes.
    """
    all_sizes = []
    for x in samples:
        grid = x.reshape(L, L)
        # Label connected components of 1s
        labeled, n_comp = ndimage_label(grid)
        for c in range(1, n_comp + 1):
            all_sizes.append(int((labeled == c).sum()))
    return all_sizes


# ── MultiBetaSampler (reuse pattern from Ex20) ──────────────────────────────

class MultiBetaSampler:
    """Wraps a ConfigurationSpace to sample from multiple betas.

    sample_target returns (config, {'beta': beta}) so the training
    loop passes the same beta to global_features.
    """
    def __init__(self, space, betas, pools):
        self._space = space
        self._betas = betas
        self._pools = pools

    def __getattr__(self, name):
        return getattr(self._space, name)

    def sample_target(self, rng, **kwargs):
        beta = float(rng.choice(self._betas))
        config = self._space.sample_target(
            rng, beta=beta, mcmc_pool=self._pools[beta])
        return config, {'beta': beta}

    def global_features(self, t=0.0, **kwargs):
        beta = kwargs.get('beta', 1.0)
        return self._space.global_features(t=t, beta=beta)


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_all(fm_model, dfm_model, kawasaki_space, dfm_space,
                 L, k, betas, device,
                 mcmc_pools=None, n_samples=2000, n_steps=200,
                 mcmc_budgets=(1000, 5000, 10000)):
    """Evaluate all methods across betas."""
    rng = np.random.default_rng(42)
    results = []
    J = kawasaki_space.J_coupling

    for beta in betas:
        print(f"\n  beta={beta}:", flush=True)

        # Ground truth
        if mcmc_pools is not None and beta in mcmc_pools:
            gt_samples = mcmc_pools[beta]
        else:
            gt_samples = np.array([
                kawasaki_mcmc(kawasaki_space, beta, kawasaki_space.N * 50, rng)
                for _ in range(1000)
            ])
        gt_e_mean, gt_e_std, gt_energies = compute_energy_stats(gt_samples, L, J)

        def _metrics(samples, method):
            validity = compute_validity(samples, k)
            valid = samples[samples.sum(axis=1) == k]
            e_mean, e_std, energies = compute_energy_stats(samples, L, J)
            e_bias = abs(e_mean - gt_e_mean)
            if len(valid) > 1:
                valid_e = np.array([ising_energy_lattice(x, L, J)
                                    for x in valid])
                ks_stat, _ = ks_2samp(valid_e, gt_energies)
            else:
                ks_stat = 1.0
            corr_gt = compute_pairwise_correlations(gt_samples)
            corr_m = (compute_pairwise_correlations(valid)
                      if len(valid) > 0 else np.zeros_like(corr_gt))
            corr_rmse = float(np.sqrt(((corr_m - corr_gt) ** 2).mean()))
            return {
                'method': method, 'beta': beta, 'validity': validity,
                'energy_mean': e_mean, 'energy_std': e_std,
                'energy_bias': e_bias, 'energy_ks': ks_stat,
                'gt_energy_mean': gt_e_mean, 'gt_energy_std': gt_e_std,
                'corr_rmse': corr_rmse, 'energies': energies,
                'samples': valid,
            }

        # FM
        print(f"    FM...", flush=True)
        fm_samples = generate_samples(
            fm_model, kawasaki_space, n_samples, n_steps, device,
            seed=100, beta=beta)
        results.append(_metrics(fm_samples, 'FM'))

        # DFM
        print(f"    DFM...", flush=True)
        dfm_samples = generate_samples(
            dfm_model, dfm_space, n_samples, n_steps, device,
            seed=200, beta=beta)
        dfm_m = _metrics(dfm_samples, 'DFM')
        results.append(dfm_m)

        # DFM+reject
        dfm_valid = dfm_samples[dfm_samples.sum(axis=1) == k]
        if len(dfm_valid) > 0:
            dfmr = _metrics(dfm_valid, 'DFM+reject')
        else:
            dfmr = dfm_m.copy()
            dfmr['method'] = 'DFM+reject'
        dfmr['eff_rate'] = dfm_m['validity']
        dfmr['validity'] = 1.0 if len(dfm_valid) > 0 else 0.0
        results.append(dfmr)

        # MCMC baselines
        for budget in mcmc_budgets:
            print(f"    MCMC-{budget}...", flush=True)
            n_mc = min(500, n_samples)
            mc_samps = np.array([
                kawasaki_mcmc(kawasaki_space, beta, budget, rng)
                for _ in range(n_mc)
            ])
            results.append(_metrics(mc_samps, f'MCMC-{budget}'))

    return results


# ── Printing ─────────────────────────────────────────────────────────────────

def print_results(results):
    print(f"\n{'Method':15s} {'beta':>5s} {'E bias':>7s} {'E KS':>6s} "
          f"{'Corr RMSE':>10s} {'Valid%':>7s} {'Eff.rate':>9s}", flush=True)
    print("-" * 65, flush=True)
    for r in results:
        eff = (f"{r['eff_rate']*100:.1f}%" if 'eff_rate' in r
               else "100%" if r['validity'] == 1.0
               else f"{r['validity']*100:.1f}%")
        print(f"{r['method']:15s} {r['beta']:5.2f} {r['energy_bias']:7.2f} "
              f"{r['energy_ks']:6.3f} {r['corr_rmse']:10.4f} "
              f"{r['validity']*100:6.1f}% {eff:>9s}", flush=True)

    # Energy table
    print(f"\n{'Method':15s} {'beta':>5s} {'mean_E':>10s} {'tgt_E':>10s} "
          f"{'std_E':>10s} {'tgt_std':>10s}", flush=True)
    print("-" * 65, flush=True)
    for r in results:
        print(f"{r['method']:15s} {r['beta']:5.2f} {r['energy_mean']:10.1f} "
              f"{r['gt_energy_mean']:10.1f} {r['energy_std']:10.2f} "
              f"{r['gt_energy_std']:10.2f}", flush=True)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(results, L, k, betas, kawasaki_space, mcmc_pools,
                 out_path, device):
    """2×3 results figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f'Ex21: Kawasaki Ising on {L}×{L} Lattice', fontsize=13)
    J = kawasaki_space.J_coupling
    beta_hard = max(betas)
    rng = np.random.default_rng(42)

    # Panel A: Sample gallery at high beta
    ax = axes[0, 0]
    ax.set_title(f'(A) Samples at β={beta_hard}', fontsize=10)
    ax.axis('off')
    n_cols = 6
    rows_data = []
    for method in ['FM', 'DFM+reject', f'MCMC-{10000}']:
        for r in results:
            if r['method'] == method and r['beta'] == beta_hard:
                samps = r.get('samples', np.array([]))
                if len(samps) >= n_cols:
                    rows_data.append((method, samps[:n_cols]))
                break
    # True samples
    if mcmc_pools and beta_hard in mcmc_pools:
        true_samps = mcmc_pools[beta_hard][:n_cols]
        rows_data.append(('True', true_samps))

    if rows_data:
        inner = ax.inset_axes([0, 0, 1, 1])
        inner.axis('off')
        nr = len(rows_data)
        for ri, (label, samps) in enumerate(rows_data):
            for ci in range(min(n_cols, len(samps))):
                left = ci / (n_cols + 0.5)
                bottom = 1 - (ri + 1) / (nr + 0.3)
                w = 0.85 / (n_cols + 0.5)
                h_ax = 0.85 / (nr + 0.3)
                sub = inner.inset_axes([left + 0.02, bottom, w, h_ax])
                grid = samps[ci].reshape(L, L)
                sub.imshow(grid, cmap='binary', vmin=0, vmax=1,
                           interpolation='nearest')
                sub.set_xticks([])
                sub.set_yticks([])
                if ci == 0:
                    sub.set_ylabel(label, fontsize=7, rotation=0,
                                   labelpad=30, va='center')

    # Panel B: Phase transition montage
    ax = axes[0, 1]
    ax.set_title('(B) Phase Transition (FM samples)', fontsize=10)
    ax.axis('off')
    n_per = 4
    inner_b = ax.inset_axes([0, 0, 1, 1])
    inner_b.axis('off')
    for bi, beta in enumerate(betas):
        for r in results:
            if r['method'] == 'FM' and r['beta'] == beta:
                samps = r.get('samples', np.array([]))
                for ci in range(min(n_per, len(samps))):
                    left = ci / (n_per + 0.5)
                    bottom = 1 - (bi + 1) / (len(betas) + 0.3)
                    w = 0.85 / (n_per + 0.5)
                    h_ax = 0.85 / (len(betas) + 0.3)
                    sub = inner_b.inset_axes([left + 0.02, bottom, w, h_ax])
                    if ci < len(samps):
                        sub.imshow(samps[ci].reshape(L, L), cmap='binary',
                                   vmin=0, vmax=1, interpolation='nearest')
                    sub.set_xticks([])
                    sub.set_yticks([])
                    if ci == 0:
                        sub.set_ylabel(f'β={beta}', fontsize=7, rotation=0,
                                       labelpad=25, va='center')
                break

    # Panel C: Energy vs beta
    ax = axes[0, 2]
    for method in ['FM', 'DFM+reject', 'MCMC-10000']:
        es, bs = [], []
        for r in results:
            if r['method'] == method:
                es.append(r['energy_mean'])
                bs.append(r['beta'])
        if es:
            ax.plot(bs, es, 'o-', label=method, lw=1.5)
    # GT
    gt_es = []
    for beta in betas:
        for r in results:
            if r['method'] == 'FM' and r['beta'] == beta:
                gt_es.append(r['gt_energy_mean'])
                break
    if gt_es:
        ax.plot(betas, gt_es, 'k--', lw=2, label='Target', alpha=0.7)
    ax.set_xlabel('β')
    ax.set_ylabel('Mean Energy')
    ax.set_title('(C) Energy vs β')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel D: Spatial correlation C(r)
    ax = axes[1, 0]
    beta_corr = 0.8 if 0.8 in betas else betas[-1]
    for method in ['FM', 'MCMC-10000']:
        for r in results:
            if r['method'] == method and r['beta'] == beta_corr:
                samps = r.get('samples', np.array([]))
                if len(samps) > 10:
                    rs_arr, cr = compute_spatial_correlation(
                        samps[:500], L, max_r=L // 2)
                    ax.plot(rs_arr, cr, 'o-', label=method, markersize=3)
                break
    # GT
    if mcmc_pools and beta_corr in mcmc_pools:
        rs_arr, cr_gt = compute_spatial_correlation(
            mcmc_pools[beta_corr][:500], L, max_r=L // 2)
        ax.plot(rs_arr, cr_gt, 'k--', label='Target', lw=2, alpha=0.7)
    ax.set_xlabel('Distance r')
    ax.set_ylabel('C(r)')
    ax.set_title(f'(D) Spatial Correlation (β={beta_corr})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel E: Domain size distribution
    ax = axes[1, 1]
    for method, color in [('FM', 'tab:blue'), ('MCMC-10000', 'tab:green')]:
        for r in results:
            if r['method'] == method and r['beta'] == beta_hard:
                samps = r.get('samples', np.array([]))
                if len(samps) > 10:
                    sizes = compute_domain_sizes(samps[:200], L)
                    ax.hist(sizes, bins=30, alpha=0.5, label=method,
                            color=color, density=True)
                break
    if mcmc_pools and beta_hard in mcmc_pools:
        gt_sizes = compute_domain_sizes(mcmc_pools[beta_hard][:200], L)
        ax.hist(gt_sizes, bins=30, alpha=0.4, label='Target',
                color='gray', density=True, histtype='step', lw=2)
    ax.set_xlabel('Domain Size')
    ax.set_ylabel('Density')
    ax.set_title(f'(E) Domain Sizes (β={beta_hard})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel F: Validity by L
    ax = axes[1, 2]
    from math import comb
    Ls = [6, 8, 10, 14, 20]
    validity_theory = [comb(L_val ** 2, L_val ** 2 // 2) / (2 ** (L_val ** 2))
                       for L_val in Ls]
    ax.semilogy(Ls, validity_theory, 'o-', color='tab:red',
                label='DFM validity (theory)', lw=2)
    ax.axhline(1.0, color='tab:blue', ls='--', lw=2, label='FM (100%)')
    ax.set_xlabel('L')
    ax.set_ylabel('Validity Rate')
    ax.set_title('(F) Validity vs Lattice Size')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out_path}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ex21: Kawasaki Ising on 2D Lattice')
    parser.add_argument('--L', type=int, default=20)
    parser.add_argument('--k', type=int, default=None,
                        help='Fixed magnetization (default: N/2)')
    parser.add_argument('--J-coupling', type=float, default=1.0)
    parser.add_argument('--betas', type=float, nargs='+',
                        default=[0.2, 0.44, 0.8, 1.5])
    parser.add_argument('--n-epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--mcmc-pool-size', type=int, default=10000)
    parser.add_argument('--mcmc-chain-length', type=int, default=100000)
    parser.add_argument('--n-eval-samples', type=int, default=2000)
    parser.add_argument('--n-gen-steps', type=int, default=200)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    L = args.L
    N = L * L
    k = args.k if args.k is not None else N // 2
    betas = args.betas

    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Device: {device}", flush=True)

    print(f"\n=== Experiment 21: Kawasaki Ising on {L}×{L} Lattice ===",
          flush=True)
    print(f"N={N}, k={k}, J={args.J_coupling}", flush=True)
    print(f"β_c ≈ {np.log(1 + np.sqrt(2)) / 2:.4f}", flush=True)
    print(f"Betas: {betas}", flush=True)

    # ── Spaces ────────────────────────────────────────────────────────────────
    kaw_space = KawasakiSpace(L, k, args.J_coupling)
    dfm_space_inst = DFMSpace(N, np.zeros((N, N)), np.zeros(N))
    # DFM uses lattice edges for message passing
    dfm_space_inst._edge_index = kaw_space.position_graph_edges()
    dfm_space_inst._edge_features = kaw_space.position_edge_features()
    # DFM node_features returns (N, 2) since it stacks [config, h]
    _dfm_node_feat_dim = 2

    # ── MCMC pools ────────────────────────────────────────────────────────────
    print("\nGenerating MCMC pools...", flush=True)
    mcmc_pools = {}
    for beta in betas:
        print(f"  β={beta}: {args.mcmc_pool_size} samples...", flush=True)
        mcmc_pools[beta] = generate_kawasaki_pool(
            kaw_space, beta, args.mcmc_pool_size,
            args.mcmc_chain_length,
            seed=args.seed + int(beta * 1000))
        _, _, energies = compute_energy_stats(mcmc_pools[beta], L,
                                              args.J_coupling)
        print(f"    Energy: mean={energies.mean():.1f}, "
              f"std={energies.std():.1f}", flush=True)

    # ── Checkpoints ───────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    fm_ckpt = os.path.join(ckpt_dir, f'ex21_fm_L{L}.pt')
    dfm_ckpt = os.path.join(ckpt_dir, f'ex21_dfm_L{L}.pt')

    # ── Train FM ──────────────────────────────────────────────────────────────
    print("\n--- Training FM (Kawasaki, k=2) ---", flush=True)
    fm_model = ConfigurationRatePredictor(
        node_feature_dim=1, edge_feature_dim=1, global_dim=2,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        transition_order=2)

    if args.resume and os.path.exists(fm_ckpt):
        fm_model.load_state_dict(torch.load(fm_ckpt, map_location='cpu'))
        print(f"  Loaded from {fm_ckpt}", flush=True)
    else:
        multi = MultiBetaSampler(kaw_space, betas, mcmc_pools)
        result = train_configuration_fm(
            fm_model, multi, {},
            n_epochs=args.n_epochs, batch_size=args.batch_size,
            lr=args.lr, device=device, seed=args.seed)
        torch.save(fm_model.state_dict(), fm_ckpt)
        print(f"  Saved to {fm_ckpt}", flush=True)

    fm_model.to(device)

    # ── Train DFM ─────────────────────────────────────────────────────────────
    print("\n--- Training DFM (unconstrained, k=1) ---", flush=True)
    dfm_model = ConfigurationRatePredictor(
        node_feature_dim=_dfm_node_feat_dim, edge_feature_dim=1, global_dim=2,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        transition_order=1)

    if args.resume and os.path.exists(dfm_ckpt):
        dfm_model.load_state_dict(torch.load(dfm_ckpt, map_location='cpu'))
        print(f"  Loaded from {dfm_ckpt}", flush=True)
    else:
        # DFM uses lattice edges but unconstrained bit flips
        multi_dfm = MultiBetaSampler(dfm_space_inst, betas, mcmc_pools)
        result = train_configuration_fm(
            dfm_model, multi_dfm, {},
            n_epochs=args.n_epochs, batch_size=args.batch_size,
            lr=args.lr, device=device, seed=args.seed + 1)
        torch.save(dfm_model.state_dict(), dfm_ckpt)
        print(f"  Saved to {dfm_ckpt}", flush=True)

    dfm_model.to(device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n--- Evaluation ---", flush=True)
    results = evaluate_all(
        fm_model, dfm_model, kaw_space, dfm_space_inst,
        L, k, betas, device,
        mcmc_pools=mcmc_pools,
        n_samples=args.n_eval_samples, n_steps=args.n_gen_steps)

    print_results(results)

    # ── Plot ──────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_results(results, L, k, betas, kaw_space, mcmc_pools,
                 os.path.join(out_dir, 'ex21_kawasaki_ising.png'), device)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
