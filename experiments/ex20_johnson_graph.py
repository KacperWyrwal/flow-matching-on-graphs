"""
Experiment 20: Flow Matching on the Johnson Graph J(n,k).

Demonstrates flow matching on a combinatorially large configuration graph
where the constraint (fixed Hamming weight) is enforced by construction.
Learns to sample from an energy-based distribution on binary strings with
exactly k ones out of n positions.
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

from johnson_fm.energy import (
    ising_energy, mcmc_kawasaki,
    generate_mcmc_pool, compute_exact_boltzmann,
)

from config_fm import (
    ConfigurationRatePredictor,
    train_configuration_fm,
    generate_samples as generate_config_samples,
)
from config_fm.spaces import JohnsonSpace, DFMSpace


# ── Generation ───────────────────────────────────────────────────────────────

# ── Evaluation ───────────────────────────────────────────────────────────────

def compute_validity(samples, k):
    """Fraction of samples with exactly k ones."""
    hw = samples.sum(axis=1)
    return float(np.mean(hw == k))


def compute_energy_stats(samples, J, h):
    """Energy mean, std, and per-sample energies."""
    energies = np.array([ising_energy(x, J, h) for x in samples])
    return float(energies.mean()), float(energies.std()), energies


def compute_pairwise_correlations(samples):
    """Compute <x_i x_j> over samples. Returns (n, n) matrix."""
    return (samples.T @ samples) / len(samples)


def compute_tv_exact(samples, k, configs, probs):
    """TV distance between empirical and exact distribution (small n only).

    configs: (M, n), probs: (M,) exact Boltzmann.
    """
    # Build config -> index mapping
    M = len(configs)
    config_to_idx = {}
    for idx in range(M):
        key = tuple(configs[idx].astype(int))
        config_to_idx[key] = idx

    # Empirical histogram
    empirical = np.zeros(M)
    n_valid = 0
    for x in samples:
        key = tuple(x.astype(int))
        if key in config_to_idx:
            empirical[config_to_idx[key]] += 1
            n_valid += 1

    if n_valid > 0:
        empirical /= n_valid

    return 0.5 * np.abs(empirical - probs).sum()


def evaluate_all(fm_model, dfm_model, J, h, n, k, betas, device,
                 johnson_space=None, dfm_space=None,
                 mcmc_pools=None, n_samples=10000, n_steps=100,
                 mcmc_budgets=(100, 500, 1000, 5000),
                 configs_exact=None, probs_exact=None):
    """Run full evaluation for all methods and betas."""
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(42)
    results = []

    # Normalize probs_exact keys to float for consistent lookup
    probs_exact_f = None
    if probs_exact is not None:
        probs_exact_f = {float(round(k, 6)): v for k, v in probs_exact.items()}

    for beta in betas:
        beta_key = float(round(beta, 6))
        print(f"\n  beta={beta}:", flush=True)
        energy_fn = lambda x, _J=J, _h=h: ising_energy(x, _J, _h)

        # Ground truth: reuse MCMC pool if available
        if mcmc_pools is not None and beta in mcmc_pools:
            print(f"    Using MCMC pool for ground truth...", flush=True)
            gt_samples = mcmc_pools[beta]
        else:
            print(f"    Ground truth MCMC...", flush=True)
            gt_samples = np.array([
                mcmc_kawasaki(energy_fn, n, k, beta, 5000, rng)
                for _ in range(1000)
            ])
        corr_gt = compute_pairwise_correlations(gt_samples)
        gt_energies = np.array([ising_energy(x, J, h) for x in gt_samples])
        gt_mean_E = float(gt_energies.mean())
        gt_std_E = float(gt_energies.std())
        print(f"    GT energy: mean={gt_mean_E:.2f}, std={gt_std_E:.2f}",
              flush=True)

        def _compute_metrics(samples, label, compute_tv=True):
            """Compute all metrics for a set of samples."""
            validity = compute_validity(samples, k)
            valid = samples[samples.sum(axis=1) == k]

            e_mean, e_std, energies = compute_energy_stats(samples, J, h)
            energy_bias = abs(e_mean - gt_mean_E)

            # KS test on valid samples only
            if len(valid) > 1:
                valid_energies = np.array([ising_energy(x, J, h)
                                           for x in valid])
                ks_stat, _ = ks_2samp(valid_energies, gt_energies)
            else:
                valid_energies = energies
                ks_stat = 1.0

            corr = (compute_pairwise_correlations(valid)
                    if len(valid) > 0 else np.zeros((n, n)))
            corr_rmse = float(np.sqrt(((corr - corr_gt) ** 2).mean()))

            tv = None
            if (compute_tv and configs_exact is not None
                    and probs_exact_f is not None
                    and beta_key in probs_exact_f and len(valid) > 0):
                tv = compute_tv_exact(valid, k,
                                      configs_exact, probs_exact_f[beta_key])

            return {
                'validity': validity, 'tv': tv,
                'energy_mean': e_mean, 'energy_std': e_std,
                'energy_bias': energy_bias, 'energy_ks': ks_stat,
                'gt_energy_mean': gt_mean_E, 'gt_energy_std': gt_std_E,
                'corr_rmse': corr_rmse,
                'energies': energies, 'samples': valid,
            }

        # FM
        print(f"    FM generation...", flush=True)
        fm_samples = generate_config_samples(
            fm_model, johnson_space, n_samples, n_steps, device,
            seed=100, beta=beta)
        fm_m = _compute_metrics(fm_samples, 'FM')
        results.append({'method': 'FM', 'beta': beta, **fm_m})

        # DFM
        print(f"    DFM generation...", flush=True)
        dfm_samples = generate_config_samples(
            dfm_model, dfm_space, n_samples, n_steps, device,
            seed=200, beta=beta)
        dfm_m = _compute_metrics(dfm_samples, 'DFM')
        results.append({'method': 'DFM', 'beta': beta, **dfm_m})

        # DFM + rejection (same samples, just filtered)
        dfm_valid = dfm_samples[dfm_samples.sum(axis=1) == k]
        if len(dfm_valid) > 0:
            dfmr_m = _compute_metrics(dfm_valid, 'DFM+reject')
        else:
            dfmr_m = dfm_m.copy()
        dfmr_m['eff_rate'] = dfm_m['validity']
        dfmr_m['validity'] = 1.0 if len(dfm_valid) > 0 else 0.0
        results.append({'method': 'DFM+reject', 'beta': beta, **dfmr_m})

        # MCMC at various budgets
        for budget in mcmc_budgets:
            print(f"    MCMC-{budget}...", flush=True)
            n_mc = min(1000, n_samples)
            mcmc_samps = np.array([
                mcmc_kawasaki(energy_fn, n, k, beta, budget, rng)
                for _ in range(n_mc)
            ])
            mc_m = _compute_metrics(mcmc_samps, f'MCMC-{budget}')
            results.append({'method': f'MCMC-{budget}', 'beta': beta, **mc_m})

    return results


# ── Printing ─────────────────────────────────────────────────────────────────

def print_results(results):
    """Print formatted results table."""
    print(f"\n{'Method':15s} {'beta':>5s} {'TV':>8s} {'E bias':>7s} "
          f"{'E KS':>6s} {'Corr RMSE':>10s} {'Valid%':>7s} {'Eff.rate':>9s}",
          flush=True)
    print("-" * 75, flush=True)

    for r in results:
        tv_str = f"{r['tv']:.4f}" if r.get('tv') is not None else "   -"
        eff_str = (f"{r['eff_rate']*100:.1f}%" if 'eff_rate' in r
                   else "100%" if r['validity'] == 1.0
                   else f"{r['validity']*100:.1f}%")
        print(f"{r['method']:15s} {r['beta']:5.1f} {tv_str:>8s} "
              f"{r.get('energy_bias', 0):7.3f} {r.get('energy_ks', 0):6.3f} "
              f"{r['corr_rmse']:10.4f} {r['validity']*100:6.1f}% "
              f"{eff_str:>9s}", flush=True)

    # Detailed energy table
    print(f"\n{'Method':15s} {'beta':>5s} {'mean_E(samp)':>13s} "
          f"{'mean_E(tgt)':>12s} {'std_E(samp)':>12s} {'std_E(tgt)':>11s}",
          flush=True)
    print("-" * 75, flush=True)
    for r in results:
        print(f"{r['method']:15s} {r['beta']:5.1f} "
              f"{r['energy_mean']:13.3f} {r.get('gt_energy_mean', 0):12.3f} "
              f"{r['energy_std']:12.3f} {r.get('gt_energy_std', 0):11.3f}",
              flush=True)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(results, J, h, n, k, betas, out_path,
                 configs_exact=None, probs_exact=None,
                 mcmc_pools=None):
    """2x3 results figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Ex20: Flow Matching on Johnson Graph J({n},{k})', fontsize=12)

    # Panel A: Energy histograms at hardest beta
    ax = axes[0, 0]
    beta_hard = max(betas)
    for r in results:
        if r['beta'] == beta_hard and 'energies' in r and len(r.get('energies', [])) > 0:
            if r['method'] in ['FM', 'DFM', f'MCMC-5000']:
                ax.hist(r['energies'], bins=40, alpha=0.4, density=True,
                        label=r['method'])
    if configs_exact is not None and probs_exact is not None and beta_hard in probs_exact:
        energies_exact = np.array([ising_energy(c, J, h) for c in configs_exact])
        ax.hist(energies_exact, bins=40, weights=probs_exact[beta_hard],
                alpha=0.4, density=True, label='Exact', color='black',
                histtype='step', linewidth=2)
    ax.set_xlabel('Energy')
    ax.set_ylabel('Density')
    ax.set_title(f'(A) Energy Histograms (β={beta_hard})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel B: Pairwise correlation scatter
    ax = axes[0, 1]
    if mcmc_pools is not None and beta_hard in mcmc_pools:
        gt_samples_plot = mcmc_pools[beta_hard]
    else:
        rng = np.random.default_rng(42)
        gt_samples_plot = np.array([
            mcmc_kawasaki(lambda x: ising_energy(x, J, h), n, k,
                          beta_hard, 5000, rng)
            for _ in range(1000)
        ])
    corr_gt = compute_pairwise_correlations(gt_samples_plot)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    gt_flat = corr_gt[mask]

    for r in results:
        if r['beta'] == beta_hard and r['method'] in ['FM', 'DFM']:
            samps = r.get('samples', None)
            if samps is not None and len(samps) > 0:
                corr_m = compute_pairwise_correlations(samps)
                m_flat = corr_m[mask]
                ax.scatter(gt_flat, m_flat, s=8, alpha=0.4,
                           label=r['method'])
    lims = [min(gt_flat.min(), 0), max(gt_flat.max(), 0.3)]
    ax.plot(lims, lims, 'k--', alpha=0.5, lw=1)
    ax.set_xlabel('True ⟨xᵢxⱼ⟩')
    ax.set_ylabel('Model ⟨xᵢxⱼ⟩')
    ax.set_title(f'(B) Pairwise Correlations (β={beta_hard})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: TV vs beta (exact only)
    ax = axes[0, 2]
    if probs_exact is not None:
        for method in ['FM', 'DFM+reject']:
            tvs = []
            bs = []
            for r in results:
                if r['method'] == method and r.get('tv') is not None:
                    tvs.append(r['tv'])
                    bs.append(r['beta'])
            if tvs:
                ax.plot(bs, tvs, 'o-', label=method, lw=2)
        for budget in [1000, 5000]:
            tvs = []
            bs = []
            for r in results:
                if r['method'] == f'MCMC-{budget}' and r.get('tv') is not None:
                    tvs.append(r['tv'])
                    bs.append(r['beta'])
            if tvs:
                ax.plot(bs, tvs, 's--', label=f'MCMC-{budget}', lw=1.5)
    ax.set_xlabel('β')
    ax.set_ylabel('TV Distance')
    ax.set_title('(C) TV vs β')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel D: Validity rate vs n (DFM)
    ax = axes[1, 0]
    from math import comb
    ns_plot = [12, 16, 20, 24, 28]
    validity_theory = [comb(nn, nn // 2) / (2 ** nn) for nn in ns_plot]
    ax.semilogy(ns_plot, validity_theory, 'o-', color='tab:red',
                label='DFM validity (theory)', lw=2)
    ax.axhline(1.0, color='tab:blue', ls='--', lw=2,
               label='FM validity (always 100%)')
    ax.set_xlabel('n')
    ax.set_ylabel('Validity Rate')
    ax.set_title('(D) Validity vs n (k=n/2)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel E: Sample quality vs MCMC budget
    ax = axes[1, 1]
    for beta_val in betas:
        budgets = []
        tvs = []
        for r in results:
            if r['method'].startswith('MCMC-') and r['beta'] == beta_val:
                budget = int(r['method'].split('-')[1])
                if r.get('tv') is not None:
                    budgets.append(budget)
                    tvs.append(r['tv'])
        if budgets:
            ax.plot(budgets, tvs, 'o-', label=f'MCMC β={beta_val}', lw=1.5)
    # FM horizontal lines
    for r in results:
        if r['method'] == 'FM' and r.get('tv') is not None:
            ax.axhline(r['tv'], ls='--', alpha=0.5,
                        label=f'FM β={r["beta"]}')
    ax.set_xlabel('MCMC Steps')
    ax.set_ylabel('TV Distance')
    ax.set_title('(E) Quality vs MCMC Budget')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # Panel F: Sample gallery
    ax = axes[1, 2]
    ax.set_title('(F) Generated Samples (4×4 binary grids)')
    ax.axis('off')
    if n == 16:
        # Show FM, DFM, MCMC samples as 4x4 grids
        inner = axes[1, 2].inset_axes([0, 0, 1, 1])
        inner.axis('off')
        n_show = 5
        row_labels = ['FM', 'DFM', 'True']
        all_show = []
        for method in ['FM', 'DFM']:
            for r in results:
                if r['method'] == method and r['beta'] == beta_hard:
                    samps = r.get('samples', np.array([]))
                    if len(samps) >= n_show:
                        all_show.append(samps[:n_show])
                    break
        # True samples
        rng2 = np.random.default_rng(42)
        true_samps = np.array([
            mcmc_kawasaki(lambda x: ising_energy(x, J, h), n, k,
                          beta_hard, 5000, rng2)
            for _ in range(n_show)
        ])
        all_show.append(true_samps)

        for row_idx, (label, samps) in enumerate(
                zip(row_labels[:len(all_show)], all_show)):
            for col_idx in range(min(n_show, len(samps))):
                left = col_idx / (n_show + 0.5)
                bottom = 1 - (row_idx + 1) / (len(all_show) + 0.3)
                w = 0.8 / (n_show + 0.5)
                h_ax = 0.8 / (len(all_show) + 0.3)
                sub = inner.inset_axes([left + 0.05, bottom, w, h_ax])
                grid = samps[col_idx].reshape(4, 4)
                sub.imshow(grid, cmap='binary', vmin=0, vmax=1)
                sub.set_xticks([])
                sub.set_yticks([])
                if col_idx == 0:
                    sub.set_ylabel(label, fontsize=7, rotation=0,
                                   labelpad=25, va='center')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out_path}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ex20: Flow Matching on Johnson Graph')
    parser.add_argument('--n', type=int, default=16)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--betas', type=float, nargs='+', default=[0.5, 1.0, 2.0])
    parser.add_argument('--n-epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--mcmc-pool-size', type=int, default=10000)
    parser.add_argument('--mcmc-chain-length', type=int, default=5000)
    parser.add_argument('--n-eval-samples', type=int, default=2000)
    parser.add_argument('--n-gen-steps', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    n, k = args.n, args.k
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

    print(f"\n=== Experiment 20: Flow Matching on J({n},{k}) ===", flush=True)
    from math import comb
    print(f"Configuration space: C({n},{k}) = {comb(n, k)}", flush=True)
    print(f"Betas: {betas}", flush=True)

    # ── Generate Ising model ─────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    J = rng.standard_normal((n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    h_field = rng.standard_normal(n) * 0.5
    print(f"Ising model: J ({n}x{n}), h ({n},)", flush=True)

    # ── MCMC pools ───────────────────────────────────────────────────────────
    print("\nGenerating MCMC pools...", flush=True)
    mcmc_pools = {}
    for beta in betas:
        print(f"  beta={beta}: {args.mcmc_pool_size} samples, "
              f"chain_length={args.mcmc_chain_length}...", flush=True)
        mcmc_pools[beta] = generate_mcmc_pool(
            J, h_field, n, k, beta,
            args.mcmc_pool_size, args.mcmc_chain_length,
            seed=args.seed + int(beta * 1000))
        energies = np.array([ising_energy(x, J, h_field)
                             for x in mcmc_pools[beta]])
        print(f"    Energy: mean={energies.mean():.2f}, "
              f"std={energies.std():.2f}", flush=True)

    # ── Exact Boltzmann (small n only) ───────────────────────────────────────
    configs_exact = None
    probs_exact = None
    if comb(n, k) <= 50000:
        print("\nComputing exact Boltzmann distribution...", flush=True)
        configs_exact, _ = compute_exact_boltzmann(J, h_field, n, k, betas[0])
        probs_exact = {}
        for beta in betas:
            _, probs_exact[beta] = compute_exact_boltzmann(
                J, h_field, n, k, beta)
            print(f"  beta={beta}: entropy={-np.sum(probs_exact[beta] * np.log(probs_exact[beta] + 1e-15)):.2f}",
                  flush=True)

    # ── Checkpointing ────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    fm_ckpt = os.path.join(ckpt_dir, f'ex20_fm_n{n}_k{k}.pt')
    dfm_ckpt = os.path.join(ckpt_dir, f'ex20_dfm_n{n}_k{k}.pt')

    # ── Configuration spaces ─────────────────────────────────────────────────
    johnson_space = JohnsonSpace(n, k, J, h_field)
    dfm_space = DFMSpace(n, J, h_field)

    # ── Train FM (transition_order=2: pairwise swaps) ─────────────────────
    print("\n--- Training FM (ConfigurationRatePredictor, k=2) ---", flush=True)
    fm_model = ConfigurationRatePredictor(
        node_feature_dim=2, edge_feature_dim=1, global_dim=2,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        transition_order=2)

    if args.resume and os.path.exists(fm_ckpt):
        fm_model.load_state_dict(torch.load(fm_ckpt, map_location='cpu'))
        print(f"  Loaded from {fm_ckpt}", flush=True)
    else:
        # Train on all betas jointly by randomly sampling beta per example
        # The training loop calls config_space.sample_target(rng, beta=..., mcmc_pool=...)
        # We pick a random beta and its pool per sample via a wrapper
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

        multi_space = MultiBetaSampler(johnson_space, betas, mcmc_pools)
        result = train_configuration_fm(
            fm_model, multi_space, {},
            n_epochs=args.n_epochs, batch_size=args.batch_size,
            lr=args.lr, device=device, seed=args.seed)
        torch.save(fm_model.state_dict(), fm_ckpt)
        print(f"  Saved to {fm_ckpt}", flush=True)

    fm_model.to(device)

    # ── Train DFM (transition_order=1: single bit flips) ──────────────────
    print("\n--- Training DFM (ConfigurationRatePredictor, k=1) ---", flush=True)
    dfm_model = ConfigurationRatePredictor(
        node_feature_dim=2, edge_feature_dim=1, global_dim=2,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        transition_order=1)

    if args.resume and os.path.exists(dfm_ckpt):
        dfm_model.load_state_dict(torch.load(dfm_ckpt, map_location='cpu'))
        print(f"  Loaded from {dfm_ckpt}", flush=True)
    else:
        multi_dfm = MultiBetaSampler(dfm_space, betas, mcmc_pools)
        result = train_configuration_fm(
            dfm_model, multi_dfm, {},
            n_epochs=args.n_epochs, batch_size=args.batch_size,
            lr=args.lr, device=device, seed=args.seed + 1)
        torch.save(dfm_model.state_dict(), dfm_ckpt)
        print(f"  Saved to {dfm_ckpt}", flush=True)

    dfm_model.to(device)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n--- Evaluation ---", flush=True)
    results = evaluate_all(
        fm_model, dfm_model, J, h_field, n, k, betas, device,
        johnson_space=johnson_space, dfm_space=dfm_space,
        mcmc_pools=mcmc_pools,
        n_samples=args.n_eval_samples, n_steps=args.n_gen_steps,
        configs_exact=configs_exact, probs_exact=probs_exact)

    print_results(results)

    # ── Plot ─────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_results(results, J, h_field, n, k, betas,
                 os.path.join(out_dir, 'ex20_johnson_graph.png'),
                 configs_exact=configs_exact, probs_exact=probs_exact,
                 mcmc_pools=mcmc_pools)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
