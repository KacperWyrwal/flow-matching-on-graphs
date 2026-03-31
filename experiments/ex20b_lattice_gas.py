"""
Experiment 20b: Lattice Gas on 2D Grid.

Binary lattice gas: place k atoms on an L×L periodic grid. Same JohnsonSpace
and ConfigurationRatePredictor as Ex20, but with lattice neighbor interactions
instead of random couplings. Demonstrates constraint-preserving sampling on
a physically interpretable problem with visible phase transition.
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

from johnson_fm.energy import (
    ising_energy, mcmc_kawasaki, generate_mcmc_pool, compute_exact_boltzmann,
)
from otfm.models.predictor import ConfigurationRatePredictor
from otfm.train.configuration import train_configuration_fm
from otfm.configuration.sample import generate_samples
from otfm.configuration.spaces.johnson import JohnsonSpace


# ── Lattice construction ─────────────────────────────────────────────────────

def make_lattice_coupling(L):
    """Build nearest-neighbor coupling matrix on L×L periodic grid."""
    n = L * L
    J = np.zeros((n, n))
    for y in range(L):
        for x in range(L):
            i = y * L + x
            j = y * L + (x + 1) % L          # right
            J[i, j] = J[j, i] = 1.0
            j = ((y + 1) % L) * L + x        # down
            J[i, j] = J[j, i] = 1.0
    return J


# ── Spatial metrics ──────────────────────────────────────────────────────────

def compute_domains(config, L):
    """Connected components of occupied sites (4-connectivity)."""
    grid = config.reshape(L, L)
    labeled, n_domains = ndimage_label(grid)
    return [int((labeled == c).sum()) for c in range(1, n_domains + 1)]


def compute_spatial_correlation(samples, L, max_r=None):
    """C(r) = <x_i x_{i+r}> averaged over sites and samples (vectorized)."""
    if max_r is None:
        max_r = L
    grids = samples.reshape(-1, L, L)
    mean_x = grids.mean()

    C_r = np.zeros(max_r + 1)
    counts = np.zeros(max_r + 1)

    for dy in range(-(L // 2), L // 2 + 1):
        for dx in range(-(L // 2), L // 2 + 1):
            r = min(abs(dx), L - abs(dx)) + min(abs(dy), L - abs(dy))
            if r > max_r:
                continue
            shifted = np.roll(np.roll(grids, dy, axis=1), dx, axis=2)
            C_r[r] += (grids * shifted).mean() - mean_x ** 2
            counts[r] += 1

    valid = counts > 0
    C_r[valid] /= counts[valid]
    rs = np.arange(max_r + 1)
    return rs[valid], C_r[valid]


def compute_validity(samples, k):
    return float(np.mean(samples.sum(axis=1) == k))


def compute_energy_stats(samples, J, h):
    energies = np.array([ising_energy(x, J, h) for x in samples])
    return float(energies.mean()), float(energies.std()), energies


# ── MultiBetaSampler ─────────────────────────────────────────────────────────

class MultiBetaSampler:
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

def evaluate_all(fm_model, space, J, h, n, k, L, betas, device,
                 mcmc_pools, n_samples=2000, n_steps=100,
                 mcmc_budgets=(1000, 5000)):
    rng = np.random.default_rng(42)
    results = []

    for beta in betas:
        print(f"\n  beta={beta}:", flush=True)
        gt = mcmc_pools[beta]
        gt_energies = np.array([ising_energy(x, J, h) for x in gt])

        def _metrics(samples, method):
            validity = compute_validity(samples, k)
            valid = samples[samples.sum(axis=1) == k]
            e_mean, e_std, energies = compute_energy_stats(samples, J, h)
            e_bias = abs(e_mean - float(gt_energies.mean()))
            ks_stat = ks_2samp(energies, gt_energies)[0] if len(energies) > 1 else 1.0
            return {
                'method': method, 'beta': beta, 'validity': validity,
                'energy_mean': e_mean, 'energy_bias': e_bias,
                'energy_ks': ks_stat,
                'gt_energy_mean': float(gt_energies.mean()),
                'energies': energies, 'samples': valid,
            }

        # FM
        print(f"    FM...", flush=True)
        fm_samples = generate_samples(
            fm_model, space, n_samples, n_steps, device,
            seed=100, beta=beta)
        results.append(_metrics(fm_samples, 'FM'))

        # MCMC baselines
        for budget in mcmc_budgets:
            print(f"    MCMC-{budget}...", flush=True)
            energy_fn = lambda x, _J=J, _h=h: ising_energy(x, _J, _h)
            mc = np.array([
                mcmc_kawasaki(energy_fn, n, k, beta, budget, rng)
                for _ in range(min(500, n_samples))
            ])
            results.append(_metrics(mc, f'MCMC-{budget}'))

    return results


def print_results(results):
    print(f"\n{'Method':15s} {'beta':>5s} {'E bias':>7s} {'E KS':>6s} "
          f"{'Valid%':>7s}", flush=True)
    print("-" * 45, flush=True)
    for r in results:
        print(f"{r['method']:15s} {r['beta']:5.1f} {r['energy_bias']:7.2f} "
              f"{r['energy_ks']:6.3f} {r['validity']*100:6.1f}%", flush=True)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(results, J, h, n, k, L, betas, mcmc_pools, out_path):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'Ex20b: Lattice Gas on {L}×{L} Grid (k={k})', fontsize=13)
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)
    beta_hard = max(betas)

    # Panel A: Phase transition montage (FM samples)
    ax = fig.add_subplot(gs[0, :])
    ax.set_title('(A) FM Samples Across β', fontsize=11)
    ax.axis('off')
    n_show = 5
    inner = ax.inset_axes([0, 0, 1, 1])
    inner.axis('off')
    for bi, beta in enumerate(betas):
        for r in results:
            if r['method'] == 'FM' and r['beta'] == beta:
                samps = r['samples']
                for ci in range(min(n_show, len(samps))):
                    left = ci / (n_show + 0.3)
                    bottom = 1 - (bi + 1) / (len(betas) + 0.2)
                    w = 0.85 / (n_show + 0.3)
                    h_ax = 0.85 / (len(betas) + 0.2)
                    sub = inner.inset_axes([left + 0.02, bottom, w, h_ax])
                    sub.imshow(samps[ci].reshape(L, L), cmap='Blues',
                               vmin=0, vmax=1, interpolation='nearest')
                    sub.set_xticks([])
                    sub.set_yticks([])
                    if ci == 0:
                        sub.set_ylabel(f'β={beta}', fontsize=8, rotation=0,
                                       labelpad=25, va='center')
                break

    # Panel B: FM vs MCMC vs True at high beta
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title(f'(B) Comparison at β={beta_hard}', fontsize=10)
    ax.axis('off')
    inner_b = ax.inset_axes([0, 0, 1, 1])
    inner_b.axis('off')
    rows_data = []
    for method in ['FM', f'MCMC-5000']:
        for r in results:
            if r['method'] == method and r['beta'] == beta_hard:
                rows_data.append((method, r['samples'][:4]))
                break
    rows_data.append(('True', mcmc_pools[beta_hard][:4]))
    for ri, (label, samps) in enumerate(rows_data):
        for ci in range(min(4, len(samps))):
            sub = inner_b.inset_axes([ci / 4.5, 1 - (ri + 1) / 3.3,
                                      0.2, 0.28])
            sub.imshow(samps[ci].reshape(L, L), cmap='Blues',
                       vmin=0, vmax=1, interpolation='nearest')
            sub.set_xticks([])
            sub.set_yticks([])
            if ci == 0:
                sub.set_ylabel(label, fontsize=7, rotation=0,
                               labelpad=22, va='center')

    # Panel C: Energy vs beta
    ax = fig.add_subplot(gs[1, 1])
    for method in ['FM', 'MCMC-5000']:
        es, bs = [], []
        for r in results:
            if r['method'] == method:
                es.append(r['energy_mean'])
                bs.append(r['beta'])
        if es:
            ax.plot(bs, es, 'o-', label=method, lw=1.5)
    gt_es = [float(np.mean([ising_energy(x, J, h) for x in mcmc_pools[b][:200]]))
             for b in betas]
    ax.plot(betas, gt_es, 'k--', lw=2, label='Target', alpha=0.7)
    ax.set_xlabel('β')
    ax.set_ylabel('Mean Energy')
    ax.set_title('(C) Energy vs β')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel D: Domain size distribution
    ax = fig.add_subplot(gs[1, 2])
    for method, color in [('FM', 'tab:blue'), ('MCMC-5000', 'tab:green')]:
        for r in results:
            if r['method'] == method and r['beta'] == beta_hard:
                samps = r['samples'][:200]
                all_sizes = []
                for x in samps:
                    all_sizes.extend(compute_domains(x, L))
                if all_sizes:
                    ax.hist(all_sizes, bins=20, alpha=0.5, label=method,
                            color=color, density=True)
                break
    gt_sizes = []
    for x in mcmc_pools[beta_hard][:200]:
        gt_sizes.extend(compute_domains(x, L))
    ax.hist(gt_sizes, bins=20, alpha=0.4, label='Target', density=True,
            histtype='step', lw=2, color='black')
    ax.set_xlabel('Domain Size')
    ax.set_ylabel('Density')
    ax.set_title(f'(D) Domain Sizes (β={beta_hard})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel E: Spatial correlation
    ax = fig.add_subplot(gs[2, 0])
    for method, color in [('FM', 'tab:blue'), ('MCMC-5000', 'tab:green')]:
        for r in results:
            if r['method'] == method and r['beta'] == beta_hard:
                samps = r['samples'][:300]
                if len(samps) > 10:
                    rs_arr, cr = compute_spatial_correlation(samps, L,
                                                             max_r=L // 2)
                    ax.plot(rs_arr, cr, 'o-', label=method, color=color,
                            markersize=3)
                break
    rs_gt, cr_gt = compute_spatial_correlation(mcmc_pools[beta_hard][:300],
                                                L, max_r=L // 2)
    ax.plot(rs_gt, cr_gt, 'k--', label='Target', lw=2, alpha=0.7)
    ax.set_xlabel('Distance r')
    ax.set_ylabel('C(r)')
    ax.set_title(f'(E) Spatial Correlation (β={beta_hard})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel F: Energy bias by beta
    ax = fig.add_subplot(gs[2, 1])
    for method in ['FM', 'MCMC-1000']:
        biases, bs = [], []
        for r in results:
            if r['method'] == method:
                biases.append(r['energy_bias'])
                bs.append(r['beta'])
        if biases:
            ax.plot(bs, biases, 'o-', label=method, lw=1.5)
    ax.set_xlabel('β')
    ax.set_ylabel('|E bias|')
    ax.set_title('(F) Energy Bias vs β')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel G: DFM validity (theoretical)
    ax = fig.add_subplot(gs[2, 2])
    from math import comb
    Ls = [4, 6, 8, 10, 12]
    ks = [L_val ** 2 // 4 for L_val in Ls]
    validity = [comb(L_val ** 2, k_val) / (2 ** (L_val ** 2))
                for L_val, k_val in zip(Ls, ks)]
    ax.semilogy(Ls, validity, 'o-', color='tab:red', lw=2,
                label='DFM validity (ρ=1/4)')
    ax.axhline(1.0, color='tab:blue', ls='--', lw=2, label='FM (100%)')
    ax.set_xlabel('L')
    ax.set_ylabel('Validity Rate')
    ax.set_title('(G) DFM Validity vs Lattice Size')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out_path}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ex20b: Lattice Gas on 2D Grid')
    parser.add_argument('--L', type=int, default=8)
    parser.add_argument('--density', type=float, default=0.5,
                        help='Fraction of occupied sites')
    parser.add_argument('--betas', type=float, nargs='+',
                        default=[0.2, 0.5, 1.0, 2.0])
    parser.add_argument('--n-epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--mcmc-pool-size', type=int, default=10000)
    parser.add_argument('--mcmc-chain-length', type=int, default=50000)
    parser.add_argument('--n-eval-samples', type=int, default=2000)
    parser.add_argument('--n-gen-steps', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    L = args.L
    n = L * L
    k = int(round(n * args.density))
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

    print(f"\n=== Experiment 20b: Lattice Gas on {L}×{L} Grid ===", flush=True)
    print(f"n={n}, k={k} (density={k/n:.2f})", flush=True)
    from math import comb
    print(f"Configuration space: C({n},{k}) = {comb(n, k):.2e}", flush=True)

    # Lattice coupling
    rng = np.random.default_rng(args.seed)
    J = make_lattice_coupling(L)
    h = np.zeros(n)
    print(f"Lattice coupling: {int(J.sum()//2)} edges", flush=True)

    # JohnsonSpace with lattice coupling
    space = JohnsonSpace(n, k, J, h)

    # MCMC pools
    print("\nGenerating MCMC pools...", flush=True)
    mcmc_pools = {}
    for beta in betas:
        print(f"  β={beta}...", flush=True)
        mcmc_pools[beta] = generate_mcmc_pool(
            J, h, n, k, beta, args.mcmc_pool_size,
            args.mcmc_chain_length,
            seed=args.seed + int(beta * 1000))
        energies = np.array([ising_energy(x, J, h)
                             for x in mcmc_pools[beta][:100]])
        print(f"    Energy: mean={energies.mean():.1f}, "
              f"std={energies.std():.1f}", flush=True)

    # Checkpoints
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    fm_ckpt = os.path.join(ckpt_dir, f'ex20b_fm_L{L}_k{k}.pt')

    # Train FM
    print("\n--- Training FM ---", flush=True)
    fm_model = ConfigurationRatePredictor(
        node_feature_dim=2, edge_feature_dim=1, global_dim=2,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        transition_order=2)

    if args.resume and os.path.exists(fm_ckpt):
        fm_model.load_state_dict(torch.load(fm_ckpt, map_location='cpu'))
        print(f"  Loaded from {fm_ckpt}", flush=True)
    else:
        multi = MultiBetaSampler(space, betas, mcmc_pools)
        result = train_configuration_fm(
            fm_model, multi, {},
            n_epochs=args.n_epochs, batch_size=args.batch_size,
            lr=args.lr, device=device, seed=args.seed)
        torch.save(fm_model.state_dict(), fm_ckpt)
        print(f"  Saved to {fm_ckpt}", flush=True)

    fm_model.to(device)

    # Evaluate
    print("\n--- Evaluation ---", flush=True)
    results = evaluate_all(
        fm_model, space, J, h, n, k, L, betas, device,
        mcmc_pools, n_samples=args.n_eval_samples,
        n_steps=args.n_gen_steps)

    print_results(results)

    # Plot
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_results(results, J, h, n, k, L, betas, mcmc_pools,
                 os.path.join(out_dir, 'ex20b_lattice_gas.png'))

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
