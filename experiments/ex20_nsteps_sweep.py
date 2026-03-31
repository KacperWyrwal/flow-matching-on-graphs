"""Ex20 n_steps sweep: test FM inference discretization."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.stats import ks_2samp

from johnson_fm.energy import (
    ising_energy, generate_mcmc_pool, compute_exact_boltzmann,
)
from config_fm import ConfigurationRatePredictor
from config_fm.spaces import JohnsonSpace
from config_fm.sample import generate_samples
from experiments.ex20_johnson_graph import (
    compute_validity, compute_energy_stats,
    compute_pairwise_correlations, compute_tv_exact,
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=16)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--betas', type=float, nargs='+', default=[0.5, 1.0, 2.0])
    parser.add_argument('--n-steps-list', type=int, nargs='+',
                        default=[50, 100, 200, 500])
    parser.add_argument('--n-eval-samples', type=int, default=2000)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    n, k = args.n, args.k
    betas = args.betas

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}", flush=True)

    # Ising model (must match training)
    rng = np.random.default_rng(args.seed)
    J = rng.standard_normal((n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    h_field = rng.standard_normal(n) * 0.5

    # Load FM model
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'checkpoints')
    fm_ckpt = os.path.join(ckpt_dir, f'ex20_fm_n{n}_k{k}.pt')
    if not os.path.exists(fm_ckpt):
        print(f"ERROR: {fm_ckpt} not found", flush=True)
        return

    model = ConfigurationRatePredictor(
        node_feature_dim=2, edge_feature_dim=1, global_dim=2,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        transition_order=2)
    model.load_state_dict(torch.load(fm_ckpt, map_location='cpu'))
    model.to(device)
    model.eval()
    print(f"Loaded FM from {fm_ckpt}", flush=True)

    johnson_space = JohnsonSpace(n, k, J, h_field)

    # Ground truth
    print("Computing ground truth...", flush=True)
    gt = {}
    for beta in betas:
        pool = generate_mcmc_pool(J, h_field, n, k, beta, 5000, 5000,
                                  seed=args.seed + int(beta * 1000))
        gt_energies = np.array([ising_energy(x, J, h_field) for x in pool])
        gt_corr = compute_pairwise_correlations(pool)
        gt[beta] = {'samples': pool, 'energies': gt_energies,
                    'corr': gt_corr,
                    'mean_E': float(gt_energies.mean()),
                    'std_E': float(gt_energies.std())}
        print(f"  beta={beta}: GT mean_E={gt_energies.mean():.2f}", flush=True)

    # Exact Boltzmann (small n)
    from math import comb
    configs_exact, probs_exact = None, {}
    if comb(n, k) <= 50000:
        print("Computing exact Boltzmann...", flush=True)
        configs_exact, _ = compute_exact_boltzmann(J, h_field, n, k, betas[0])
        for beta in betas:
            _, probs_exact[float(round(beta, 6))] = compute_exact_boltzmann(
                J, h_field, n, k, beta)

    # Sweep
    print(f"\nn={n}, k={k}", flush=True)
    print(f"{'n_steps':>8s} {'beta':>5s} {'TV':>8s} {'E bias':>7s} "
          f"{'E KS':>6s} {'Corr RMSE':>10s} {'mean_E':>8s} {'tgt_E':>7s}",
          flush=True)
    print("-" * 65, flush=True)

    for n_steps in args.n_steps_list:
        for beta in betas:
            beta_key = float(round(beta, 6))
            samples = generate_samples(
                model, johnson_space, args.n_eval_samples,
                n_steps, device, seed=100, beta=beta)

            valid = samples[samples.sum(axis=1) == k]
            if len(valid) == 0:
                print(f"{n_steps:8d} {beta:5.1f}    (no valid samples)",
                      flush=True)
                continue

            energies = np.array([ising_energy(x, J, h_field) for x in valid])
            mean_E = float(energies.mean())
            e_bias = abs(mean_E - gt[beta]['mean_E'])
            ks_stat, _ = ks_2samp(energies, gt[beta]['energies'])
            corr = compute_pairwise_correlations(valid)
            corr_rmse = float(np.sqrt(
                ((corr - gt[beta]['corr']) ** 2).mean()))

            tv_str = "   -"
            if configs_exact is not None and beta_key in probs_exact:
                tv = compute_tv_exact(valid, k, configs_exact,
                                      probs_exact[beta_key])
                tv_str = f"{tv:.4f}"

            print(f"{n_steps:8d} {beta:5.1f} {tv_str:>8s} {e_bias:7.3f} "
                  f"{ks_stat:6.3f} {corr_rmse:10.4f} {mean_E:8.2f} "
                  f"{gt[beta]['mean_E']:7.2f}", flush=True)


if __name__ == '__main__':
    main()
