"""
Experiment 22: Degree-Preserving Graph Generation via Double Edge Swaps.

Learns to sample from a community-structured distribution over graphs
with fixed degree sequence, using double edge swaps as transitions.
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
import networkx as nx

from otfm.models.predictor import ConfigurationRatePredictor
from otfm.train.configuration import train_configuration_fm
from otfm.configuration.sample import generate_samples
from otfm.configuration.spaces.degree_sequence import DegreeSequenceSpace


# ── Energy / MCMC ────────────────────────────────────────────────────────────

def community_energy(A, communities, n):
    """Energy: sum of inter-community edges."""
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1 and communities[i] != communities[j]:
                E += 1.0
    return E


def mcmc_degree_preserving(space, energy_fn, beta, n_steps, rng,
                           config=None):
    """Metropolis-Hastings with double edge swap proposals."""
    if config is None:
        config = space.sample_source(rng)
    A = space._config_to_adj(config)
    E_current = energy_fn(A)

    for _ in range(n_steps):
        edges = list(zip(*np.where(np.triu(A) > 0)))
        if len(edges) < 2:
            continue
        idx = rng.choice(len(edges), size=2, replace=False)
        a, b = edges[idx[0]]
        c, d = edges[idx[1]]

        if a == c or a == d or b == c or b == d:
            continue

        # Try first rewiring
        if A[a, c] == 0 and A[b, d] == 0:
            A_new = A.copy()
            A_new[a, b] = A_new[b, a] = 0
            A_new[c, d] = A_new[d, c] = 0
            A_new[a, c] = A_new[c, a] = 1
            A_new[b, d] = A_new[d, b] = 1
            E_new = energy_fn(A_new)
            dE = E_new - E_current
            if dE < 0 or rng.uniform() < np.exp(-beta * dE):
                A = A_new
                E_current = E_new
                continue

        # Try second rewiring
        if A[a, d] == 0 and A[b, c] == 0:
            A_new = A.copy()
            A_new[a, b] = A_new[b, a] = 0
            A_new[c, d] = A_new[d, c] = 0
            A_new[a, d] = A_new[d, a] = 1
            A_new[b, c] = A_new[c, b] = 1
            E_new = energy_fn(A_new)
            dE = E_new - E_current
            if dE < 0 or rng.uniform() < np.exp(-beta * dE):
                A = A_new
                E_current = E_new

    return space._adj_to_config(A)


def generate_mcmc_pool(space, energy_fn, beta, pool_size,
                       chain_length, seed=42):
    """Generate MCMC pool with thinning."""
    rng = np.random.default_rng(seed)
    pool = []
    config = mcmc_degree_preserving(space, energy_fn, beta,
                                     chain_length, rng)
    for _ in range(pool_size):
        config = mcmc_degree_preserving(space, energy_fn, beta,
                                         max(chain_length // 10, 500),
                                         rng, config=config)
        pool.append(config.copy())
    return np.array(pool, dtype=np.float32)


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
        if isinstance(config, tuple):
            return config
        return config, {'beta': beta}

    def global_features(self, t=0.0, **kwargs):
        beta = kwargs.get('beta', 1.0)
        return self._space.global_features(t=t, beta=beta)


# ── Evaluation ───────────────────────────────────────────────────────────────

def compute_modularity(A, communities, n):
    """Newman-Girvan modularity Q."""
    m = A.sum() / 2
    if m == 0:
        return 0.0
    degrees = A.sum(axis=1)
    Q = 0.0
    for i in range(n):
        for j in range(n):
            if communities[i] == communities[j]:
                Q += A[i, j] - degrees[i] * degrees[j] / (2 * m)
    return float(Q / (2 * m))


def compute_clustering(A, n):
    """Average clustering coefficient."""
    G = nx.from_numpy_array(A)
    return float(nx.average_clustering(G))


def degree_validity(config, space):
    """Check if config has correct degree sequence."""
    A = space._config_to_adj(config)
    degrees = A.sum(axis=1).astype(int)
    return np.array_equal(degrees, space.degree_seq)


def evaluate_all(fm_model, space, communities, n, betas, device,
                 mcmc_pools, energy_fn, n_samples=500, n_steps=100,
                 mcmc_budgets=(500, 2000, 5000)):
    """Evaluate FM and baselines."""
    rng = np.random.default_rng(42)
    results = []

    for beta in betas:
        print(f"\n  beta={beta}:", flush=True)
        gt = mcmc_pools[beta]
        gt_energies = np.array([community_energy(space._config_to_adj(x),
                                                  communities, n)
                                for x in gt])
        gt_mods = np.array([compute_modularity(space._config_to_adj(x),
                                                communities, n)
                            for x in gt[:200]])
        gt_clust = np.array([compute_clustering(space._config_to_adj(x), n)
                             for x in gt[:200]])

        def _metrics(samples, method):
            validity = float(np.mean([degree_validity(x, space)
                                      for x in samples]))
            energies = np.array([community_energy(space._config_to_adj(x),
                                                   communities, n)
                                 for x in samples])
            e_bias = abs(float(energies.mean() - gt_energies.mean()))
            ks_stat, _ = ks_2samp(energies, gt_energies) if len(energies) > 1 else (1.0, 0)

            mods = np.array([compute_modularity(space._config_to_adj(x),
                                                 communities, n)
                             for x in samples[:200]])
            mod_rmse = float(np.sqrt(((mods.mean() - gt_mods.mean()) ** 2)))

            clusts = np.array([compute_clustering(space._config_to_adj(x), n)
                               for x in samples[:100]])
            clust_rmse = float(np.sqrt(
                ((clusts.mean() - gt_clust[:100].mean()) ** 2)))

            return {
                'method': method, 'beta': beta, 'validity': validity,
                'energy_mean': float(energies.mean()),
                'energy_bias': e_bias, 'energy_ks': ks_stat,
                'gt_energy_mean': float(gt_energies.mean()),
                'modularity': float(mods.mean()),
                'gt_modularity': float(gt_mods.mean()),
                'mod_rmse': mod_rmse,
                'clustering': float(clusts.mean()),
                'clust_rmse': clust_rmse,
                'energies': energies, 'samples': samples,
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
            mc_samps = np.array([
                mcmc_degree_preserving(space, energy_fn, beta, budget, rng)
                for _ in range(min(200, n_samples))
            ])
            results.append(_metrics(mc_samps, f'MCMC-{budget}'))

    return results


def print_results(results):
    print(f"\n{'Method':15s} {'beta':>5s} {'E bias':>7s} {'E KS':>6s} "
          f"{'Mod':>6s} {'Clust':>6s} {'Valid%':>7s}", flush=True)
    print("-" * 60, flush=True)
    for r in results:
        print(f"{r['method']:15s} {r['beta']:5.1f} {r['energy_bias']:7.2f} "
              f"{r['energy_ks']:6.3f} {r['modularity']:6.3f} "
              f"{r['clustering']:6.3f} {r['validity']*100:6.1f}%", flush=True)


# ── Plotting ─────────────────────────────────────────────────────────────────

def draw_graph(ax, A, communities, n, title=''):
    """Draw graph with community coloring."""
    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, seed=42)
    colors = ['tab:blue' if communities[i] == 0 else 'tab:orange'
              for i in range(n)]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                           node_size=60, edgecolors='k', linewidths=0.3)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.5)
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def plot_results(results, space, communities, n, betas, mcmc_pools,
                 out_path):
    """2×3 results figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Ex22: Degree-Preserving Graph Generation (n={n})',
                 fontsize=12)
    beta_hard = max(betas)
    rng = np.random.default_rng(42)

    # Panel A: Graph gallery
    ax = axes[0, 0]
    ax.set_title(f'(A) FM vs True (β={beta_hard})', fontsize=10)
    ax.axis('off')
    inner = ax.inset_axes([0, 0, 1, 1])
    inner.axis('off')
    for r in results:
        if r['method'] == 'FM' and r['beta'] == beta_hard:
            fm_samps = r['samples']
            break
    gt_samps = mcmc_pools[beta_hard]
    for row, (label, samps) in enumerate([('FM', fm_samps), ('True', gt_samps)]):
        for col in range(min(4, len(samps))):
            sub = inner.inset_axes([col / 4.5, 1 - (row + 1) / 2.3,
                                    0.2, 0.4])
            A = space._config_to_adj(samps[col])
            draw_graph(sub, A, communities, n,
                       title=f'{label}' if col == 0 else '')

    # Panel B: Morphing sequence
    ax = axes[0, 1]
    ax.set_title('(B) FM Trajectory', fontsize=10)
    ax.axis('off')
    if len(fm_samps) > 0:
        # Generate one trajectory with snapshots
        c0 = space.sample_source(rng)
        times = [0.0, 0.25, 0.5, 0.75, 1.0]
        inner_b = ax.inset_axes([0, 0, 1, 1])
        inner_b.axis('off')
        for ti, t_val in enumerate(times):
            sub = inner_b.inset_axes([ti / 5.5, 0.1, 0.17, 0.8])
            # For visualization, just show the t=0 source and t=1 generated
            if ti == 0:
                draw_graph(sub, space._config_to_adj(c0), communities, n,
                           title=f't={t_val}')
            else:
                idx = min(ti, len(fm_samps) - 1)
                draw_graph(sub, space._config_to_adj(fm_samps[idx]),
                           communities, n, title=f't={t_val}')

    # Panel C: Modularity distribution
    ax = axes[0, 2]
    for r in results:
        if r['beta'] == beta_hard:
            if r['method'] in ['FM', 'MCMC-5000']:
                samps = r['samples'][:200]
                mods = [compute_modularity(space._config_to_adj(x),
                                            communities, n) for x in samps]
                ax.hist(mods, bins=20, alpha=0.5, label=r['method'],
                        density=True)
    gt_mods = [compute_modularity(space._config_to_adj(x), communities, n)
               for x in mcmc_pools[beta_hard][:200]]
    ax.hist(gt_mods, bins=20, alpha=0.4, label='Target', density=True,
            histtype='step', lw=2, color='black')
    ax.set_xlabel('Modularity Q')
    ax.set_ylabel('Density')
    ax.set_title(f'(C) Modularity (β={beta_hard})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel D: Eigenvalue spectrum
    ax = axes[1, 0]
    for method, color in [('FM', 'tab:blue'), ('MCMC-5000', 'tab:green')]:
        for r in results:
            if r['method'] == method and r['beta'] == beta_hard:
                all_eigs = []
                for x in r['samples'][:100]:
                    A = space._config_to_adj(x)
                    eigs = np.linalg.eigvalsh(A)
                    all_eigs.extend(eigs.tolist())
                ax.hist(all_eigs, bins=40, alpha=0.4, label=method,
                        density=True, color=color)
                break
    gt_eigs = []
    for x in mcmc_pools[beta_hard][:100]:
        gt_eigs.extend(np.linalg.eigvalsh(space._config_to_adj(x)).tolist())
    ax.hist(gt_eigs, bins=40, alpha=0.4, label='Target', density=True,
            histtype='step', lw=2, color='black')
    ax.set_xlabel('Eigenvalue')
    ax.set_title(f'(D) Eigenvalue Spectrum (β={beta_hard})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel E: Validity
    ax = axes[1, 1]
    for r in results:
        if r['method'] == 'FM':
            ax.bar(r['beta'], r['validity'] * 100, width=0.15,
                   color='tab:blue', alpha=0.7)
    ax.set_xlabel('β')
    ax.set_ylabel('Degree Validity (%)')
    ax.set_title('(E) FM Degree Validity')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Panel F: Clustering coefficient
    ax = axes[1, 2]
    for method in ['FM', 'MCMC-5000']:
        cs, bs = [], []
        for r in results:
            if r['method'] == method:
                cs.append(r['clustering'])
                bs.append(r['beta'])
        if cs:
            ax.plot(bs, cs, 'o-', label=method, lw=1.5)
    gt_cs = []
    for beta in betas:
        clusts = [compute_clustering(space._config_to_adj(x), n)
                  for x in mcmc_pools[beta][:100]]
        gt_cs.append(float(np.mean(clusts)))
    ax.plot(betas, gt_cs, 'k--', lw=2, label='Target', alpha=0.7)
    ax.set_xlabel('β')
    ax.set_ylabel('Clustering Coefficient')
    ax.set_title('(F) Clustering vs β')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out_path}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Ex22: Degree-Preserving Graph Generation')
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--degree', type=int, default=6)
    parser.add_argument('--n-communities', type=int, default=2)
    parser.add_argument('--betas', type=float, nargs='+',
                        default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument('--n-epochs', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--mcmc-pool-size', type=int, default=5000)
    parser.add_argument('--mcmc-chain-length', type=int, default=10000)
    parser.add_argument('--n-eval-samples', type=int, default=500)
    parser.add_argument('--n-gen-steps', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    n = args.n
    d = args.degree
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

    print(f"\n=== Experiment 22: Degree-Preserving Graphs ===", flush=True)
    print(f"n={n}, degree={d}, communities={args.n_communities}", flush=True)

    # Community assignment
    rng = np.random.default_rng(args.seed)
    communities = np.array([i % args.n_communities for i in range(n)])
    rng.shuffle(communities)
    print(f"Communities: {[int((communities == c).sum()) for c in range(args.n_communities)]}",
          flush=True)

    # Degree sequence
    degree_seq = np.full(n, d, dtype=int)

    # Energy function
    def energy_fn(A):
        return community_energy(A, communities, n)

    # Space
    space = DegreeSequenceSpace(n, degree_seq, communities=communities,
                                betas=betas)

    # MCMC pools
    print("\nGenerating MCMC pools...", flush=True)
    mcmc_pools = {}
    for beta in betas:
        print(f"  β={beta}...", flush=True)
        mcmc_pools[beta] = generate_mcmc_pool(
            space, energy_fn, beta,
            args.mcmc_pool_size, args.mcmc_chain_length,
            seed=args.seed + int(beta * 1000))
        energies = [community_energy(space._config_to_adj(x), communities, n)
                    for x in mcmc_pools[beta][:100]]
        print(f"    Energy: mean={np.mean(energies):.1f}, "
              f"std={np.std(energies):.1f}", flush=True)

    space.pools = mcmc_pools
    space.betas = betas

    # Checkpoints
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    fm_ckpt = os.path.join(ckpt_dir, f'ex22_fm_n{n}_d{d}.pt')

    # Train FM
    print("\n--- Training FM (k=4 double swaps) ---", flush=True)
    node_feat_dim = 1 + args.n_communities  # degree + community one-hot
    edge_feat_dim = 2  # A_ij + same_community
    fm_model = ConfigurationRatePredictor(
        node_feature_dim=node_feat_dim, edge_feature_dim=edge_feat_dim,
        global_dim=2,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        transition_order=4)

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
        fm_model, space, communities, n, betas, device,
        mcmc_pools, energy_fn,
        n_samples=args.n_eval_samples, n_steps=args.n_gen_steps)

    print_results(results)

    # Plot
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_results(results, space, communities, n, betas, mcmc_pools,
                 os.path.join(out_dir, 'ex22_degree_preserving.png'))

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
