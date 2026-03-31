"""
Experiment 9: Distribution Inpainting with Classifier-Free Guidance on a 3x3 Grid.

A ConditionalGNNRateMatrixPredictor is trained with context dropout (p=0.15),
enabling classifier-free guidance at inference. Guidance weight w interpolates
between unconditional (w=-1) and amplified conditional (w>0). The experiment
sweeps w in {0.0, 0.5, 1.0, 2.0, 3.0} and identifies the best setting.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from graph_ot_fm import GraphStructure, total_variation, make_grid_graph
from meta_fm import (
    ConditionalGNNRateMatrixPredictor,
    InpaintingDataset,
    rate_matrix_to_edge_index,
    train_conditional,
    sample_trajectory_guided,
    get_device,
)


ROWS, COLS, N = 3, 3, 9

COMMUNITIES = {
    'A': [0, 1, 3],
    'B': [1, 2, 5],
    'C': [3, 6, 7],
    'D': [5, 7, 8],
}
COMMUNITY_KEYS = list(COMMUNITIES.keys())
GUIDANCE_WEIGHTS = [0.0, 0.5, 1.0, 2.0, 3.0]


def classify_community(dist):
    scores = {k: dist[nodes].sum() for k, nodes in COMMUNITIES.items()}
    return max(scores, key=scores.get)


def make_community_sample(community_key, rng, alpha=5.0, noise=0.01):
    dist = np.full(N, noise)
    for node, w in zip(COMMUNITIES[community_key],
                       rng.dirichlet(np.full(3, alpha))):
        dist[node] += w
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist


def corrupt(mu_clean, mask_idx):
    """Zero out masked nodes and renormalize."""
    mu_corr = mu_clean.copy()
    mu_corr[mask_idx] = 0.0
    s = mu_corr.sum()
    if s > 1e-12:
        mu_corr /= s
    else:
        mu_corr = np.ones(N) / N
    mask = np.ones(N, dtype=float)
    mask[mask_idx] = 0.0
    return mu_corr, mask


def naive_inpaint(mu_corr, mask):
    """Fill masked nodes with mean of observed nodes."""
    result = mu_corr.copy()
    obs_mean = mu_corr[mask == 1].mean()
    result[mask == 0] = obs_mean
    result /= result.sum()
    return result


def plot_grid_hm(ax, dist, title='', vmax=None, mask=None):
    grid = dist.reshape(ROWS, COLS)
    ax.imshow(grid, cmap='YlOrRd', vmin=0, vmax=vmax or dist.max(), aspect='equal')
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    if mask is not None:
        for node in np.where(mask == 0)[0]:
            r, c = divmod(node, COLS)
            ax.text(c, r, '×', ha='center', va='center',
                    fontsize=14, color='blue', fontweight='bold')


def run_eval(model, test_cases, n_masked, guidance_weight, device, rng_eval):
    """Evaluate model on test_cases at a given guidance weight."""
    results = []
    for case in test_cases:
        mu_clean, community = case['mu_clean'], case['community']
        mask_idx = rng_eval.choice(N, size=n_masked, replace=False)
        mu_corr, mask = corrupt(mu_clean, mask_idx)

        context = np.stack([mu_corr, mask], axis=-1)
        _, traj = sample_trajectory_guided(model, mu_corr, context,
                                           guidance_weight=guidance_weight,
                                           n_steps=200, device=device)
        mu_inp = traj[-1]
        mu_naive = naive_inpaint(mu_corr, mask)

        results.append({
            'mu_clean': mu_clean, 'mu_corr': mu_corr, 'mask': mask,
            'mu_inp': mu_inp, 'mu_naive': mu_naive, 'community': community,
            'tv_learned': total_variation(mu_inp, mu_clean),
            'tv_naive': total_variation(mu_naive, mu_clean),
            'community_correct': int(classify_community(mu_inp) == community),
            'traj': traj,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-weighting', type=str, default='uniform',
                        choices=['original', 'uniform', 'linear'],
                        help='Loss weighting scheme (default: uniform)')
    parser.add_argument('--n-epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=4)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    device = get_device()

    R = make_grid_graph(ROWS, COLS, weighted=False)
    graph = GraphStructure(R)
    edge_index = rate_matrix_to_edge_index(R)

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f'meta_model_ex9_inpainting_{args.loss_weighting}_{args.n_epochs}ep_h{args.hidden_dim}_l{args.n_layers}.pt')

    print("=== Experiment 9: Distribution Inpainting on 3×3 Grid ===\n")

    # ── Generate 200 clean distributions (50 per community) ───────────────────
    n_per_community = 50
    clean_dists = []
    clean_labels = []
    for key in COMMUNITY_KEYS:
        for _ in range(n_per_community):
            clean_dists.append(make_community_sample(key, rng))
            clean_labels.append(key)

    print(f"Generated {len(clean_dists)} clean distributions ({n_per_community} per community)")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = ConditionalGNNRateMatrixPredictor(
        edge_index=edge_index, n_nodes=N, context_dim=2, hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"Loaded checkpoint from {ckpt_path}")
        losses = None
    else:
        print("\nBuilding InpaintingDataset (10000 samples, 50 masks per distribution)...")
        dataset = InpaintingDataset(graph=graph, clean_distributions=clean_dists,
                                    n_masks_per_dist=50, n_masked_nodes=3,
                                    n_samples=10000, seed=42)

        print("Training ConditionalGNNRateMatrixPredictor (500 epochs, context_drop_prob=0.15)...")
        history = train_conditional(model, dataset, n_epochs=args.n_epochs, batch_size=256,
                                    lr=1e-3, device=device, context_drop_prob=0.15,
                                    loss_weighting=args.loss_weighting)
        losses = history['losses']
        print(f"Initial loss: {losses[0]:.6f}, Final loss: {losses[-1]:.6f}")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    model.eval()

    # ── 40 held-out test cases (10 per community) ────────────────────────────
    rng_test = np.random.default_rng(77)
    test_cases = []
    for key in COMMUNITY_KEYS:
        for _ in range(10):
            test_cases.append({'mu_clean': make_community_sample(key, rng_test),
                                'community': key})

    # ── Guidance weight sweep (M=3 masked nodes) ─────────────────────────────
    guidance_results = {}
    for w in GUIDANCE_WEIGHTS:
        rng_w = np.random.default_rng(77)  # same masks for fair comparison
        guidance_results[w] = run_eval(model, test_cases, n_masked=3,
                                       guidance_weight=w, device=device, rng_eval=rng_w)

    # ── Find best guidance weight by mean TV ─────────────────────────────────
    mean_tvs = {w: np.mean([r['tv_learned'] for r in guidance_results[w]])
                for w in GUIDANCE_WEIGHTS}
    best_w = min(mean_tvs, key=mean_tvs.get)
    results = guidance_results[best_w]

    # ── Ablation: M = 1..5 at best guidance weight ───────────────────────────
    ablation_results = {}
    for M in range(1, 6):
        rng_abl = np.random.default_rng(M * 100)
        ablation_results[M] = run_eval(model, test_cases, n_masked=M,
                                       guidance_weight=best_w, device=device,
                                       rng_eval=rng_abl)

    # ── Console output ─────────────────────────────────────────────────────────
    print("\n=== Experiment 9: Inpainting Results ===")
    if losses is not None:
        print(f"Training: initial loss = {losses[0]:.6f}, final loss = {losses[-1]:.6f}")
    print("\nGuidance weight sweep:")
    for w in GUIDANCE_WEIGHTS:
        gr = guidance_results[w]
        tv_vals = [r['tv_learned'] for r in gr]
        comm_acc = np.mean([r['community_correct'] for r in gr]) * 100
        print(f"  w={w}: TV={np.mean(tv_vals):.4f} ± {np.std(tv_vals):.4f}, "
              f"community_acc={comm_acc:.0f}%")

    print(f"\nBest guidance weight: w={best_w}")
    tv_l = [r['tv_learned'] for r in results]
    tv_n = [r['tv_naive'] for r in results]
    comm_acc_best = np.mean([r['community_correct'] for r in results]) * 100
    improvement = (np.mean(tv_n) - np.mean(tv_l)) / np.mean(tv_n) * 100
    print(f"\nAt best w={best_w}:")
    print(f"  Mean TV (learned):  {np.mean(tv_l):.4f} ± {np.std(tv_l):.4f}")
    print(f"  Mean TV (naive):    {np.mean(tv_n):.4f} ± {np.std(tv_n):.4f}")
    print(f"  Improvement:        {improvement:.1f}%")
    print(f"  Community preservation: {comm_acc_best:.0f}%")

    print("\nAblation (M masked nodes, best w):")
    for M in range(1, 6):
        ar = ablation_results[M]
        tv_m = np.mean([r['tv_learned'] for r in ar])
        acc_m = np.mean([r['community_correct'] for r in ar]) * 100
        print(f"  M={M}: TV={tv_m:.4f}, community_acc={acc_m:.0f}%")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle('Experiment 9: Distribution Inpainting with Classifier-Free Guidance',
                 fontsize=13)
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35)

    # Panel A: training loss
    ax_A = fig.add_subplot(gs[0, 0])
    if losses is not None:
        ax_A.plot(losses, color='steelblue', lw=1.5)
        ax_A.set_yscale('log')
    else:
        ax_A.text(0.5, 0.5, 'Loaded from\ncheckpoint',
                  transform=ax_A.transAxes, ha='center', va='center', fontsize=12)
    ax_A.set_xlabel('Epoch'); ax_A.set_ylabel('MSE Loss')
    ax_A.set_title('Panel A: Training Loss (p_drop=0.15)'); ax_A.grid(True, alpha=0.3)

    # Panel B: 4 example cases (one per community) at best guidance weight
    ax_B = fig.add_subplot(gs[0, 1])
    ax_B.axis('off')
    ax_B.set_title(f'Panel B: Examples at w={best_w}\n(one per community)', pad=12)
    inner_B = gs[0, 1].subgridspec(4, 3, hspace=0.8, wspace=0.3)
    example_cases = [next(r for r in results if r['community'] == k)
                     for k in COMMUNITY_KEYS]
    col_titles = ['Corrupted', 'Inpainted', 'True']
    for row, res in enumerate(example_cases):
        vmax_b = max(res['mu_clean'].max(), res['mu_inp'].max(), res['mu_corr'].max())
        for col, (dist, title) in enumerate([
                (res['mu_corr'], col_titles[0] if row == 0 else ''),
                (res['mu_inp'],  col_titles[1] if row == 0 else ''),
                (res['mu_clean'], col_titles[2] if row == 0 else '')]):
            ax_in = fig.add_subplot(inner_B[row, col])
            mask_arg = res['mask'] if col == 0 else None
            lbl = f'Com {res["community"]}' if col == 0 else title
            plot_grid_hm(ax_in, dist, lbl, vmax=vmax_b, mask=mask_arg)

    # Panel C: TV vs guidance weight (mean ± std)
    ax_C = fig.add_subplot(gs[0, 2])
    tv_means = [np.mean([r['tv_learned'] for r in guidance_results[w]])
                for w in GUIDANCE_WEIGHTS]
    tv_stds = [np.std([r['tv_learned'] for r in guidance_results[w]])
               for w in GUIDANCE_WEIGHTS]
    ax_C.errorbar(GUIDANCE_WEIGHTS, tv_means, yerr=tv_stds,
                  fmt='o-', color='steelblue', lw=2, ms=8, capsize=4)
    ax_C.axvline(best_w, color='red', ls='--', alpha=0.6, label=f'Best w={best_w}')
    ax_C.set_xlabel('Guidance weight w')
    ax_C.set_ylabel('Mean TV (learned vs true)')
    ax_C.set_title('Panel C: TV vs guidance weight\n(mean ± std, 40 test cases)')
    ax_C.legend(fontsize=8); ax_C.grid(True, alpha=0.3)

    # Panel D: community accuracy vs guidance weight
    ax_D = fig.add_subplot(gs[1, 0])
    comm_accs = [np.mean([r['community_correct'] for r in guidance_results[w]]) * 100
                 for w in GUIDANCE_WEIGHTS]
    ax_D.plot(GUIDANCE_WEIGHTS, comm_accs, 's-', color='salmon', lw=2, ms=8)
    ax_D.axvline(best_w, color='red', ls='--', alpha=0.6, label=f'Best w={best_w}')
    ax_D.set_xlabel('Guidance weight w')
    ax_D.set_ylabel('Community preservation (%)')
    ax_D.set_title('Panel D: Community accuracy vs guidance weight')
    ax_D.set_ylim(0, 115); ax_D.legend(fontsize=8); ax_D.grid(True, alpha=0.3)

    # Panel E: TV histogram at best w — learned vs naive
    ax_E = fig.add_subplot(gs[1, 1])
    tv_l_best = [r['tv_learned'] for r in results]
    tv_n_best = [r['tv_naive'] for r in results]
    bins = np.linspace(0, max(max(tv_l_best), max(tv_n_best)) + 0.02, 20)
    ax_E.hist(tv_l_best, bins=bins, alpha=0.7, label=f'Learned (w={best_w})',
              color='steelblue')
    ax_E.hist(tv_n_best, bins=bins, alpha=0.7, label='Naive (mean fill)',
              color='salmon')
    ax_E.set_xlabel('TV(inpainted, true)')
    ax_E.set_ylabel('Count')
    ax_E.set_title(f'Panel E: Learned vs naive at w={best_w}\n(M=3 masked nodes)')
    ax_E.legend(fontsize=8); ax_E.grid(True, alpha=0.3)

    # Panel F: ablation — TV and community accuracy vs M at best w
    ax_F = fig.add_subplot(gs[1, 2])
    Ms = list(range(1, 6))
    tv_by_M = [np.mean([r['tv_learned'] for r in ablation_results[M]]) for M in Ms]
    acc_by_M = [np.mean([r['community_correct'] for r in ablation_results[M]]) * 100
                for M in Ms]
    ax_F2 = ax_F.twinx()
    ax_F.plot(Ms, tv_by_M, 'o-', color='steelblue', lw=2, label='TV (left)')
    ax_F2.plot(Ms, acc_by_M, 's--', color='salmon', lw=2, label='Community acc % (right)')
    ax_F.set_xlabel('Number of masked nodes M')
    ax_F.set_ylabel('Mean TV', color='steelblue')
    ax_F2.set_ylabel('Community accuracy (%)', color='salmon')
    ax_F.set_title(f'Panel F: Ablation — M masked nodes\n(w={best_w})')
    ax_F.grid(True, alpha=0.3)
    lines1, labels1 = ax_F.get_legend_handles_labels()
    lines2, labels2 = ax_F2.get_legend_handles_labels()
    ax_F.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'ex9_inpainting.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
