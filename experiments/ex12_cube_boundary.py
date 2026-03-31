"""
Experiment 12 (v3): Diffusion-Based Boundary Observation on 4×4×4 Cube.

Pipeline:
  1. Source: single interior peak distribution
  2. Diffuse: mu_diffused = mu_source @ expm(tau_diff * R)
  3. Observe: boundary values of mu_diffused (interior zeroed, renormalized)
  4. Task: recover mu_source from boundary observation only

The 4×4×4 cube has 64 nodes, 56 boundary, 8 interior (all at depth 1).
This is the simplest nontrivial case: every interior node is adjacent to the
boundary, so diffusion reaches the surface in one hop.

After this works, scale to 5×5×5.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.linalg import expm

from graph_ot_fm import (
    total_variation,
    make_cube_graph,
    cube_boundary_mask,
)
from meta_fm import (
    FlexibleConditionalGNNRateMatrixPredictor,
    CubeBoundaryDataset,
    train_flexible_conditional,
    sample_trajectory_flexible,
    get_device,
)
from meta_fm.model import rate_matrix_to_edge_index


# ── Source / observation helpers ──────────────────────────────────────────────

def make_interior_peak_dist(N, rng, interior_idx):
    """Single peak at a random interior node."""
    peak_node = int(rng.choice(interior_idx))
    dist = np.ones(N) * 0.2 / (N - 1)
    dist[peak_node] = 0.8
    dist += rng.normal(0, 0.01, N)
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist, peak_node


def compute_boundary_observation(mu_source, R, mask, tau_diff=1.0):
    """Diffuse source, then observe boundary only (renormalized)."""
    mu_diffused = mu_source @ expm(tau_diff * R)
    mu_obs = mu_diffused * mask
    mu_obs = np.clip(mu_obs, 1e-12, None)
    mu_obs /= mu_obs.sum()
    return mu_obs


# ── Baselines ─────────────────────────────────────────────────────────────────

def baseline_naive(mu_obs, mask):
    """Fill interior with mean boundary value, renormalize."""
    result = mu_obs.copy()
    result[mask == 0] = mu_obs[mask == 1].mean()
    result /= result.sum()
    return result


def baseline_laplacian(mu_obs, mask, R):
    """Harmonic extension (Dirichlet problem): L_II * u_I = -L_IB * u_B."""
    L = -R.copy()
    boundary_idx = np.where(mask == 1)[0]
    interior_idx = np.where(mask == 0)[0]
    L_II = L[np.ix_(interior_idx, interior_idx)]
    L_IB = L[np.ix_(interior_idx, boundary_idx)]
    u_B  = mu_obs[boundary_idx]
    u_I  = np.linalg.solve(L_II, -L_IB @ u_B)
    result = mu_obs.copy()
    result[interior_idx] = u_I
    result = np.clip(result, 0, None)
    s = result.sum()
    return result / s if s > 1e-12 else result


# ── Metrics ───────────────────────────────────────────────────────────────────

def peak_recovery_top1(recovered, true_peak, interior_idx):
    """Is the highest-mass interior node the true peak?"""
    interior_vals = recovered[interior_idx]
    top1 = interior_idx[int(np.argmax(interior_vals))]
    return int(top1 == true_peak)


def peak_location_adjacent(recovered, true_peak, interior_idx, R):
    """Is the top-1 interior node adjacent (graph distance ≤ 1) to true peak?"""
    interior_vals = recovered[interior_idx]
    top1 = interior_idx[int(np.argmax(interior_vals))]
    if top1 == true_peak:
        return 1
    # adjacent in graph = R[i,j] > 0
    return int(R[top1, true_peak] > 0)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_test_case(model, R, edge_index, mask, mu_source, peak_node,
                       interior_idx, tau_diff, device):
    N = R.shape[0]
    mu_obs = compute_boundary_observation(mu_source, R, mask, tau_diff)
    context = np.stack([mu_obs, mask], axis=-1)
    mu_start = np.ones(N) / N

    _, traj = sample_trajectory_flexible(
        model, mu_start, context, edge_index, n_steps=200, device=device)
    mu_learned = traj[-1]
    mu_lap   = baseline_laplacian(mu_obs, mask, R)
    mu_naive = baseline_naive(mu_obs, mask)

    return {
        'mu_source':  mu_source,
        'mu_obs':     mu_obs,
        'mu_learned': mu_learned,
        'mu_lap':     mu_lap,
        'mu_naive':   mu_naive,
        'peak_node':  peak_node,
        # Full TV
        'tv_learned': total_variation(mu_learned, mu_source),
        'tv_lap':     total_variation(mu_lap,     mu_source),
        'tv_naive':   total_variation(mu_naive,   mu_source),
        # Interior TV (8 nodes)
        'tv_int_learned': total_variation(mu_learned[interior_idx], mu_source[interior_idx]),
        'tv_int_lap':     total_variation(mu_lap[interior_idx],     mu_source[interior_idx]),
        'tv_int_naive':   total_variation(mu_naive[interior_idx],   mu_source[interior_idx]),
        # Peak recovery
        'pk_learned': peak_recovery_top1(mu_learned, peak_node, interior_idx),
        'pk_lap':     peak_recovery_top1(mu_lap,     peak_node, interior_idx),
        'pk_naive':   peak_recovery_top1(mu_naive,   peak_node, interior_idx),
        # Peak location (adjacent)
        'adj_learned': peak_location_adjacent(mu_learned, peak_node, interior_idx, R),
        'adj_lap':     peak_location_adjacent(mu_lap,     peak_node, interior_idx, R),
        'adj_naive':   peak_location_adjacent(mu_naive,   peak_node, interior_idx, R),
    }


# ── Visualization helper ──────────────────────────────────────────────────────

def cube_slice(dist, size, z):
    """Extract z-th horizontal slice as (size, size) array."""
    arr = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            arr[x, y] = dist[x * size * size + y * size + z]
    return arr


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-weighting', type=str, default='uniform',
                        choices=['original', 'uniform', 'linear'])
    parser.add_argument('--n-epochs',      type=int,   default=1000)
    parser.add_argument('--hidden-dim',    type=int,   default=64)
    parser.add_argument('--n-layers',      type=int,   default=4)
    parser.add_argument('--lr',            type=float, default=5e-4)
    parser.add_argument('--cube-size',     type=int,   default=4)
    parser.add_argument('--tau-diff',      type=float, default=1.0)
    parser.add_argument('--n-train-dists', type=int,   default=200)
    parser.add_argument('--n-samples',     type=int,   default=10000)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    device = get_device()

    size         = args.cube_size
    N            = size ** 3
    R            = make_cube_graph(size)
    mask         = cube_boundary_mask(size)
    interior_idx = np.where(mask == 0)[0]
    n_boundary   = int(mask.sum())
    n_interior   = N - n_boundary

    print(f"=== Experiment 12: Cube Boundary (v3, diffusion-based) ===")
    print(f"Graph: {size}×{size}×{size} cube ({N} nodes, "
          f"{n_boundary} boundary, {n_interior} interior)")
    print(f"tau_diff={args.tau_diff}, flow start=uniform\n")

    checkpoint_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        checkpoint_dir,
        f'meta_model_ex12v3_cube_{args.loss_weighting}_{args.n_epochs}ep'
        f'_h{args.hidden_dim}_l{args.n_layers}_s{size}_td{args.tau_diff}.pt')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FlexibleConditionalGNNRateMatrixPredictor(
        context_dim=2, hidden_dim=args.hidden_dim, n_layers=args.n_layers)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        print(f"Loaded checkpoint from {ckpt_path}")
        losses = None
    else:
        print(f"Generating {args.n_train_dists} training distributions...")
        train_sources, train_obs = [], []
        for _ in range(args.n_train_dists):
            mu_src, _ = make_interior_peak_dist(N, rng, interior_idx)
            mu_obs    = compute_boundary_observation(mu_src, R, mask, args.tau_diff)
            train_sources.append(mu_src)
            train_obs.append(mu_obs)

        print(f"Building CubeBoundaryDataset ({args.n_samples} samples)...")
        dataset = CubeBoundaryDataset(
            R=R, mask=mask,
            clean_distributions=train_sources,
            boundary_observations=train_obs,
            n_samples=args.n_samples,
            start_from='uniform',
            seed=42)
        print(f"Dataset built: {len(dataset)} samples")

        print(f"Training ({args.n_epochs} epochs, lr={args.lr}, "
              f"hidden={args.hidden_dim}, layers={args.n_layers})...")
        history = train_flexible_conditional(
            model, dataset,
            n_epochs=args.n_epochs,
            batch_size=256,
            lr=args.lr,
            device=device,
            loss_weighting=args.loss_weighting,
        )
        losses = history['losses']
        print(f"Training: initial loss = {losses[0]:.6f}, "
              f"final loss = {losses[-1]:.6f}")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    model.eval()

    # ── Evaluation: 50 held-out test cases ────────────────────────────────────
    rng_eval   = np.random.default_rng(99)
    n_test     = 50
    edge_index = rate_matrix_to_edge_index(R)

    test_results = []
    for _ in range(n_test):
        mu_src, peak_node = make_interior_peak_dist(N, rng_eval, interior_idx)
        result = evaluate_test_case(
            model, R, edge_index, mask, mu_src, peak_node,
            interior_idx, args.tau_diff, device)
        test_results.append(result)

    # ── Console output ─────────────────────────────────────────────────────────
    def M(key): return np.mean([r[key] for r in test_results])
    def S(key): return np.std([r[key]  for r in test_results])

    print(f"\n=== Experiment 12: Cube Boundary (4×4×4, diffusion-based) ===")
    print(f"Graph: {size}×{size}×{size} cube ({N} nodes, "
          f"{n_boundary} boundary, {n_interior} interior)")
    print(f"Diffusion time: tau_diff={args.tau_diff}")

    print(f"\nFull TV:")
    print(f"  Learned:     {M('tv_learned'):.4f} ± {S('tv_learned'):.4f}")
    print(f"  Laplacian:   {M('tv_lap'):.4f} ± {S('tv_lap'):.4f}")
    print(f"  Naive:       {M('tv_naive'):.4f} ± {S('tv_naive'):.4f}")

    print(f"\nInterior TV:")
    print(f"  Learned:     {M('tv_int_learned'):.4f} ± {S('tv_int_learned'):.4f}")
    print(f"  Laplacian:   {M('tv_int_lap'):.4f} ± {S('tv_int_lap'):.4f}")
    print(f"  Naive:       {M('tv_int_naive'):.4f} ± {S('tv_int_naive'):.4f}")

    print(f"\nPeak recovery (top-1 = correct interior node):")
    print(f"  Learned:     {M('pk_learned')*100:.0f}%")
    print(f"  Laplacian:   {M('pk_lap')*100:.0f}%")
    print(f"  Naive:       {M('pk_naive')*100:.0f}%")

    print(f"\nPeak location (top-1 within Manhattan dist 1):")
    print(f"  Learned:     {M('adj_learned')*100:.0f}%")
    print(f"  Laplacian:   {M('adj_lap')*100:.0f}%")
    print(f"  Naive:       {M('adj_naive')*100:.0f}%")

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f'Experiment 12 (v3): Diffusion-Based Boundary Observation\n'
        f'{size}³ cube — {n_boundary} boundary observed / {n_interior} interior hidden'
        f' — τ_diff={args.tau_diff}',
        fontsize=11)
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.38)

    # ── Panel A: Training loss ─────────────────────────────────────────────────
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

    # ── Panel B: Interior node bar chart for one test case ────────────────────
    ax_B = fig.add_subplot(gs[0, 1])
    ex = test_results[0]
    int_nodes = interior_idx
    x_int = np.arange(n_interior)
    width = 0.25
    ax_B.bar(x_int - width, ex['mu_source'][int_nodes],  width,
             color='dimgray',     alpha=0.9, label='True source')
    ax_B.bar(x_int,          ex['mu_learned'][int_nodes], width,
             color='steelblue',   alpha=0.85, label='Learned')
    ax_B.bar(x_int + width,  ex['mu_lap'][int_nodes],     width,
             color='forestgreen', alpha=0.85, label='Laplacian')
    pk = ex['peak_node']
    pk_pos = int(np.where(int_nodes == pk)[0][0])
    ax_B.axvline(pk_pos, color='red', lw=1.2, linestyle='--', alpha=0.6)
    ax_B.set_xticks(x_int)
    ax_B.set_xticklabels([f'int[{i}]' for i in range(n_interior)], fontsize=7)
    ax_B.set_ylabel('Mass')
    ax_B.set_title('B: Interior Node Values (example case)\nRed dashed = true peak')
    ax_B.legend(fontsize=8)
    ax_B.grid(True, alpha=0.3, axis='y')

    # ── Panel C: Full TV comparison ────────────────────────────────────────────
    ax_C = fig.add_subplot(gs[0, 2])
    methods = ['Learned', 'Laplacian', 'Naive']
    colors  = ['steelblue', 'forestgreen', 'salmon']
    tv_vals = [M('tv_learned'), M('tv_lap'), M('tv_naive')]
    tv_err  = [S('tv_learned'), S('tv_lap'), S('tv_naive')]
    bars = ax_C.bar(methods, tv_vals, color=colors, alpha=0.85,
                    yerr=tv_err, capsize=4)
    for bar, val in zip(bars, tv_vals):
        ax_C.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                  f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax_C.set_ylabel('Mean TV (full, 64 nodes)')
    ax_C.set_title('C: Full TV Comparison')
    ax_C.grid(True, alpha=0.3, axis='y')

    # ── Panel D: Interior TV comparison ───────────────────────────────────────
    ax_D = fig.add_subplot(gs[1, 0])
    tv_int_vals = [M('tv_int_learned'), M('tv_int_lap'), M('tv_int_naive')]
    tv_int_err  = [S('tv_int_learned'), S('tv_int_lap'), S('tv_int_naive')]
    bars_D = ax_D.bar(methods, tv_int_vals, color=colors, alpha=0.85,
                      yerr=tv_int_err, capsize=4)
    for bar, val in zip(bars_D, tv_int_vals):
        ax_D.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                  f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax_D.set_ylabel(f'Mean TV (interior, {n_interior} nodes)')
    ax_D.set_title('D: Interior TV Comparison')
    ax_D.grid(True, alpha=0.3, axis='y')

    # ── Panel E: Peak recovery bar chart ──────────────────────────────────────
    ax_E = fig.add_subplot(gs[1, 1])
    pk_acc  = [M('pk_learned')*100,  M('pk_lap')*100,  M('pk_naive')*100]
    adj_acc = [M('adj_learned')*100, M('adj_lap')*100, M('adj_naive')*100]
    x_E = np.arange(len(methods))
    w_E = 0.35
    b1 = ax_E.bar(x_E - w_E/2, pk_acc,  w_E, color=colors, alpha=0.85,
                  label='Exact top-1')
    b2 = ax_E.bar(x_E + w_E/2, adj_acc, w_E, color=colors, alpha=0.45,
                  label='Adjacent (dist ≤ 1)', hatch='//')
    for bar, val in zip(list(b1) + list(b2), pk_acc + adj_acc):
        ax_E.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  f'{val:.0f}%', ha='center', va='bottom', fontsize=7)
    ax_E.set_xticks(x_E)
    ax_E.set_xticklabels(methods)
    ax_E.set_ylim(0, 115)
    ax_E.set_ylabel('Peak recovery (%)')
    ax_E.set_title(f'E: Peak Recovery (top-1 of {n_interior} interior nodes)')
    ax_E.legend(fontsize=8)
    ax_E.grid(True, alpha=0.3, axis='y')

    # ── Panel F: Interior node values for 3 test cases ────────────────────────
    ax_F = fig.add_subplot(gs[1, 2])
    ax_F.axis('off')
    ax_F.set_title('F: Interior Values — 3 Test Cases', pad=12)
    inner_F = gs[1, 2].subgridspec(3, 1, hspace=0.7)

    for row_i, case_idx in enumerate([0, 1, 2]):
        r = test_results[case_idx]
        ax_in = fig.add_subplot(inner_F[row_i])
        x_f = np.arange(n_interior)
        ax_in.bar(x_f - width, r['mu_source'][int_nodes],  width,
                  color='dimgray',     alpha=0.9)
        ax_in.bar(x_f,          r['mu_learned'][int_nodes], width,
                  color='steelblue',   alpha=0.85)
        ax_in.bar(x_f + width,  r['mu_lap'][int_nodes],     width,
                  color='forestgreen', alpha=0.85)
        pk_f   = r['peak_node']
        pk_pos_f = int(np.where(int_nodes == pk_f)[0][0])
        ax_in.axvline(pk_pos_f, color='red', lw=1.0, linestyle='--', alpha=0.7)
        ax_in.set_xticks([])
        ax_in.set_ylabel('Mass', fontsize=6)
        pk_l = peak_recovery_top1(r['mu_learned'], pk_f, interior_idx)
        pk_p = peak_recovery_top1(r['mu_lap'],     pk_f, interior_idx)
        ax_in.set_title(
            f'Case {case_idx+1} — learned correct={bool(pk_l)}, lap correct={bool(pk_p)}',
            fontsize=7)

    # Legend for Panel F
    from matplotlib.patches import Patch
    handles = [Patch(color='dimgray', label='True'),
               Patch(color='steelblue', label='Learned'),
               Patch(color='forestgreen', label='Laplacian')]
    ax_F.legend(handles=handles, loc='upper right', fontsize=8,
                bbox_to_anchor=(1.0, 1.0))

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'ex12_cube_boundary.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
