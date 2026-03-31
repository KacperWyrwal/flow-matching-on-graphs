"""
Ex18 Precision Diagnosis: Round-trip test for Laplacian sharpening baseline.

Tests whether float32 precision loss explains the nonzero TV of the
exact inverse baseline obs @ expm(-tau * R).

Run: uv run experiments/ex18_precision_check.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm

from experiments.ex17_ot_generalization import generate_training_graphs
from experiments.ex18_source_recovery import generate_source_distribution
from graph_ot_fm import total_variation

TAU_VALUES = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
N_SOURCES_PER_GRAPH = 10
N_GRAPHS = 8  # use first 8 training graphs for speed


def run_diagnostics():
    print("=== Ex18 Precision Diagnosis ===\n", flush=True)

    rng = np.random.default_rng(42)
    graphs = generate_training_graphs(seed=42)[:N_GRAPHS]

    # Collect per-tau statistics
    stats = {tau: {'tv_64_noclip': [], 'tv_64_clip': [], 'tv_32': [],
                   'cond': [], 'n_neg': []}
             for tau in TAU_VALUES}

    for g_idx, (name, R, pos) in enumerate(graphs):
        N = R.shape[0]
        R64 = R.astype(np.float64)

        for s_idx in range(N_SOURCES_PER_GRAPH):
            mu_source = generate_source_distribution(N, pos, rng, R=R).astype(np.float64)

            for tau in TAU_VALUES:
                P = expm(tau * R64)
                P_inv = expm(-tau * R64)
                cond = np.linalg.cond(P)

                # (a) float64, no clipping
                obs64 = mu_source @ P
                n_neg = int(np.sum(obs64 < 0))
                recovered64_nc = obs64 @ P_inv
                s = recovered64_nc.sum()
                if s > 1e-15:
                    recovered64_nc = recovered64_nc / s
                tv_64_nc = total_variation(recovered64_nc, mu_source)

                # (b) float64, with clipping
                obs64_c = np.clip(obs64, 0, None)
                obs64_c /= obs64_c.sum() + 1e-15
                recovered64_c = obs64_c @ P_inv
                recovered64_c = np.clip(recovered64_c, 0, None)
                s = recovered64_c.sum()
                if s > 1e-15:
                    recovered64_c /= s
                tv_64_c = total_variation(recovered64_c, mu_source)

                # (c) float32 round-trip (current code path)
                obs32 = obs64.astype(np.float32)
                obs32 = np.clip(obs32, 0, None)
                obs32 = obs32 / (obs32.sum() + 1e-15)
                recovered32 = obs32.astype(np.float64) @ P_inv
                recovered32 = np.clip(recovered32, 0, None)
                s = recovered32.sum()
                if s > 1e-15:
                    recovered32 /= s
                tv_32 = total_variation(recovered32, mu_source)

                stats[tau]['tv_64_noclip'].append(tv_64_nc)
                stats[tau]['tv_64_clip'].append(tv_64_c)
                stats[tau]['tv_32'].append(tv_32)
                stats[tau]['cond'].append(cond)
                stats[tau]['n_neg'].append(n_neg)

        if (g_idx + 1) % 2 == 0:
            print(f"  Processed {g_idx+1}/{len(graphs)} graphs", flush=True)

    # ── Console table ──
    print(f"\n{'tau':>5s} | {'TV(f64,noclip)':>15s} | {'TV(f64,clip)':>13s} | "
          f"{'TV(f32)':>10s} | {'cond(P)':>12s} | {'n_neg':>6s}", flush=True)
    print("-" * 75, flush=True)
    for tau in TAU_VALUES:
        s = stats[tau]
        print(f"{tau:5.1f} | "
              f"{np.mean(s['tv_64_noclip']):11.2e} ± {np.std(s['tv_64_noclip']):.1e} | "
              f"{np.mean(s['tv_64_clip']):9.2e} ± {np.std(s['tv_64_clip']):.1e} | "
              f"{np.mean(s['tv_32']):7.4f} ± {np.std(s['tv_32']):.2e} | "
              f"{np.mean(s['cond']):12.1f} | "
              f"{np.mean(s['n_neg']):6.1f}", flush=True)

    # ── Diagnosis ──
    max_64_nc = max(np.mean(stats[t]['tv_64_noclip']) for t in TAU_VALUES)
    max_64_c = max(np.mean(stats[t]['tv_64_clip']) for t in TAU_VALUES)
    max_32 = max(np.mean(stats[t]['tv_32']) for t in TAU_VALUES)

    print(f"\nDiagnosis:", flush=True)
    if max_64_nc < 1e-10:
        print("  float64 no-clip: PERFECT (< 1e-10). Inverse is exact.", flush=True)
    else:
        print(f"  float64 no-clip: TV up to {max_64_nc:.2e}. "
              "expm may have numerical issues.", flush=True)
    if max_64_c < 1e-6:
        print("  float64 clip: NEGLIGIBLE. Clipping not a significant source.", flush=True)
    else:
        print(f"  float64 clip: TV up to {max_64_c:.2e}. "
              "Clipping contributes to error.", flush=True)
    if max_32 > 10 * max(max_64_c, 1e-10):
        print(f"  float32: TV up to {max_32:.4f}. "
              "CONFIRMED: float32 cast is the dominant error source.", flush=True)
    else:
        print(f"  float32: TV up to {max_32:.4f}. "
              "Float32 is not the sole culprit.", flush=True)

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: TV vs tau
    for label, key, color, marker in [
        ('float64 no-clip', 'tv_64_noclip', 'tab:green', 'o'),
        ('float64 clip', 'tv_64_clip', 'tab:blue', 's'),
        ('float32', 'tv_32', 'tab:red', '^'),
    ]:
        means = [np.mean(stats[t][key]) for t in TAU_VALUES]
        stds = [np.std(stats[t][key]) for t in TAU_VALUES]
        means = np.array(means)
        stds = np.array(stds)
        ax1.plot(TAU_VALUES, means, f'{marker}-', color=color, label=label, lw=1.5)
        ax1.fill_between(TAU_VALUES, np.clip(means - stds, 1e-16, None),
                         means + stds, alpha=0.15, color=color)
    ax1.set_yscale('log')
    ax1.set_xlabel('τ')
    ax1.set_ylabel('TV (round-trip error)')
    ax1.set_title('(A) Round-trip TV vs τ by dtype')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel B: TV(f32) vs condition number
    all_cond = []
    all_tv32 = []
    all_tau_color = []
    for tau in TAU_VALUES:
        for c, tv in zip(stats[tau]['cond'], stats[tau]['tv_32']):
            all_cond.append(c)
            all_tv32.append(tv)
            all_tau_color.append(tau)
    sc = ax2.scatter(all_cond, all_tv32, c=all_tau_color, cmap='viridis',
                     s=8, alpha=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('Condition number of expm(τR)')
    ax2.set_ylabel('TV (float32 round-trip)')
    ax2.set_title('(B) Float32 error vs conditioning')
    plt.colorbar(sc, ax=ax2, label='τ')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'ex18_precision_check.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out_path}", flush=True)

    return stats


if __name__ == '__main__':
    stats = run_diagnostics()
