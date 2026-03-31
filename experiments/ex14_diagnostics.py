"""
Diagnostic script: Investigate Ex14b performance vs cube experiments.

Compares diffusion properties of cortical graph vs 5x5x5 cube,
leadfield conditioning, and baseline performance vs tau.

Output:
  ex14_diffusion_diagnostics.png
  ex14_baseline_vs_tau.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm
from sklearn.linear_model import Lasso

from graph_ot_fm import make_cube_graph, total_variation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_single_peak(N, peak, background=0.2):
    d = np.ones(N) * background / N
    d[peak] += (1.0 - background)
    d /= d.sum()
    return d


def spectral_gap(R):
    """Smallest nonzero |eigenvalue| of rate matrix."""
    evals = np.linalg.eigvals(R)
    evals_re = np.sort(np.real(evals))[::-1]   # descending; 0 is largest
    nonzero = evals_re[evals_re < -1e-8]
    return float(abs(nonzero[0])) if len(nonzero) > 0 else float('nan')


def tv_at_tau(R, source, tau):
    diffused = source @ expm(tau * R)
    return float(total_variation(diffused, source))


def baseline_sloreta(y, A, lam=1e-3):
    M, N = A.shape
    C = A @ A.T + lam * np.eye(M)
    W = A.T @ np.linalg.solve(C, np.eye(M))
    mu_raw = W @ y
    R_diag = np.maximum(np.einsum('ij,ji->i', W, A), 1e-12)
    mu = np.abs(mu_raw) / np.sqrt(R_diag)
    mu = np.clip(mu, 0, None)
    s = mu.sum()
    return mu / s if s > 1e-12 else np.ones(N) / N


def baseline_mne(y, A, lam=1e-3):
    M = A.shape[0]
    mu = A.T @ np.linalg.solve(A @ A.T + lam * np.eye(M), y)
    mu = np.clip(mu, 0, None)
    s = mu.sum()
    return mu / s if s > 1e-12 else np.ones(A.shape[1]) / A.shape[1]


def baseline_lasso(y, A, alpha=1e-4):
    model = Lasso(alpha=alpha, positive=True, max_iter=5000, tol=1e-4)
    model.fit(A, y)
    mu = np.clip(model.coef_, 0, None)
    s = mu.sum()
    return mu / s if s > 1e-12 else np.ones(A.shape[1]) / A.shape[1]


def run_baselines(y, A, lam=1e-3, alpha=1e-4):
    return {
        'sloreta': baseline_sloreta(y, A, lam),
        'mne':     baseline_mne(y, A, lam),
        'lasso':   baseline_lasso(y, A, alpha),
    }


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading ex14_eeg_data.npz...")
data = np.load(os.path.join(ROOT, 'ex14_eeg_data.npz'), allow_pickle=True)
R_cortical = data['R']
A = data['A']
N_cortical = R_cortical.shape[0]

R_cube = make_cube_graph(5)
N_cube = R_cube.shape[0]

print(f"Cortical graph: {N_cortical} nodes")
print(f"Cube graph:     {N_cube} nodes")

# ── 1 & 2. Spectral properties ────────────────────────────────────────────────

print("\n=== Spectral Properties ===")
for name, R, N in [('Cortical', R_cortical, N_cortical),
                   ('Cube 5³',  R_cube,     N_cube)]:
    evals = np.sort(np.real(np.linalg.eigvals(R)))[::-1]
    gap = spectral_gap(R)
    degrees = -np.diag(R)
    adj = R.copy(); np.fill_diagonal(adj, 0)
    n_edges = int((adj > 0).sum()) // 2
    print(f"\n{name}:")
    print(f"  Nodes: {N}, Edges: {n_edges}")
    print(f"  Mean degree: {degrees.mean():.2f}, "
          f"Min: {degrees.min():.2f}, Max: {degrees.max():.2f}")
    print(f"  Spectral gap: {gap:.4f}")
    print(f"  Top 5 eigenvalues: {evals[:5].round(4).tolist()}")
    print(f"  Bottom 5 eigenvalues: {evals[-5:].round(4).tolist()}")

# ── 2. Diffusion amount comparison ────────────────────────────────────────────

print("\n=== Diffusion Amount: TV(source, diffused) ===")
taus_check = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

rng = np.random.default_rng(42)
# Single-peak sources
src_cortical = make_single_peak(N_cortical, peak=10)
src_cube     = make_single_peak(N_cube,     peak=62)   # near center of 5³

print(f"\n{'tau':>6}  {'Cortical TV':>12}  {'Cube TV':>10}")
print(f"  {'---':>4}  {'----------':>12}  {'-------':>10}")
for tau in taus_check:
    tv_c = tv_at_tau(R_cortical, src_cortical, tau)
    tv_k = tv_at_tau(R_cube,     src_cube,     tau)
    print(f"  {tau:>5.2f}  {tv_c:>12.4f}  {tv_k:>10.4f}")

# Find tau on cortical that matches cube TV at various reference points
print("\n  Cube tau=1.0 TV on cortical requires:")
target_tv_cube = tv_at_tau(R_cube, src_cube, 1.0)
print(f"    Cube TV at tau=1.0: {target_tv_cube:.4f}")
# Scan cortical
for tau in np.linspace(0.1, 5.0, 100):
    if tv_at_tau(R_cortical, src_cortical, tau) >= target_tv_cube:
        print(f"    Cortical equivalent tau ≈ {tau:.2f}")
        break

# ── 3. Leadfield diagnostics ──────────────────────────────────────────────────

print("\n=== Leadfield Diagnostics ===")
print(f"  Shape: {A.shape}")
print(f"  Rank:  {np.linalg.matrix_rank(A)}")
print(f"  Condition number: {np.linalg.cond(A):.1f}")
sv = np.linalg.svd(A, compute_uv=False)
print(f"  Singular values: max={sv[0]:.4f}, min={sv[-1]:.6f}, "
      f"median={np.median(sv):.4f}")
col_norms = np.linalg.norm(A, axis=0)
print(f"  Column norm range: [{col_norms.min():.4f}, {col_norms.max():.4f}]")
print(f"  Column norm std/mean: {col_norms.std()/col_norms.mean():.3f} "
      f"(0=uniform, >0.5=high variability)")

# ── 4. Diffusion visualization ────────────────────────────────────────────────

# Pick 3 representative cortical parcels: first Vis, first SomMot, first Limbic
parcel_names = [str(n) for n in data['parcel_names']]
def find_parcel(substr):
    for i, n in enumerate(parcel_names):
        if substr in n:
            return i
    return 0

p_vis    = find_parcel('_Vis_')
p_sommot = find_parcel('_SomMot_')
p_limbic = find_parcel('_Limbic_')
sample_parcels = [('Vis', p_vis), ('SomMot', p_sommat := p_sommot),
                  ('Limbic', p_limbic)]

taus_plot = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9))
fig1.suptitle('Ex14 Diffusion Diagnostics: Cortical Graph vs Cube', fontsize=11)

# Panel A: TV vs tau comparison
ax = axes1[0, 0]
tv_cort = [tv_at_tau(R_cortical, src_cortical, t) for t in taus_plot]
tv_cube_ = [tv_at_tau(R_cube,    src_cube,     t) for t in taus_plot]
ax.plot(taus_plot, tv_cort, 'o-', color='#2166ac', lw=2, label='Cortical (100 nodes)')
ax.plot(taus_plot, tv_cube_, 's--', color='#e6194b', lw=2, label='Cube 5³ (125 nodes)')
ax.axvspan(0.05, 0.20, alpha=0.12, color='#2166ac', label='Current cortical range')
ax.axvspan(0.50, 2.00, alpha=0.12, color='#e6194b', label='Cube training range')
ax.set_xlabel('τ (diffusion time)')
ax.set_ylabel('TV(source, diffused)')
ax.set_title('A: Diffusion Amount vs τ', fontsize=9)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')

# Panel B: Singular value distribution of leadfield
ax = axes1[0, 1]
ax.semilogy(sv, 'o-', color='steelblue', lw=1.5, markersize=3)
ax.axhline(sv[0] / 30341.0, color='red', lw=1, linestyle='--',
           label=f'Effective noise floor\n(cond={np.linalg.cond(A):.0f})')
ax.set_xlabel('Singular value index')
ax.set_ylabel('Singular value')
ax.set_title('B: Leadfield Singular Values', fontsize=9)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# Panel C: TV vs tau for 3 source locations
ax = axes1[1, 0]
colors_p = ['#2166ac', '#3cb44b', '#e6194b']
for (net, pidx), col in zip(sample_parcels, colors_p):
    src = make_single_peak(N_cortical, pidx)
    tvs = [tv_at_tau(R_cortical, src, t) for t in taus_plot]
    ax.plot(taus_plot, tvs, 'o-', color=col, lw=1.5,
            label=f'{net} (parcel {pidx})')
ax.axvspan(0.05, 0.20, alpha=0.12, color='gray', label='Current range')
ax.set_xlabel('τ')
ax.set_ylabel('TV(source, diffused)')
ax.set_title('C: Diffusion by Source Location', fontsize=9)
ax.legend(fontsize=7)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# Panel D: Entropy vs tau
ax = axes1[1, 1]
entropy_uniform = np.log(N_cortical)
for (net, pidx), col in zip(sample_parcels, colors_p):
    src = make_single_peak(N_cortical, pidx)
    ents = []
    for t in taus_plot:
        d = src @ expm(t * R_cortical)
        ents.append(-np.sum(d * np.log(d + 1e-10)) / entropy_uniform * 100)
    ax.plot(taus_plot, ents, 'o-', color=col, lw=1.5, label=net)
ax.axhline(100, color='gray', lw=0.8, linestyle=':')
ax.axvspan(0.05, 0.20, alpha=0.12, color='gray')
ax.set_xlabel('τ')
ax.set_ylabel('Entropy (% of uniform)')
ax.set_title('D: Entropy of Diffused Source vs τ', fontsize=9)
ax.legend(fontsize=7)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

fig1.tight_layout()
path1 = os.path.join(HERE, 'ex14_diffusion_diagnostics.png')
fig1.savefig(path1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"\nSaved {path1}")

# ── 5. Baseline performance vs tau ────────────────────────────────────────────

print("\n=== Baseline Performance vs τ ===")
taus_eval = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
N_test = 50
rng2 = np.random.default_rng(99)

# Tune baselines on tau=0.1 cases
tune_tvs_lam = {}
for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
    tvs = []
    for _ in range(10):
        n_peaks = int(rng2.integers(1, 4))
        peaks = rng2.choice(N_cortical, n_peaks, replace=False)
        src = np.ones(N_cortical) * 0.2 / N_cortical
        w = rng2.dirichlet(np.full(n_peaks, 2.0))
        for p, wi in zip(peaks, w): src[p] += 0.8 * wi
        src /= src.sum()
        mu_d = src @ expm(0.1 * R_cortical)
        y = A @ mu_d + rng2.normal(0, 0.01, A.shape[0])
        tvs.append(total_variation(baseline_mne(y, A, lam), src))
    tune_tvs_lam[lam] = np.mean(tvs)
best_lam = min(tune_tvs_lam, key=tune_tvs_lam.get)

tune_tvs_alpha = {}
for alpha in [1e-4, 1e-3, 1e-2, 1e-1]:
    rng2 = np.random.default_rng(99)
    tvs = []
    for _ in range(10):
        n_peaks = int(rng2.integers(1, 4))
        peaks = rng2.choice(N_cortical, n_peaks, replace=False)
        src = np.ones(N_cortical) * 0.2 / N_cortical
        w = rng2.dirichlet(np.full(n_peaks, 2.0))
        for p, wi in zip(peaks, w): src[p] += 0.8 * wi
        src /= src.sum()
        mu_d = src @ expm(0.1 * R_cortical)
        y = A @ mu_d + rng2.normal(0, 0.01, A.shape[0])
        tvs.append(total_variation(baseline_lasso(y, A, alpha), src))
    tune_tvs_alpha[alpha] = np.mean(tvs)
best_alpha = min(tune_tvs_alpha, key=tune_tvs_alpha.get)
print(f"  Tuned: lam={best_lam}, alpha={best_alpha}")

results_by_tau = {t: {'sloreta': [], 'mne': [], 'lasso': []} for t in taus_eval}
rng3 = np.random.default_rng(77)

print(f"\n  Running {N_test} test cases per τ value...")
for tau in taus_eval:
    for _ in range(N_test):
        n_peaks = int(rng3.integers(1, 4))
        peaks = rng3.choice(N_cortical, n_peaks, replace=False).tolist()
        src = np.ones(N_cortical) * 0.2 / N_cortical
        w = rng3.dirichlet(np.full(n_peaks, 2.0))
        for p, wi in zip(peaks, w): src[p] += 0.8 * wi
        src /= src.sum()
        mu_d = src @ expm(tau * R_cortical)
        y_clean = A @ mu_d
        sp = np.mean(y_clean ** 2)
        np_ = sp / (10 ** (20 / 10))
        y = y_clean + rng3.normal(0, np.sqrt(np_), len(y_clean))
        preds = run_baselines(y, A, lam=best_lam, alpha=best_alpha)
        for k, mu_hat in preds.items():
            results_by_tau[tau][k].append(total_variation(mu_hat, src))

print(f"\n  {'τ':>5}  {'sLORETA':>9}  {'MNE':>9}  {'LASSO':>9}  "
      f"  Δ(sLORETA-LASSO)")
print(f"  {'-'*60}")
means = {}
for tau in taus_eval:
    row = {k: np.mean(v) for k, v in results_by_tau[tau].items()}
    means[tau] = row
    delta = row['sloreta'] - row['lasso']
    print(f"  {tau:>5.2f}  {row['sloreta']:>9.4f}  {row['mne']:>9.4f}  "
          f"{row['lasso']:>9.4f}    {delta:+.4f}")

# ── 6. Recommendation ─────────────────────────────────────────────────────────

print("\n=== Recommendation ===")
tv_cube_1 = tv_at_tau(R_cube, src_cube, 1.0)
print(f"  Cube at tau=1.0: TV(source, diffused) = {tv_cube_1:.4f}")
tv_cube_05 = tv_at_tau(R_cube, src_cube, 0.5)
print(f"  Cube at tau=0.5: TV(source, diffused) = {tv_cube_05:.4f}")

# Find cortical tau matching cube TV at 0.5 and 1.0
for label, target in [('0.5', tv_cube_05), ('1.0', tv_cube_1)]:
    for tau in np.linspace(0.05, 10.0, 500):
        tv = tv_at_tau(R_cortical, src_cortical, tau)
        if tv >= target:
            print(f"  Cortical tau equiv to cube tau={label}: ≈ {tau:.2f} "
                  f"(TV={tv:.4f})")
            break

# Identify at what tau LASSO starts struggling (TV > 0.65)
lasso_threshold_tau = None
for tau in taus_eval:
    if means[tau]['lasso'] > 0.65:
        lasso_threshold_tau = tau
        break
if lasso_threshold_tau:
    print(f"\n  LASSO TV exceeds 0.65 at tau ≈ {lasso_threshold_tau:.2f}")
    print(f"  → Recommended training range: [{lasso_threshold_tau:.2f}, "
          f"{min(lasso_threshold_tau * 3, 5.0):.2f}]")
else:
    print("\n  LASSO stays below 0.65 across all tested τ values.")
    print("  → Need larger τ to make the inverse problem hard enough.")
    print("  → Recommended training range: [0.5, 3.0]")

# ── Figure 2: Baseline TV vs tau ──────────────────────────────────────────────

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Ex14 Baseline Performance vs Diffusion Time (τ)', fontsize=11)

method_styles = [
    ('sloreta', 'sLORETA', '#4363d8', 'o-'),
    ('mne',     'MNE',     '#3cb44b', 's--'),
    ('lasso',   'LASSO',   '#f58231', '^:'),
]

# Panel A: TV vs tau (linear scale)
for key, label, color, ls in method_styles:
    tv_vals = [means[t][key] for t in taus_eval]
    ax2a.plot(taus_eval, tv_vals, ls, color=color, lw=2, markersize=6, label=label)
ax2a.axvspan(0.05, 0.20, alpha=0.15, color='gray', label='Current training range')
ax2a.axhline(0.65, color='black', lw=0.8, linestyle='--', label='TV=0.65 threshold')
ax2a.set_xlabel('τ (diffusion time)')
ax2a.set_ylabel('Mean TV (50 test cases)')
ax2a.set_title('A: Baseline TV vs τ', fontsize=9)
ax2a.legend(fontsize=8)
ax2a.grid(True, alpha=0.3)
ax2a.set_xscale('log')

# Panel B: TV vs tau (show diffusion amount on top)
ax2b_r = ax2b.twinx()
for key, label, color, ls in method_styles:
    tv_vals = [means[t][key] for t in taus_eval]
    ax2b.plot(taus_eval, tv_vals, ls, color=color, lw=2, markersize=6, label=label)
tv_diffusion = [tv_at_tau(R_cortical, src_cortical, t) for t in taus_eval]
ax2b_r.plot(taus_eval, tv_diffusion, 'k--', lw=1.5, alpha=0.5,
            label='TV(src, diffused)')
ax2b_r.set_ylabel('TV(source, diffused)', color='gray', fontsize=8)
ax2b_r.tick_params(axis='y', labelcolor='gray')
ax2b.axvspan(0.05, 0.20, alpha=0.15, color='gray')
ax2b.set_xlabel('τ (diffusion time)')
ax2b.set_ylabel('Baseline TV')
ax2b.set_title('B: Baseline TV + Diffusion Amount', fontsize=9)
lines1, labels1 = ax2b.get_legend_handles_labels()
lines2, labels2 = ax2b_r.get_legend_handles_labels()
ax2b.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
ax2b.grid(True, alpha=0.3)
ax2b.set_xscale('log')

fig2.tight_layout()
path2 = os.path.join(HERE, 'ex14_baseline_vs_tau.png')
fig2.savefig(path2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"\nSaved {path2}")
print("\nDone.")
