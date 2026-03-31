"""
Diagnostic: Investigate Ex14c data quality.

Loads cached simulation data and analyses source distribution properties,
EEG signal quality, leadfield interactions, and baseline performance.

Run: uv run experiments/ex14c_diagnostics.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy

from graph_ot_fm import total_variation
from experiments.ex14c_realistic import (
    load_simulation_cache,
    baseline_mne_eeg,
    baseline_sloreta,
    baseline_lasso_eeg,
    baseline_backproj_eeg,
    compute_electrode_parcels,
    tune_baselines_eeg,
)
from experiments.ex14b_eeg_train import make_cortical_source

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, 'data')


def find_sim_cache():
    """Find any ex14c simulation cache file."""
    pattern = os.path.join(DATA_DIR, 'ex14c_sims_*.npz')
    hits = glob.glob(pattern)
    if not hits:
        raise FileNotFoundError(
            f"No ex14c simulation cache found in {DATA_DIR}/.\n"
            "Run: uv run experiments/ex14c_realistic.py --n-simulations 200 --n-test 50")
    hits.sort()
    return hits[0]


def entropy_ratio(dist):
    p = np.clip(dist, 1e-12, None)
    p = p / p.sum()
    return (-np.sum(p * np.log(p))) / np.log(len(p))


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    npz_path = os.path.join(ROOT, 'ex14_eeg_data.npz')
    data = np.load(npz_path, allow_pickle=True)
    A = data['A']
    N = A.shape[1]

    cache_path = find_sim_cache()
    print(f"Loading simulation cache: {os.path.basename(cache_path)}")
    train_pairs, test_pairs = load_simulation_cache(cache_path)
    print(f"  Train: {len(train_pairs)} pairs, Test: {len(test_pairs)} pairs")

    electrode_parcels = compute_electrode_parcels(A)

    # Tune baselines on first 20 train pairs
    best_lam, best_alpha = tune_baselines_eeg(train_pairs[:20], A)
    print(f"  Best MNE lambda: {best_lam}, LASSO alpha: {best_alpha}")

    # ── Section 1: Source distribution properties ─────────────────────────────
    print("\n=== 1. Source Distribution Properties ===")
    sample_indices = [0, 10, 50, 100, 200, 300, 400, 500, 600, 700]
    sample_indices = [i for i in sample_indices if i < len(train_pairs)]

    all_train_mu = np.array([p['mu_source'] for p in train_pairs])
    all_max      = all_train_mu.max(axis=1)
    all_top5     = np.sort(all_train_mu, axis=1)[:, -5:].sum(axis=1)
    all_top10    = np.sort(all_train_mu, axis=1)[:, -10:].sum(axis=1)
    all_ent_ratio = np.array([entropy_ratio(mu) for mu in all_train_mu])
    all_eff_sup  = (all_train_mu > 0.01).sum(axis=1)

    print(f"\nAggregate over {len(train_pairs)} training samples:")
    print(f"  Max value:          mean={all_max.mean():.4f}, "
          f"std={all_max.std():.4f}, min={all_max.min():.4f}, max={all_max.max():.4f}")
    print(f"  Top-5 mass:         mean={all_top5.mean():.4f}, std={all_top5.std():.4f}")
    print(f"  Top-10 mass:        mean={all_top10.mean():.4f}, std={all_top10.std():.4f}")
    print(f"  Entropy ratio:      mean={all_ent_ratio.mean():.4f}, std={all_ent_ratio.std():.4f}")
    print(f"  Eff. support >1%:   mean={all_eff_sup.mean():.1f}, std={all_eff_sup.std():.1f}")

    print("\nPer-sample breakdown:")
    for i in sample_indices:
        mu       = train_pairs[i]['mu_source']
        n_active = train_pairs[i]['n_active']
        active   = train_pairs[i]['active_parcels']
        top5_idx = np.argsort(mu)[-5:][::-1]
        print(f"\n  Sample {i:3d}: n_active={n_active}, active_parcels={active}")
        print(f"    Max: {mu.max():.6f}  Min: {mu.min():.6f}")
        print(f"    Top-5 parcels: {top5_idx.tolist()} → vals {mu[top5_idx].round(5).tolist()}")
        print(f"    Entropy ratio: {entropy_ratio(mu):.3f}  (0=peaked, 1=uniform)")
        print(f"    Eff. support:  {(mu > 0.01).sum()} parcels (>{100/N:.1f}%)")
        print(f"    Top-1 mass:    {mu.max():.4f}")
        print(f"    Top-5 mass:    {np.sort(mu)[-5:].sum():.4f}")
        print(f"    Top-10 mass:   {np.sort(mu)[-10:].sum():.4f}")

    # ── Section 2: Compare to Ex14b peaked distributions ─────────────────────
    print("\n=== 2. Comparison to Ex14b Peaked Distributions ===")
    rng = np.random.default_rng(42)
    peaked_maxes  = []
    peaked_top5   = []
    peaked_ent    = []
    for _ in range(50):
        mu_p, _ = make_cortical_source(N, 1, rng)
        peaked_maxes.append(mu_p.max())
        peaked_top5.append(np.sort(mu_p)[-5:].sum())
        peaked_ent.append(entropy_ratio(mu_p))

    print(f"\nEx14b peaked (1-peak, n=50):")
    print(f"  Max value:     mean={np.mean(peaked_maxes):.4f}")
    print(f"  Top-5 mass:    mean={np.mean(peaked_top5):.4f}")
    print(f"  Entropy ratio: mean={np.mean(peaked_ent):.4f}")
    print(f"\nEx14c realistic (all training):")
    print(f"  Max value:     mean={all_max.mean():.4f}")
    print(f"  Top-5 mass:    mean={all_top5.mean():.4f}")
    print(f"  Entropy ratio: mean={all_ent_ratio.mean():.4f}")
    ratio = all_ent_ratio.mean() / np.mean(peaked_ent)
    print(f"\n  → Realistic sources are {ratio:.1f}× more diffuse (entropy) than peaked")

    # ── Section 3: EEG signal quality ─────────────────────────────────────────
    print("\n=== 3. EEG Signal Quality ===")
    for i in [0, 50, 100]:
        if i >= len(train_pairs):
            continue
        y  = train_pairs[i]['y']
        mu = train_pairs[i]['mu_source']
        mu_bp = np.clip(A.T @ y, 0, None)
        s = mu_bp.sum()
        mu_bp /= s if s > 1e-12 else 1.0
        tv_bp = total_variation(mu_bp, mu)
        maxmin = abs(y.max() / y.min()) if abs(y.min()) > 1e-15 else float('inf')
        print(f"\n  Sample {i}:")
        print(f"    EEG range:       [{y.min():.4e}, {y.max():.4e}]")
        print(f"    EEG std:         {y.std():.4e}")
        print(f"    EEG max/min:     {maxmin:.2f}")
        print(f"    Backprojection TV: {tv_bp:.4f}")

    # ── Section 4: Leadfield interaction ──────────────────────────────────────
    print("\n=== 4. Leadfield Interaction: Realistic vs Peaked Source ===")
    for i in range(5):
        if i >= len(train_pairs):
            continue
        mu_r      = train_pairs[i]['mu_source']
        actives   = train_pairs[i]['active_parcels']
        ap        = actives[0]
        mu_p      = np.ones(N) * 0.002
        mu_p[ap]  = 0.80
        mu_p     /= mu_p.sum()
        y_r       = A @ mu_r
        y_p       = A @ mu_p
        corr      = np.corrcoef(y_r, y_p)[0, 1]
        print(f"\n  Sample {i}: active_parcel={ap}")
        print(f"    EEG corr (realistic vs peaked): {corr:.4f}")
        print(f"    Realistic entropy:  {entropy_ratio(mu_r):.3f}")
        print(f"    Peaked entropy:     {entropy_ratio(mu_p):.3f}")

    # ── Section 5: Baseline deep-dive on test set ─────────────────────────────
    print("\n=== 5. Baseline Deep-Dive on Test Set ===")
    all_tv_mne     = []
    all_tv_slor    = []
    all_tv_lasso   = []
    all_tv_bp      = []

    for pair in test_pairs:
        mu_t = pair['mu_source']
        y    = pair['y']
        all_tv_mne.append(  total_variation(baseline_mne_eeg(y, A, best_lam),   mu_t))
        all_tv_slor.append( total_variation(baseline_sloreta(y, A, best_lam),   mu_t))
        all_tv_lasso.append(total_variation(baseline_lasso_eeg(y, A, best_alpha), mu_t))
        all_tv_bp.append(   total_variation(baseline_backproj_eeg(y, A),          mu_t))

    print(f"\nBaseline TV over {len(test_pairs)} test cases:")
    print(f"  {'Method':12s}  {'Mean':>7s}  {'Std':>7s}  {'Min':>7s}  {'Max':>7s}")
    for name, vals in [('MNE', all_tv_mne), ('sLORETA', all_tv_slor),
                       ('LASSO', all_tv_lasso), ('Backproj', all_tv_bp)]:
        v = np.array(vals)
        print(f"  {name:12s}  {v.mean():.4f}   {v.std():.4f}  "
              f"{v.min():.4f}   {v.max():.4f}")

    print("\nFirst 5 test cases detail:")
    for i in range(min(5, len(test_pairs))):
        mu_t  = test_pairs[i]['mu_source']
        y     = test_pairs[i]['y']
        mu_m  = baseline_mne_eeg(y, A, best_lam)
        mu_s  = baseline_sloreta(y, A, best_lam)
        mu_l  = baseline_lasso_eeg(y, A, best_alpha)
        tv_m  = total_variation(mu_m, mu_t)
        tv_s  = total_variation(mu_s, mu_t)
        tv_l  = total_variation(mu_l, mu_t)
        true_top = np.argmax(mu_t)
        print(f"\n  Test {i} (n_active={test_pairs[i]['n_active']}):")
        print(f"    True:  top_parcel={true_top}, max={mu_t.max():.4f}")
        print(f"    MNE:   top_parcel={np.argmax(mu_m)}, max={mu_m.max():.4f}, "
              f"TV={tv_m:.4f}, correct={np.argmax(mu_m)==true_top}")
        print(f"    sLOR:  top_parcel={np.argmax(mu_s)}, max={mu_s.max():.4f}, "
              f"TV={tv_s:.4f}, correct={np.argmax(mu_s)==true_top}")
        print(f"    LASSO: top_parcel={np.argmax(mu_l)}, max={mu_l.max():.4f}, "
              f"TV={tv_l:.4f}, correct={np.argmax(mu_l)==true_top}")

    # ── Section 6: Figure ─────────────────────────────────────────────────────
    print("\nGenerating diagnostic figure...")
    all_test_mu = np.array([p['mu_source'] for p in test_pairs])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Ex14c Diagnostics: Source Distribution & Baseline Quality', fontsize=11)

    # Panel A: histogram of source max values
    ax = axes[0, 0]
    peaked_maxes_arr = np.array(peaked_maxes)
    shared_bins = np.linspace(0, 1, 31)
    ax.hist(all_max,          bins=shared_bins, alpha=0.7, color='#2166ac', label='Ex14c realistic')
    ax.hist(peaked_maxes_arr, bins=shared_bins, alpha=0.7, color='#f58231', label='Ex14b peaked')
    ax.axvline(all_max.mean(),          color='#2166ac', lw=1.5, ls='--')
    ax.axvline(peaked_maxes_arr.mean(), color='#f58231', lw=1.5, ls='--')
    ax.set_xlabel('Max value of source distribution')
    ax.set_ylabel('Count')
    ax.set_title('A: Source Peak Concentration', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: histogram of entropy ratio
    ax = axes[0, 1]
    peaked_ent_arr = np.array(peaked_ent)
    shared_ent_bins = np.linspace(0, 1, 31)
    ax.hist(all_ent_ratio,  bins=shared_ent_bins, alpha=0.7, color='#2166ac', label='Ex14c realistic')
    ax.hist(peaked_ent_arr, bins=shared_ent_bins, alpha=0.7, color='#f58231', label='Ex14b peaked')
    ax.axvline(all_ent_ratio.mean(),  color='#2166ac', lw=1.5, ls='--',
               label=f'mean={all_ent_ratio.mean():.3f}')
    ax.axvline(peaked_ent_arr.mean(), color='#f58231', lw=1.5, ls='--',
               label=f'mean={peaked_ent_arr.mean():.3f}')
    ax.set_xlabel('Entropy ratio (1 = uniform)')
    ax.set_ylabel('Count')
    ax.set_title('B: Source Entropy (0=peaked, 1=uniform)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel C: scatter source max vs MNE TV on test set
    ax = axes[0, 2]
    test_max  = all_test_mu.max(axis=1)
    sc = ax.scatter(test_max, all_tv_mne, alpha=0.5, s=20, c='#3cb44b',
                    label='MNE')
    ax.scatter(test_max, all_tv_slor, alpha=0.5, s=20, c='#4363d8',
               label='sLORETA', marker='^')
    # Trend line for MNE
    z = np.polyfit(test_max, all_tv_mne, 1)
    xr = np.linspace(test_max.min(), test_max.max(), 50)
    ax.plot(xr, np.polyval(z, xr), 'g--', lw=1.5, alpha=0.8)
    ax.set_xlabel('Source max value')
    ax.set_ylabel('Baseline TV')
    ax.set_title('C: Source Concentration vs Baseline TV', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D: bar plots of 3 example source distributions
    ax = axes[1, 0]
    colors_d = ['#2166ac', '#f58231', '#e6194b']
    n_show   = min(3, len(test_pairs))
    x        = np.arange(N)
    for k, (idx, color) in enumerate(zip([0, min(5, len(test_pairs)-1),
                                           min(10, len(test_pairs)-1)], colors_d)):
        mu_sorted = np.sort(test_pairs[idx]['mu_source'])[::-1]
        ent = entropy_ratio(test_pairs[idx]['mu_source'])
        ax.plot(np.arange(N), mu_sorted, color=color, lw=1.2, alpha=0.8,
                label=f"case {idx} (ent={ent:.2f})")
    ax.axhline(1.0 / N, color='grey', lw=1, ls='--', label=f'uniform (1/{N})')
    ax.set_xlabel('Parcel rank')
    ax.set_ylabel('Probability')
    ax.set_title('D: Source Distribution Shape (sorted)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Panel E: EEG channel values for same 3 cases
    ax = axes[1, 1]
    for k, (idx, color) in enumerate(zip([0, min(5, len(test_pairs)-1),
                                           min(10, len(test_pairs)-1)], colors_d)):
        y_case   = test_pairs[idx]['y']
        y_sorted = np.sort(np.abs(y_case))[::-1]
        ax.plot(y_sorted, color=color, lw=1.2, alpha=0.8, label=f"case {idx}")
    ax.set_xlabel('Channel rank (by |amplitude|)')
    ax.set_ylabel('|EEG amplitude|')
    ax.set_title('E: EEG Channel Amplitudes (sorted)', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel F: TV box plots
    ax = axes[1, 2]
    box_data   = [all_tv_mne, all_tv_slor, all_tv_lasso, all_tv_bp]
    box_labels = ['MNE', 'sLORETA', 'LASSO', 'Backproj']
    box_colors = ['#3cb44b', '#4363d8', '#f58231', '#e6194b']
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 1.5})
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Total Variation')
    ax.set_title(f'F: Baseline TV Distribution ({len(test_pairs)} test cases)', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    out_path = os.path.join(HERE, 'ex14c_diagnostics.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")

    # ── Recommendation ────────────────────────────────────────────────────────
    print("\n=== Recommendation ===")
    print(f"\n  Source distribution summary:")
    print(f"    Ex14c realistic: max={all_max.mean():.3f}, "
          f"entropy_ratio={all_ent_ratio.mean():.3f}, "
          f"eff_support={all_eff_sup.mean():.0f} parcels")
    print(f"    Ex14b peaked:    max={np.mean(peaked_maxes):.3f}, "
          f"entropy_ratio={np.mean(peaked_ent):.3f}, eff_support=~5 parcels")
    print(f"\n  Baseline performance on test set:")
    print(f"    MNE TV = {np.mean(all_tv_mne):.3f}")
    print(f"    sLORETA TV = {np.mean(all_tv_slor):.3f}")
    print(f"    LASSO TV = {np.mean(all_tv_lasso):.3f}")

    if all_ent_ratio.mean() > 0.75:
        print(f"\n  ISSUE: Sources too diffuse (entropy ratio={all_ent_ratio.mean():.3f} > 0.75)")
        print("  The realistic sources are nearly uniform across parcels. This makes")
        print("  reconstruction trivially hard — even a uniform prediction has low TV.")
        print("\n  Recommendations:")
        print("  1. Decrease spatial_extent: try --spatial-extent 3.0 or 5.0")
        print("     (10mm spread on ~15-20mm parcels creates nearly uniform distributions)")
        print("  2. Or: Mix peaked (Ex14b-style) and realistic sources during training")
        print("  3. Or: Use vertex-level evaluation (not parcel-level) where spatial")
        print("     resolution is higher")
        target_ext = 3.0 if all_ent_ratio.mean() > 0.85 else 5.0
        print(f"\n  Suggested: --spatial-extent {target_ext}")
    elif all_max.mean() < 0.1:
        print(f"\n  ISSUE: Source max too low (mean={all_max.mean():.3f} < 0.10)")
        print("  Sources are too spread out. Decrease spatial_extent.")
    else:
        print("\n  Sources look reasonably peaked.")
        if np.mean(all_tv_mne) > 0.80:
            print("  Baselines are already struggling — good regime for learning.")
        else:
            print(f"  Baselines achieve TV={np.mean(all_tv_mne):.3f} (MNE) — "
                  "model may have limited room to improve.")

    print("\nDone.")


if __name__ == '__main__':
    main()
