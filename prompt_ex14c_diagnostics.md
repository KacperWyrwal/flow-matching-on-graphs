# Diagnostic Task: Investigate Ex14c Data Quality

## Context

Experiment 14c (realistic MNE-simulated EEG) shows the learned model
performing worse than MNE and sLORETA baselines. This is unexpected
given that Ex14b (peaked distributions on the same cortical graph)
showed the learned model beating all baselines. The issue is likely
in the data generation or normalization, not the model.

## What to Check

### 1. Source distribution properties

Load the cached simulation data and examine what the source distributions
actually look like.

```python
# Load simulation cache
train_pairs, test_pairs = load_simulation_cache('ex14c_simulations.npz')

# For 10 random training pairs, print:
for i in [0, 10, 50, 100, 200, 300, 400, 500, 600, 700]:
    mu = train_pairs[i]['mu_source']
    n_active = train_pairs[i]['n_active']
    active = train_pairs[i]['active_parcels']
    
    print(f"\nSample {i}: n_active={n_active}, active_parcels={active}")
    print(f"  Max value: {mu.max():.6f}")
    print(f"  Min value: {mu.min():.6f}")
    print(f"  Top 5 parcels: {np.argsort(mu)[-5:][::-1]} with values {np.sort(mu)[-5:][::-1]}")
    print(f"  Entropy: {-(mu * np.log(mu + 1e-10)).sum():.3f} (max={np.log(100):.3f})")
    print(f"  Effective support (>1/100): {(mu > 0.01).sum()} parcels")
    print(f"  Mass in top 1 parcel: {mu.max():.4f}")
    print(f"  Mass in top 5 parcels: {np.sort(mu)[-5:].sum():.4f}")
    print(f"  Mass in top 10 parcels: {np.sort(mu)[-10:].sum():.4f}")
```

Key question: Is the source distribution peaked (mass concentrated in
a few parcels) or diffuse (spread across many parcels)? If the top
parcel has only 5% of the mass, the distribution is nearly uniform
and very hard to reconstruct.

### 2. Compare to Ex14b distributions

```python
# Generate a few Ex14b-style peaked distributions for comparison
from ex14b_eeg_train import make_cortical_source
rng = np.random.default_rng(42)
for i in range(5):
    mu_peaked, peaks = make_cortical_source(100, 1, rng)
    print(f"\nEx14b peaked: peak at {peaks}, max={mu_peaked.max():.4f}, "
          f"top 5 mass={np.sort(mu_peaked)[-5:].sum():.4f}")

# Direct comparison:
# Ex14b peaked: max ~0.80, top 5 mass ~0.84
# Ex14c realistic: max ~???, top 5 mass ~???
```

### 3. EEG signal quality

```python
# Check the EEG measurements
for i in [0, 50, 100]:
    y = train_pairs[i]['y']
    mu = train_pairs[i]['mu_source']
    
    print(f"\nSample {i}:")
    print(f"  EEG range: [{y.min():.6e}, {y.max():.6e}]")
    print(f"  EEG std: {y.std():.6e}")
    print(f"  EEG SNR estimate: {np.mean(y**2):.6e}")
    
    # Check if EEG is informative: does it vary across channels?
    print(f"  EEG channel variance: {np.var(y):.6e}")
    print(f"  EEG max/min ratio: {abs(y.max()/y.min()) if abs(y.min()) > 1e-15 else 'inf':.2f}")
    
    # Check backprojection: does A^T y look like the source?
    mu_bp = np.clip(A.T @ y, 0, None)
    mu_bp /= mu_bp.sum()
    tv_bp = 0.5 * np.abs(mu_bp - mu).sum()
    print(f"  Backprojection TV: {tv_bp:.4f}")
```

### 4. Leadfield interaction with realistic sources

```python
# For a realistic source, how does the EEG differ from a peaked source
# at the same active parcel?

for i in range(5):
    mu_realistic = train_pairs[i]['mu_source']
    active_parcel = train_pairs[i]['active_parcels'][0]
    
    # Create a peaked version at the same parcel
    mu_peaked = np.ones(100) * 0.002
    mu_peaked[active_parcel] = 0.80
    mu_peaked /= mu_peaked.sum()
    
    # EEG from both
    y_realistic = A @ mu_realistic
    y_peaked = A @ mu_peaked
    
    # Correlation between the two EEG patterns
    corr = np.corrcoef(y_realistic, y_peaked)[0, 1]
    print(f"\nSample {i}: active_parcel={active_parcel}")
    print(f"  EEG correlation (realistic vs peaked): {corr:.4f}")
    print(f"  Realistic source entropy: {-(mu_realistic * np.log(mu_realistic + 1e-10)).sum():.3f}")
    print(f"  Peaked source entropy: {-(mu_peaked * np.log(mu_peaked + 1e-10)).sum():.3f}")
```

### 5. Baseline deep dive

```python
# Run baselines on a few cases and examine the reconstructions
for i in range(5):
    mu_true = test_pairs[i]['mu_source']
    y = test_pairs[i]['y']
    
    mu_mne = baseline_mne_eeg(y, A, lam=best_lam)
    mu_sloreta = baseline_sloreta(y, A, lam=best_lam)
    mu_lasso = baseline_lasso_eeg(y, A, alpha=best_alpha)
    
    tv_mne = 0.5 * np.abs(mu_mne - mu_true).sum()
    tv_slor = 0.5 * np.abs(mu_sloreta - mu_true).sum()
    tv_lasso = 0.5 * np.abs(mu_lasso - mu_true).sum()
    
    print(f"\nTest case {i} (n_active={test_pairs[i]['n_active']}):")
    print(f"  True: top parcel={np.argmax(mu_true)}, max={mu_true.max():.4f}")
    print(f"  MNE:  top parcel={np.argmax(mu_mne)}, max={mu_mne.max():.4f}, TV={tv_mne:.4f}")
    print(f"  sLOR: top parcel={np.argmax(mu_sloreta)}, max={mu_sloreta.max():.4f}, TV={tv_slor:.4f}")
    print(f"  LASSO: top parcel={np.argmax(mu_lasso)}, max={mu_lasso.max():.4f}, TV={tv_lasso:.4f}")
    
    # Does the baseline find the right parcel?
    print(f"  MNE correct parcel: {np.argmax(mu_mne) == np.argmax(mu_true)}")
    print(f"  sLOR correct parcel: {np.argmax(mu_sloreta) == np.argmax(mu_true)}")
```

### 6. Visualization

Generate a single diagnostic figure with 6 panels:

**Panel A:** Histogram of source distribution max values across all training
pairs. Is the max typically 0.8 (peaked) or 0.05 (diffuse)?

**Panel B:** Histogram of source distribution entropy. Compare to 
max entropy (log 100 = 4.6). If most are near max entropy, sources are
too diffuse.

**Panel C:** Scatter plot of source max value vs baseline TV (for MNE).
Does MNE do better on diffuse sources? Does it struggle on peaked ones?

**Panel D:** For 3 example cases, show bar plots of the source distribution
(100 parcels, sorted by value). Shows the shape of realistic sources.

**Panel E:** For the same 3 cases, show the EEG topomap (64 channels as
a bar chart or sorted values). Shows whether the EEG carries spatial info.

**Panel F:** Compare TV distributions: box plots of TV for MNE, sLORETA,
LASSO across all test cases. Shows the spread and whether any method
is consistently good.

Save as `ex14c_diagnostics.png`.

## Expected Findings

Most likely scenario: the realistic sources are too diffuse. With
spatial_extent=10mm on a cortex where parcels are ~15-20mm in diameter,
each source might activate only its own parcel significantly. But if
the parcellation is coarser or the spatial extent is larger, sources
become nearly uniform — and all methods struggle.

Alternative scenario: the EEG normalization differs between how baselines
see it (raw amplitudes) and how the model sees it (potentially 
pre-normalized). Check that the model receives the same y as the baselines.

## Output

Print all diagnostics to console. Save `ex14c_diagnostics.png`.
Based on the findings, recommend:
- Whether to adjust spatial_extent
- Whether normalization needs fixing
- Whether the source distribution format needs changing
- What the model can realistically achieve on this data
