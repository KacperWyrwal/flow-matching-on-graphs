# Experiment 14a: `ex14a_eeg_setup.py` — Data Pipeline & Baseline Validation

## Purpose

Validate the entire EEG data pipeline before training any models.
No neural networks, no training. Just:
1. Build the cortical graph from real anatomy
2. Compute the leadfield matrix from real physics
3. Simulate sources and EEG measurements
4. Run baselines and verify they produce reasonable results
5. Visualize everything on brain surface plots

If anything is wrong (graph disconnected, leadfield degenerate, baselines
nonsensical), we find out here — not after hours of training.

## Step-by-Step Pipeline

### Step 1: Download and setup

```python
import mne
import numpy as np
import os

# Download fsaverage (one-time)
fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)

# Read parcellation
labels = mne.read_labels_from_annot(
    'fsaverage', parc='Schaefer2018_100Parcels_7Networks_order',
    subjects_dir=subjects_dir)
labels = [l for l in labels if 'unknown' not in l.name.lower()]
n_parcels = len(labels)

print(f"Parcellation: {n_parcels} parcels")
print(f"Hemispheres: {sum(1 for l in labels if l.hemi=='lh')} LH, "
      f"{sum(1 for l in labels if l.hemi=='rh')} RH")

# Extract network assignments from label names
# Schaefer labels are like "7Networks_LH_Vis_1"
networks = {}
for i, l in enumerate(labels):
    parts = l.name.split('_')
    # Find the network name (after "7Networks" and hemisphere)
    net = parts[2] if len(parts) > 2 else 'Unknown'
    networks.setdefault(net, []).append(i)
print(f"Networks: {list(networks.keys())}")
for net, parcels in networks.items():
    print(f"  {net}: {len(parcels)} parcels")
```

### Step 2: Build cortical graph

```python
# Setup source space
src = mne.setup_source_space('fsaverage', spacing='oct6',
                              subjects_dir=subjects_dir, add_dist=False)

# Compute parcel centroids
parcel_centroids = np.zeros((n_parcels, 3))
for i, label in enumerate(labels):
    hemi_idx = 0 if label.hemi == 'lh' else 1
    verts = label.vertices
    positions = src[hemi_idx]['rr'][verts]
    parcel_centroids[i] = positions.mean(axis=0)

# Build adjacency from cortical surface mesh
adj = build_parcel_graph(labels, src)  # function from spec

# Convert to rate matrix
R = adj.copy()
np.fill_diagonal(R, -R.sum(axis=1))

# Diagnostics
n_edges = int((adj > 0).sum()) // 2
degrees = adj.sum(axis=1)
print(f"\nGraph diagnostics:")
print(f"  Nodes: {n_parcels}")
print(f"  Edges: {n_edges}")
print(f"  Mean degree: {degrees.mean():.1f}")
print(f"  Min degree: {degrees.min():.0f}")
print(f"  Max degree: {degrees.max():.0f}")
print(f"  Connected components: {n_connected_components(adj)}")

# VERIFY: graph must be connected
assert n_connected_components(adj) == 1, "Graph is disconnected!"
```

### Step 3: Compute leadfield matrix

```python
# Create EEG info with standard montage
montage = mne.channels.make_standard_montage('standard_1020')
# Select a subset of channels (64 is standard for research EEG)
# The standard_1020 montage has ~90 positions; pick 64
ch_names = montage.ch_names[:64]  # or select specific ones
info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types='eeg')
info.set_montage(montage)

# BEM model
bem_model = mne.make_bem_model('fsaverage', subjects_dir=subjects_dir,
                                conductivity=(0.3, 0.006, 0.3))
bem_sol = mne.make_bem_solution(bem_model)

# Forward solution
fwd = mne.make_forward_solution(
    info, trans='fsaverage', src=src, bem=bem_sol,
    eeg=True, meg=False, mindist=5.0)
fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True)

# Extract and parcellate leadfield
leadfield_full = fwd_fixed['sol']['data']  # (n_channels, n_dipoles)
print(f"\nLeadfield diagnostics:")
print(f"  Full leadfield shape: {leadfield_full.shape}")

# Average within parcels
A = np.zeros((len(ch_names), n_parcels))
for i, label in enumerate(labels):
    hemi_idx = 0 if label.hemi == 'lh' else 1
    src_verts = src[hemi_idx]['vertno']
    label_mask = np.isin(src_verts, label.vertices)
    offset = 0 if hemi_idx == 0 else len(src[0]['vertno'])
    indices = np.where(label_mask)[0] + offset
    if len(indices) > 0:
        A[:, i] = leadfield_full[:, indices].mean(axis=1)

# Check for degenerate columns (parcels with no sources)
col_norms = np.linalg.norm(A, axis=0)
zero_cols = np.where(col_norms < 1e-10)[0]
if len(zero_cols) > 0:
    print(f"  WARNING: {len(zero_cols)} parcels have zero leadfield columns!")
    print(f"  Parcel indices: {zero_cols}")

print(f"  Parcellated leadfield shape: {A.shape}")
print(f"  Rank: {np.linalg.matrix_rank(A)}")
print(f"  Condition number: {np.linalg.cond(A):.1f}")
print(f"  Column norm range: [{col_norms.min():.6f}, {col_norms.max():.6f}]")

# Normalize rows (each sensor has unit sensitivity)
for m in range(A.shape[0]):
    norm = np.linalg.norm(A[m])
    if norm > 1e-12:
        A[m] /= norm
```

### Step 4: Simulate test sources and EEG

```python
rng = np.random.default_rng(42)

# Create 5 test cases with known sources
test_cases = []
for i in range(5):
    n_peaks = int(rng.integers(1, 4))
    mu_source, peak_parcels = make_cortical_source(n_parcels, n_peaks, rng)
    tau_diff = 1.0
    mu_diffused = mu_source @ expm(tau_diff * R)
    y_clean = A @ mu_diffused
    
    # Add noise at SNR=20dB
    signal_power = np.mean(y_clean ** 2)
    noise_power = signal_power / (10 ** (20 / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(y_clean))
    y_noisy = y_clean + noise
    
    test_cases.append({
        'mu_source': mu_source,
        'peak_parcels': peak_parcels,
        'peak_names': [labels[p].name for p in peak_parcels],
        'peak_networks': [labels[p].name.split('_')[2] for p in peak_parcels],
        'mu_diffused': mu_diffused,
        'y_clean': y_clean,
        'y_noisy': y_noisy,
        'tau_diff': tau_diff,
    })

print(f"\nTest cases generated:")
for i, tc in enumerate(test_cases):
    print(f"  Case {i}: {len(tc['peak_parcels'])} peaks at "
          f"{tc['peak_names']}, networks: {tc['peak_networks']}")
    print(f"    EEG range: [{tc['y_clean'].min():.6f}, {tc['y_clean'].max():.6f}]")
```

### Step 5: Run baselines

```python
# Tune baselines
best_lam, best_alpha = tune_baselines_eeg(test_cases[:2], A)
print(f"\nBaseline tuning:")
print(f"  Best MNE lambda: {best_lam}")
print(f"  Best LASSO alpha: {best_alpha}")

# Run all baselines on all test cases
for i, tc in enumerate(test_cases):
    y = tc['y_noisy']
    mu_true = tc['mu_source']
    
    mu_mne = baseline_mne_eeg(y, A, lam=best_lam)
    mu_sloreta = baseline_sloreta(y, A, lam=best_lam)
    mu_lasso = baseline_lasso_eeg(y, A, alpha=best_alpha)
    mu_bp = baseline_backproj_eeg(y, A)
    
    tc['mu_mne'] = mu_mne
    tc['mu_sloreta'] = mu_sloreta
    tc['mu_lasso'] = mu_lasso
    tc['mu_bp'] = mu_bp
    
    tc['tv_mne'] = total_variation(mu_mne, mu_true)
    tc['tv_sloreta'] = total_variation(mu_sloreta, mu_true)
    tc['tv_lasso'] = total_variation(mu_lasso, mu_true)
    tc['tv_bp'] = total_variation(mu_bp, mu_true)
    
    tc['pk_mne'] = peak_recovery_topk(mu_mne, tc['peak_parcels'])
    tc['pk_sloreta'] = peak_recovery_topk(mu_sloreta, tc['peak_parcels'])
    tc['pk_lasso'] = peak_recovery_topk(mu_lasso, tc['peak_parcels'])
    tc['pk_bp'] = peak_recovery_topk(mu_bp, tc['peak_parcels'])

# Print summary
print(f"\nBaseline results:")
print(f"{'Case':>6} {'MNE TV':>8} {'sLOR TV':>8} {'LASSO TV':>8} {'BP TV':>8} "
      f"{'MNE pk':>7} {'sLOR pk':>8} {'LASSO pk':>8}")
for i, tc in enumerate(test_cases):
    print(f"{i:>6} {tc['tv_mne']:>8.3f} {tc['tv_sloreta']:>8.3f} "
          f"{tc['tv_lasso']:>8.3f} {tc['tv_bp']:>8.3f} "
          f"{tc['pk_mne']*100:>6.0f}% {tc['pk_sloreta']*100:>7.0f}% "
          f"{tc['pk_lasso']*100:>7.0f}%")
```

### Step 6: Visualization

```python
# Plot 1: Graph on brain surface
# Show parcellation with edges overlaid on inflated cortical surface

# Plot 2: Leadfield patterns
# Show topographic maps for a few parcels (what does activity at
# parcel X look like on the scalp?)

# Plot 3: Test case reconstructions (THE KEY DIAGNOSTIC)
# For each test case: 5 brain views (true, sLORETA, MNE, LASSO, backproj)
# Use nilearn.plotting.plot_surf or mne.viz

# Plot 4: EEG topographic maps
# Show the simulated scalp EEG pattern for each test case
# mne.viz.plot_topomap

# Plot 5: Bar chart of baseline metrics across test cases
```

## Output

### Console
```
=== Experiment 14a: EEG Data Pipeline Validation ===

Parcellation: 100 parcels (50 LH, 50 RH)
Networks: Vis(14), SomMot(14), DorsAttn(14), SalVentAttn(16),
          Limbic(10), Cont(14), Default(18)

Graph:
  Nodes: 100, Edges: XXX, Mean degree: X.X
  Connected: Yes

Leadfield:
  Shape: (64, 100), Rank: XX
  Condition number: XXX.X

Baselines (mean across 5 test cases):
  sLORETA: TV=X.XXX, peak=XX%
  MNE:     TV=X.XXX, peak=XX%
  LASSO:   TV=X.XXX, peak=XX%
  Backproj: TV=X.XXX, peak=XX%
```

### Figures
- `ex14a_graph_diagnostics.png` — graph structure, degree distribution
- `ex14a_leadfield_patterns.png` — topographic maps for sample parcels
- `ex14a_baseline_reconstructions.png` — brain surface plots of baselines
- `ex14a_baseline_metrics.png` — bar charts of TV and peak recovery

### Saved data (for Script 2)
```python
np.savez('ex14_eeg_data.npz',
    R=R,                      # (100, 100) rate matrix
    A=A,                      # (64, 100) leadfield
    parcel_centroids=parcel_centroids,  # (100, 3)
    parcel_names=[l.name for l in labels],
    network_assignments=network_assignments,  # (100,) int
    adj=adj,                  # (100, 100) adjacency
)
```

This `.npz` file is all Script 2 needs — no MNE dependency at training time.

## Dependencies

```
mne>=1.6
nilearn>=0.10
```

## Checks Before Proceeding to Script 2

Before running Script 2, verify:
- [ ] Graph is connected (single component)
- [ ] Leadfield has full column rank (or close)
- [ ] No parcels have zero leadfield columns
- [ ] Baselines produce non-trivial results (TV < 0.9, peak > 0%)
- [ ] Brain surface visualizations look correct (parcels in right locations)
- [ ] sLORETA is the strongest baseline (expected from literature)
