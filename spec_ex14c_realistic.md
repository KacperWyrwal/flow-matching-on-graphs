# Experiment 14c: Realistic Simulated EEG with MNE

## Motivation

Ex14b used hand-crafted peaked distributions as sources. Real brain activity
has spatial extent (cortical patches, not point sources), realistic amplitude
distributions, and structured noise. Training and testing on MNE-simulated
realistic data removes the mismatch between synthetic training data and
what the model would encounter in practice.

## Data Generation Pipeline

### Step 1: Setup (reuse from Ex14a)

```python
import mne
import numpy as np

# Load precomputed data from Ex14a
data = np.load('ex14_eeg_data.npz', allow_pickle=True)
R = data['R']              # (100, 100) rate matrix
A = data['A']              # (64, 100) parcellated leadfield
adj = data['adj']          # (100, 100) adjacency
parcel_names = data['parcel_names']

# Also need full source space and labels for MNE simulation
fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)
src = mne.setup_source_space('fsaverage', spacing='oct6',
                              subjects_dir=subjects_dir)
labels = mne.read_labels_from_annot(
    'fsaverage', parc='Schaefer2018_100Parcels_7Networks_order',
    subjects_dir=subjects_dir)
labels = [l for l in labels if 'unknown' not in l.name.lower()]
```

### Step 2: Generate realistic source activations

```python
def generate_realistic_source(labels, src, n_active=None, rng=None,
                               spatial_extent=10.0, sfreq=256,
                               duration=0.3):
    """
    Generate one realistic source activation using MNE.
    
    Args:
        labels: list of parcellation labels
        src: source space
        n_active: number of active regions (1-3, random if None)
        rng: numpy random generator
        spatial_extent: spread of activation in mm (default 10mm)
        sfreq: sampling frequency
        duration: duration in seconds
    
    Returns:
        stc: SourceEstimate with realistic activation
        active_labels: list of labels that were activated
        peak_time: time of peak activation (seconds)
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_active is None:
        n_active = int(rng.integers(1, 4))
    
    n_parcels = len(labels)
    active_indices = rng.choice(n_parcels, size=n_active, replace=False)
    active_labels = [labels[i] for i in active_indices]
    
    n_times = int(sfreq * duration)
    times = np.arange(n_times) / sfreq
    
    # Use mne.simulation.simulate_stc or build manually
    # Each active region gets a time course:
    #   - Evoked response: Gaussian envelope peaking at a random time
    #   - Or oscillatory: sinusoid with envelope
    
    stc_data_lh = np.zeros((len(src[0]['vertno']), n_times))
    stc_data_rh = np.zeros((len(src[1]['vertno']), n_times))
    
    peak_time = float(rng.uniform(0.05, duration - 0.05))
    
    for label in active_labels:
        hemi_idx = 0 if label.hemi == 'lh' else 1
        src_verts = src[hemi_idx]['vertno']
        
        # Find vertices in this label that are in the source space
        label_mask = np.isin(src_verts, label.vertices)
        label_src_indices = np.where(label_mask)[0]
        
        if len(label_src_indices) == 0:
            continue
        
        # Time course: Gaussian envelope
        amplitude = float(rng.uniform(5e-9, 50e-9))  # 5-50 nAm
        sigma_t = float(rng.uniform(0.02, 0.08))  # 20-80ms width
        time_course = amplitude * np.exp(
            -0.5 * ((times - peak_time) / sigma_t) ** 2)
        
        # Spatial pattern: Gaussian falloff from label center
        center_pos = src[hemi_idx]['rr'][src_verts[label_src_indices]].mean(axis=0)
        all_positions = src[hemi_idx]['rr'][src_verts]
        distances = np.linalg.norm(all_positions - center_pos, axis=1)
        sigma_s = spatial_extent / 1000.0  # mm to meters
        spatial_pattern = np.exp(-0.5 * (distances / sigma_s) ** 2)
        
        # Combine: spatial pattern * time course
        if hemi_idx == 0:
            stc_data_lh += np.outer(spatial_pattern, time_course)
        else:
            stc_data_rh += np.outer(spatial_pattern, time_course)
    
    # Create SourceEstimate
    stc = mne.SourceEstimate(
        np.vstack([stc_data_lh, stc_data_rh]),
        vertices=[src[0]['vertno'], src[1]['vertno']],
        tmin=0, tstep=1.0/sfreq)
    
    return stc, active_labels, active_indices, peak_time


def parcellate_stc(stc, labels, src):
    """
    Average source activation within each parcel to get (n_parcels,)
    distribution at each time point.
    
    Returns: (n_parcels, n_times) array
    """
    n_parcels = len(labels)
    n_times = stc.data.shape[1]
    parcel_tc = np.zeros((n_parcels, n_times))
    
    for i, label in enumerate(labels):
        hemi_idx = 0 if label.hemi == 'lh' else 1
        src_verts = src[hemi_idx]['vertno']
        label_mask = np.isin(src_verts, label.vertices)
        offset = 0 if hemi_idx == 0 else len(src[0]['vertno'])
        indices = np.where(label_mask)[0] + offset
        
        if len(indices) > 0:
            parcel_tc[i] = np.abs(stc.data[indices]).mean(axis=0)
    
    return parcel_tc


def stc_to_distribution(parcel_tc, time_idx):
    """
    Convert parcel time course at a specific time point to a
    normalized distribution.
    
    Takes absolute values (power) and normalizes to sum to 1.
    """
    values = parcel_tc[:, time_idx]
    values = np.clip(values, 0, None)
    s = values.sum()
    if s > 1e-12:
        return values / s
    return np.ones(len(values)) / len(values)
```

### Step 3: Simulate EEG from source

```python
def simulate_eeg_from_stc(stc, fwd, info, snr_db=20, rng=None):
    """
    Simulate EEG from a source estimate using MNE's forward model.
    
    Returns: (n_channels, n_times) EEG data
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Project through forward model
    leadfield = fwd['sol']['data']
    eeg_clean = leadfield @ stc.data  # (n_channels, n_times)
    
    # Add realistic noise
    if snr_db is not None:
        signal_power = np.mean(eeg_clean ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = rng.normal(0, np.sqrt(noise_power), eeg_clean.shape)
        eeg = eeg_clean + noise
    else:
        eeg = eeg_clean
    
    return eeg
```

### Step 4: Generate training dataset

```python
def generate_training_data(labels, src, fwd, info, A, R,
                            n_simulations=1000, snr_db=20, seed=42):
    """
    Generate training pairs from realistic MNE simulations.
    
    For each simulation:
    1. Generate realistic source activation (1-3 active regions)
    2. Simulate EEG
    3. Extract the peak time point
    4. Parcellate source at peak time -> distribution
    5. Get EEG at peak time -> sensor readings
    
    Returns: list of dicts with keys:
        mu_source: (100,) normalized source distribution at peak
        y: (64,) EEG measurement at peak
        active_parcels: list of active parcel indices
        n_active: number of active regions
        peak_time: time of peak activation
    """
    rng = np.random.default_rng(seed)
    pairs = []
    
    for i in range(n_simulations):
        stc, active_labels, active_indices, peak_time = \
            generate_realistic_source(labels, src, rng=rng)
        
        eeg = simulate_eeg_from_stc(stc, fwd, info, snr_db=snr_db, rng=rng)
        
        parcel_tc = parcellate_stc(stc, labels, src)
        peak_sample = int(peak_time * 256)  # assuming sfreq=256
        peak_sample = min(peak_sample, parcel_tc.shape[1] - 1)
        
        mu_source = stc_to_distribution(parcel_tc, peak_sample)
        y = eeg[:, peak_sample]
        
        pairs.append({
            'mu_source': mu_source,
            'y': y,
            'active_parcels': active_indices.tolist(),
            'n_active': len(active_indices),
            'peak_time': peak_time,
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_simulations} simulations")
    
    return pairs
```

## Key Difference from Ex14b

In Ex14b, source distributions were sharp peaks:
```
dist[peak_node] = 0.8  # 80% mass on one parcel
```

Here, source distributions have realistic spatial extent:
```
spatial_pattern = exp(-d²/2σ²)  # Gaussian spread across parcels
```

This means:
- Sources are smoother (mass spread across several neighboring parcels)
- The "correct answer" is a smooth bump, not a delta
- Recovery should be easier in some ways (smooth targets)
  but harder in others (less distinct spatial signatures)

## No Diffusion Step

Important change: in Ex14b we artificially diffused the source before
measuring EEG. Here we DON'T diffuse — the EEG is directly generated
from the source activation through the leadfield. The "mixing" comes
from the physics (volume conduction through the head), not from an
artificial diffusion step.

This means:
- No tau_diff parameter in the context
- The context is just the EEG measurement: node_context + FiLM from y
- The model learns to invert the leadfield, not to undo diffusion
- context_dim changes: node_context=(N,2), global_cond=(64,) not (65,)

## Caching Strategy

### Cache 1: Simulation pairs (`ex14c_simulations.npz`)

Contains the MNE-generated (source distribution, EEG) pairs. Independent
of our flow matching framework. Regenerate only when changing simulation
parameters.

```python
SIM_CACHE = 'ex14c_simulations.npz'

def save_simulation_cache(path, train_pairs, test_pairs, metadata):
    """
    Save simulated EEG pairs to disk.
    
    metadata includes: n_simulations, spatial_extent, snr_db, seed,
    n_parcels, n_channels, sfreq, duration
    """
    np.savez(path,
        # Training pairs
        train_mu_sources=np.array([p['mu_source'] for p in train_pairs]),
        train_ys=np.array([p['y'] for p in train_pairs]),
        train_active_parcels=[p['active_parcels'] for p in train_pairs],
        train_n_active=np.array([p['n_active'] for p in train_pairs]),
        # Test pairs
        test_mu_sources=np.array([p['mu_source'] for p in test_pairs]),
        test_ys=np.array([p['y'] for p in test_pairs]),
        test_active_parcels=[p['active_parcels'] for p in test_pairs],
        test_n_active=np.array([p['n_active'] for p in test_pairs]),
        # Metadata
        **metadata,
    )

def load_simulation_cache(path):
    """Load cached simulation pairs."""
    data = np.load(path, allow_pickle=True)
    train_pairs = [
        {'mu_source': data['train_mu_sources'][i],
         'y': data['train_ys'][i],
         'active_parcels': data['train_active_parcels'][i],
         'n_active': int(data['train_n_active'][i])}
        for i in range(len(data['train_mu_sources']))
    ]
    test_pairs = [
        {'mu_source': data['test_mu_sources'][i],
         'y': data['test_ys'][i],
         'active_parcels': data['test_active_parcels'][i],
         'n_active': int(data['test_n_active'][i])}
        for i in range(len(data['test_mu_sources']))
    ]
    return train_pairs, test_pairs
```

### Usage in main()

```python
sim_cache = os.path.join(cache_dir, 
    f'ex14c_sims_n{args.n_simulations}_ext{args.spatial_extent}'
    f'_snr{args.snr_db}_seed{args.seed}.npz')

if os.path.exists(sim_cache) and not args.regenerate:
    print(f"Loading cached simulations from {sim_cache}")
    train_pairs, test_pairs = load_simulation_cache(sim_cache)
    print(f"  Loaded {len(train_pairs)} train, {len(test_pairs)} test pairs")
else:
    print(f"Generating {args.n_simulations} training simulations...")
    train_pairs = generate_training_data(
        labels, src, fwd, info, A, R,
        n_simulations=args.n_simulations,
        snr_db=args.snr_db, seed=args.seed)
    
    print(f"Generating {args.n_test} test simulations...")
    test_pairs = generate_training_data(
        labels, src, fwd, info, A, R,
        n_simulations=args.n_test,
        snr_db=args.snr_db, seed=args.seed + 1000)
    
    save_simulation_cache(sim_cache, train_pairs, test_pairs, {
        'n_simulations': args.n_simulations,
        'spatial_extent': args.spatial_extent,
        'snr_db': args.snr_db,
    })
    print(f"  Saved to {sim_cache}")
```

The OT dataset (coupling computation, flow sampling) is regenerated each
run since it takes only 5-10 minutes and depends on mode (uniform vs
Dirichlet), number of samples, etc.

## Training

### Dataset

```python
class RealisticEEGDataset(torch.utils.data.Dataset):
    """
    Training data from realistic MNE-simulated EEG.
    
    Flow: uniform (or Dirichlet) -> source distribution
    Conditioning: EEG measurement via FiLM + spatial
    No diffusion time parameter.
    
    Args:
        R: (N, N) rate matrix
        training_pairs: list of dicts with 'mu_source' and 'y'
        electrode_parcels: (n_channels,) mapping electrodes to parcels
        mode: 'uniform' or 'dirichlet'
        dirichlet_alpha: concentration parameter (only for dirichlet)
        n_starts_per_pair: Dirichlet starts per source (only for dirichlet)
        n_samples: total training samples to generate
        seed: random seed
    
    Returns per sample:
        mu_tau:       (N,) distribution at flow time
        tau:          (1,) flow time
        node_context: (N, 2) [sensor_val at sensor parcels, is_sensor]
        global_cond:  (n_channels,) raw EEG vector
        R_target:     (N, N) factorized target rate matrix
        edge_index:   (2, E) graph edges
        n_nodes:      int
    """
    def __init__(self, R, training_pairs, electrode_parcels,
                 mode='dirichlet', dirichlet_alpha=1.0,
                 n_starts_per_pair=10, n_samples=15000, seed=42):
        rng = np.random.default_rng(seed)
        N = R.shape[0]
        graph_struct = GraphStructure(R)
        cost = compute_cost_matrix(graph_struct)
        cache = GeodesicCache(graph_struct)
        edge_index = rate_matrix_to_edge_index(R)
        
        all_triples = []
        for pair in training_pairs:
            mu_source = pair['mu_source']
            y = pair['y']
            
            node_ctx, global_ctx = build_eeg_context_spatial(
                y, electrode_parcels, N)
            
            n_starts = n_starts_per_pair if mode == 'dirichlet' else 1
            for _ in range(n_starts):
                if mode == 'dirichlet':
                    mu_start = rng.dirichlet(np.full(N, dirichlet_alpha))
                else:
                    mu_start = np.ones(N) / N
                
                pi = compute_ot_coupling(mu_start, mu_source, cost)
                cache.precompute_for_coupling(pi)
                all_triples.append((mu_source, node_ctx, global_ctx, pi))
        
        print(f"  Precomputed {len(all_triples)} OT couplings")
        
        self.samples = []
        for _ in range(n_samples):
            mu_source, node_ctx, global_ctx, pi = \
                all_triples[int(rng.integers(len(all_triples)))]
            tau = float(rng.uniform(0.0, 0.999))
            
            mu_tau = marginal_distribution_fast(cache, pi, tau)
            R_target = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi, tau)
            
            self.samples.append((
                torch.tensor(mu_tau, dtype=torch.float32),
                torch.tensor([tau], dtype=torch.float32),
                torch.tensor(node_ctx, dtype=torch.float32),
                torch.tensor(global_ctx, dtype=torch.float32),
                torch.tensor(R_target, dtype=torch.float32),
                edge_index,
                N,
            ))
```

### Context building (no tau_diff)

```python
def build_eeg_context_spatial(y, electrode_parcels, N):
    """
    Place electrode readings at their nearest parcels.
    No tau_diff — EEG is directly from source, no artificial diffusion.
    
    Returns:
        node_context: (N, 2) [sensor_val, is_sensor]
        global_cond:  (n_channels,) raw EEG vector
    """
    sensor_vals = np.zeros(N)
    is_sensor = np.zeros(N)
    counts = np.zeros(N)
    for m, parcel_idx in enumerate(electrode_parcels):
        sensor_vals[parcel_idx] += y[m]
        is_sensor[parcel_idx] = 1.0
        counts[parcel_idx] += 1
    # Average for parcels with multiple electrodes
    mask = counts > 0
    sensor_vals[mask] /= counts[mask]
    
    node_context = np.stack([sensor_vals, is_sensor], axis=-1)
    global_cond = y.copy()  # just the raw EEG, no tau_diff
    return node_context, global_cond
```

### Model

```python
model = FiLMConditionalGNNRateMatrixPredictor(
    node_context_dim=2,
    global_dim=64,     # just EEG channels, no tau_diff
    hidden_dim=128,
    n_layers=6)
# Identity FiLM initialization applied automatically
```

### Training parameters

- 1000 simulations for training, 200 for testing
- Dirichlet mode: 10 starts per pair = 10,000 OT couplings
- 20000 training samples
- 1000 epochs, lr=5e-4, EMA decay=0.999
- Rate KL loss, uniform time weighting
- Gradient clipping max_norm=1.0

## Evaluation

### Test data

200 held-out simulations. Extract peak time point from each.

### Metrics

Same as Ex14b plus:
- **Spatial extent accuracy:** does the reconstructed activation have
  the correct spatial extent (not too peaked, not too diffuse)?
- **Amplitude correlation:** Pearson r between true and reconstructed
  parcel activations (before normalization)

### Baselines

Same: sLORETA, MNE, LASSO, backprojection. All applied to the same
EEG measurement at the peak time point.

## Plots

Same layout as Ex14b. The brain surface visualization (Panel B) is
especially important here — showing that the learned model recovers
realistic spatial extent while sLORETA is too diffuse.

## CLI

```python
parser.add_argument('--n-simulations', type=int, default=1000,
                    help='Number of training simulations')
parser.add_argument('--n-test', type=int, default=200,
                    help='Number of test simulations')
parser.add_argument('--spatial-extent', type=float, default=10.0,
                    help='Source spatial extent in mm')
parser.add_argument('--snr-db', type=float, default=20.0)
parser.add_argument('--mode', type=str, default='posterior',
                    choices=['point_estimate', 'posterior'])
parser.add_argument('--dirichlet-alpha', type=float, default=1.0)
parser.add_argument('--n-starts-per-pair', type=int, default=10,
                    help='Dirichlet starts per source (posterior mode only)')
parser.add_argument('--n-samples', type=int, default=20000,
                    help='Total training flow samples')
parser.add_argument('--n-epochs', type=int, default=1000)
parser.add_argument('--hidden-dim', type=int, default=128)
parser.add_argument('--n-layers', type=int, default=6)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--ema-decay', type=float, default=0.999)
parser.add_argument('--loss-type', type=str, default='rate_kl',
                    choices=['rate_kl', 'mse'])
parser.add_argument('--posterior-k', type=int, default=20,
                    help='Number of posterior samples at test time')
parser.add_argument('--regenerate', action='store_true',
                    help='Force regeneration of simulation cache')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data-path', type=str, default='ex14_eeg_data.npz',
                    help='Path to precomputed graph/leadfield from Ex14a')
```

## Script Structure

```python
def main():
    args = parse_args()
    
    # 1. Load graph and leadfield from Ex14a
    data = np.load(args.data_path, allow_pickle=True)
    R, A, adj = data['R'], data['A'], data['adj']
    
    # 2. Setup MNE (only needed for simulation, not training)
    fs_dir = mne.datasets.fetch_fsaverage()
    # ... setup src, labels, fwd, info, electrode_parcels
    
    # 3. Load or generate simulation cache
    sim_cache = f'ex14c_sims_n{args.n_simulations}_ext{args.spatial_extent}_snr{args.snr_db}.npz'
    if os.path.exists(sim_cache) and not args.regenerate:
        train_pairs, test_pairs = load_simulation_cache(sim_cache)
    else:
        train_pairs = generate_training_data(...)
        test_pairs = generate_training_data(...)
        save_simulation_cache(sim_cache, train_pairs, test_pairs, {...})
    
    # 4. Build flow matching dataset (OT couplings — not cached)
    dataset = RealisticEEGDataset(
        R=R, training_pairs=train_pairs,
        electrode_parcels=electrode_parcels,
        mode=args.mode,
        dirichlet_alpha=args.dirichlet_alpha,
        n_starts_per_pair=args.n_starts_per_pair,
        n_samples=args.n_samples, seed=args.seed)
    
    # 5. Train model
    model = FiLMConditionalGNNRateMatrixPredictor(
        node_context_dim=2, global_dim=A.shape[0],
        hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    history = train_film_conditional(model, dataset, ...)
    
    # 6. Evaluate on test pairs
    # 7. Run baselines on test pairs
    # 8. Plot results
```

## Dependencies

- mne (already installed from Ex14a)
- Everything else same as Ex14b

## Expected Outcome

With realistic sources:
- The learned model should still beat sLORETA and MNE on TV
- The advantage may be smaller than Ex14b (baselines handle smooth
  sources better than peaked ones)
- The posterior sampling should provide meaningful uncertainty on real
  brain anatomy
- Network-level identification should remain strong

If results are comparable to Ex14b: the framework handles realistic
data as well as synthetic. Ready for the paper.

If worse: the model was overfitting to peaked distributions in training.
Need to mix peaked and smooth sources in training.
