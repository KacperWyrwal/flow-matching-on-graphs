# Experiment 14b: `ex14b_eeg_train.py` — Model Training & Evaluation

## Purpose

Train and evaluate the flow matching model on the real cortical mesh
with real physics, using data validated by Script 14a. This script
loads the precomputed graph, leadfield, and adjacency from the .npz
file — no MNE dependency needed.

## Data Loading

```python
data = np.load('ex14_eeg_data.npz', allow_pickle=True)
R = data['R']                      # (100, 100)
A = data['A']                      # (64, 100)
adj = data['adj']                  # (100, 100)
parcel_centroids = data['parcel_centroids']  # (100, 3)
parcel_names = data['parcel_names']          # (100,) str
network_assignments = data['network_assignments']  # (100,) int

N = R.shape[0]
n_channels = A.shape[0]
edge_index = rate_matrix_to_edge_index(R)
```

## Training Data Generation

```python
def generate_eeg_training_data(R, A, n_dists=300, tau_range=(0.5, 2.0),
                                snr_db=20, seed=42):
    """
    Generate training pairs: (source, EEG observation, tau_diff).
    
    Returns list of dicts with keys:
        mu_source, y, tau_diff, peak_parcels, n_peaks
    """
    rng = np.random.default_rng(seed)
    N = R.shape[0]
    pairs = []
    
    for _ in range(n_dists):
        n_peaks = int(rng.integers(1, 4))
        mu_source, peak_parcels = make_cortical_source(N, n_peaks, rng)
        tau_diff = float(rng.uniform(*tau_range))
        
        # Forward model: diffuse + measure
        mu_diffused = mu_source @ expm(tau_diff * R)
        y_clean = A @ mu_diffused
        
        # Add noise
        if snr_db is not None and snr_db < 100:
            signal_power = np.mean(y_clean ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            y = y_clean + rng.normal(0, np.sqrt(noise_power), len(y_clean))
        else:
            y = y_clean
        
        pairs.append({
            'mu_source': mu_source,
            'y': y,
            'tau_diff': tau_diff,
            'peak_parcels': peak_parcels,
            'n_peaks': n_peaks,
        })
    
    return pairs
```

## Mode A: Point Estimate (uniform start)

Train the FiLM model with uniform starts, same as Ex13.

### Dataset

```python
class EEGPointEstimateDataset(torch.utils.data.Dataset):
    """
    Flow: uniform -> source, conditioned on EEG + tau_diff.
    Context: spatial sensor placement + FiLM from raw EEG vector.
    """
    def __init__(self, R, A, training_pairs, electrode_parcels,
                 n_samples=15000, seed=42):
        rng = np.random.default_rng(seed)
        N = R.shape[0]
        graph_struct = GraphStructure(R)
        cost = compute_cost_matrix(graph_struct)
        cache = GeodesicCache(graph_struct)
        edge_index = rate_matrix_to_edge_index(R)
        
        mu_start = np.ones(N) / N
        
        precomputed = []
        for pair in training_pairs:
            node_ctx, global_ctx = build_eeg_context_spatial(
                pair['y'], electrode_parcels, N, pair['tau_diff'])
            pi = compute_ot_coupling(mu_start, pair['mu_source'], cost)
            cache.precompute_for_coupling(pi)
            precomputed.append((pair['mu_source'], node_ctx, global_ctx, pi))
        
        self.samples = []
        for _ in range(n_samples):
            mu_source, node_ctx, global_ctx, pi = \
                precomputed[int(rng.integers(len(precomputed)))]
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

### Training
```python
model = FiLMConditionalGNNRateMatrixPredictor(
    node_context_dim=2,
    global_dim=n_channels + 1,  # 64 + 1 = 65
    hidden_dim=128,
    n_layers=6)

# Identity FiLM initialization (critical!)
# Already applied inside the model class after the fix

history = train_film_conditional(
    model, dataset,
    n_epochs=args.n_epochs,
    batch_size=256,
    lr=args.lr,
    device=device,
    loss_weighting='uniform',
    ema_decay=0.999)
```

## Mode B: Posterior Sampling (Dirichlet starts)

Train with Dirichlet(1,...,1) starts for posterior sampling.

### Dataset

```python
class EEGPosteriorDataset(torch.utils.data.Dataset):
    """
    Flow: Dirichlet(1,...,1) -> source, conditioned on EEG + tau_diff.
    Multiple Dirichlet starts per source for diversity.
    """
    def __init__(self, R, A, training_pairs, electrode_parcels,
                 n_starts_per_pair=10, n_samples=15000, seed=42):
        rng = np.random.default_rng(seed)
        N = R.shape[0]
        graph_struct = GraphStructure(R)
        cost = compute_cost_matrix(graph_struct)
        cache = GeodesicCache(graph_struct)
        edge_index = rate_matrix_to_edge_index(R)
        
        all_triples = []
        for pair in training_pairs:
            node_ctx, global_ctx = build_eeg_context_spatial(
                pair['y'], electrode_parcels, N, pair['tau_diff'])
            
            for _ in range(n_starts_per_pair):
                mu_start = rng.dirichlet(np.ones(N))
                pi = compute_ot_coupling(mu_start, pair['mu_source'], cost)
                cache.precompute_for_coupling(pi)
                all_triples.append((pair['mu_source'], node_ctx,
                                     global_ctx, pi))
        
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

### Scaling concern

300 source distributions × 10 Dirichlet starts = 3000 OT couplings on
a 100-node graph. Each coupling is a 100×100 LP. This is smaller than
the cube (125 nodes) so should be faster. Estimate: 5-15 minutes for
dataset generation.

## Evaluation

### Test data

Generate 90 test cases (same structure as training, different seed):
- n_peaks in {1, 2, 3}, tau_diff in {0.5, 1.0, 1.5}
- 10 cases per combination

### For point estimate model

```python
for tc in test_cases:
    y = tc['y']
    node_ctx, global_ctx = build_eeg_context_spatial(
        y, electrode_parcels, N, tc['tau_diff'])
    mu_start = np.ones(N) / N
    _, traj = sample_trajectory_film(
        model, mu_start, node_ctx, global_ctx, edge_index,
        n_steps=200, device=device)
    tc['mu_learned'] = traj[-1]
```

### For posterior model

```python
for tc in test_cases:
    y = tc['y']
    node_ctx, global_ctx = build_eeg_context_spatial(
        y, electrode_parcels, N, tc['tau_diff'])
    
    samples = []
    for k in range(K):
        mu_start = rng.dirichlet(np.ones(N))
        _, traj = sample_trajectory_film(
            model_posterior, mu_start, node_ctx, global_ctx, edge_index,
            n_steps=200, device=device)
        samples.append(traj[-1])
    
    tc['posterior_samples'] = samples
    tc['posterior_mean'] = np.mean(samples, axis=0)
    tc['posterior_std'] = np.std(samples, axis=0)
```

### Metrics

**Reconstruction:**
- Full TV (all 100 parcels)
- Peak recovery (top-k)
- Hemispheric accuracy (is the peak in the correct hemisphere?)
- Network accuracy (is the peak in the correct Schaefer network out of 7?)

**Posterior (if mode B):**
- Calibration: r between posterior std and |posterior mean - true|
- Diversity: mean pairwise TV between posterior samples
- Network-level conditional P: P(correct network | EEG observation)
- Diversity scaling: more diversity for hard cases (deep sources,
  high noise, multi-peak)?

**Noise robustness (optional sweep):**
- Evaluate at SNR = {inf, 20, 10, 5} dB
- Report TV and peak recovery vs SNR for learned model and baselines

### GNN+softmax baseline (optional)

Train a DirectGNNPredictor with the same architecture for comparison:
```python
parser.add_argument('--train-direct-gnn', action='store_true')
```

## Plots

### Figure 1: Main results (2×3 grid)

- **Panel A**: Training loss curve (with EMA smoothed overlay if enabled)

- **Panel B**: Brain surface visualization (THE figure).
  One test case: show inflated cortical surface (lateral and medial views)
  colored by:
  - True source distribution
  - Learned reconstruction (posterior mean or point estimate)
  - sLORETA reconstruction
  - Posterior std map (if mode B)
  Use nilearn.plotting.plot_surf or save parcel values and render externally.

  If nilearn plotting is complex, alternative: show a circular/matrix
  plot of 100 parcel values as a heatmap with network grouping (7 blocks
  along the diagonal).

- **Panel C**: Full TV comparison. Bar chart: learned vs sLORETA vs MNE
  vs LASSO vs backprojection. Split by n_peaks if space allows.

- **Panel D**: Network-level accuracy. For each method: what fraction of
  test cases correctly identify the Schaefer network containing the peak?
  This is a coarser but clinically relevant metric — getting the right
  brain region is more important than the exact parcel.

- **Panel E**: Calibration plot (if mode B). Scatter of std vs |error|,
  colored by network. Report r.

- **Panel F**: Noise robustness. TV vs SNR for all methods. Lines with
  markers. The learned model should degrade gracefully.

### Figure 2 (supplementary): Posterior analysis

- Conditional probabilities across 7 networks for 3 test cases
- Individual posterior samples on brain surface
- Diversity vs difficulty (n_peaks, tau_diff)

## Console Output

```
=== Experiment 14b: Synthetic EEG Model Training ===
Graph: 100 parcels, XX edges, 64 EEG channels
Mode: {point_estimate | posterior}

Training:
  Initial loss: X.XXXXXX
  Final loss: X.XXXXXX
  Epochs: XXXX

Point estimate results (90 test cases):
  Full TV:
    Learned:     X.XXXX ± X.XXXX
    sLORETA:     X.XXXX ± X.XXXX
    MNE:         X.XXXX ± X.XXXX
    LASSO:       X.XXXX ± X.XXXX
    Backproj:    X.XXXX ± X.XXXX

  Peak recovery (correct parcel):
    Learned: XX%, sLORETA: XX%, MNE: XX%, LASSO: XX%

  Network accuracy (correct out of 7):
    Learned: XX%, sLORETA: XX%, MNE: XX%, LASSO: XX%

  Hemispheric accuracy:
    Learned: XX%, sLORETA: XX%, MNE: XX%, LASSO: XX%

Posterior results (if mode B):
  Posterior mean TV: X.XXXX ± X.XXXX
  Calibration r: X.XX
  Diversity (mean pairwise TV): X.XXXX
  Network identification from posterior: XX%
```

## CLI

```python
parser.add_argument('--mode', type=str, default='point_estimate',
                    choices=['point_estimate', 'posterior', 'both'])
parser.add_argument('--data-path', type=str, default='ex14_eeg_data.npz',
                    help='Path to precomputed EEG data from ex14a')
parser.add_argument('--n-epochs', type=int, default=1000)
parser.add_argument('--hidden-dim', type=int, default=128)
parser.add_argument('--n-layers', type=int, default=6)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--ema-decay', type=float, default=0.999)
parser.add_argument('--n-train-dists', type=int, default=300)
parser.add_argument('--n-samples', type=int, default=15000)
parser.add_argument('--snr-db', type=float, default=20.0)
parser.add_argument('--posterior-k', type=int, default=20)
parser.add_argument('--dirichlet-alpha', type=float, default=1.0)
parser.add_argument('--train-direct-gnn', action='store_true')
parser.add_argument('--noise-sweep', action='store_true',
                    help='Run evaluation at multiple SNR levels')
```

## Dependencies

Only numpy, torch, scipy, sklearn (for LASSO), matplotlib.
MNE/nilearn only needed for brain visualization (optional, can defer to
a separate plotting script).

## Expected Outcome

The cortical graph has properties that should favor our framework:
- Irregular topology (not a regular grid) — tests topology adaptation
- Folded geometry (nearby in 3D ≠ nearby on cortex) — graph distance
  is more informative than Euclidean distance
- Known network structure (7 functional networks) — posterior analysis
  can report network-level probabilities

Expected results:
- Learned model beats MNE and backprojection on TV and peak recovery
- Learned model is competitive with sLORETA (which is specifically designed
  for EEG and uses the resolution matrix)
- Learned model beats LASSO on multi-peak cases (LASSO finds some peaks
  but misses spatial coherence)
- Posterior sampling provides calibrated uncertainty (r > 0.5)
- Network-level identification from posterior is high (>70%) since
  the 7 networks partition the 100 parcels into groups of ~14

If learned model beats sLORETA: strong result, publishable standalone.
If comparable: still good — we additionally provide uncertainty that
sLORETA doesn't.
If worse: the cortical graph or the leadfield conditioning needs work.
