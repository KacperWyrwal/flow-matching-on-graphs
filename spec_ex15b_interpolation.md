# Experiment 15b: Bike Sharing Spatial Interpolation

## Task

Given bike counts at 30% of stations, reconstruct the full distribution
across all 60 stations. Posterior sampling provides uncertainty over
the unobserved stations.

This is Ex13 (sparse sensors) on a real graph with real data.

## Data

From Ex15a pipeline:
- 60 Citi Bike stations in Manhattan
- Trip-based adjacency graph (1288 edges)
- Hourly activity distributions (normalized bike trip counts per station)
- ~720 hourly snapshots per month

Split: first 3 weeks training (~500 snapshots), last week testing (~170).

### Preprocessing

```python
# Load from Ex15a cache
data = np.load('ex15a_citibike_data.npz', allow_pickle=True)
R = data['R']                  # (60, 60) rate matrix
distributions = data['distributions']  # (T, 60) hourly distributions
hours = data['hours']          # (T,) hour of day
dows = data['dows']            # (T,) day of week
positions = data['positions']  # (60, 2) x,y coordinates in meters

N = R.shape[0]  # 60
n_train = int(0.8 * len(distributions))
train_dists = distributions[:n_train]
test_dists = distributions[n_train:]
```

## Observation Model

For each snapshot, randomly select 30% of stations as observed:

```python
def mask_distribution(mu, obs_fraction=0.3, rng=None):
    """
    Randomly observe a fraction of stations.
    Returns observed distribution (unobserved zeroed, renormalized)
    and binary mask.
    """
    N = len(mu)
    n_obs = max(1, int(N * obs_fraction))
    obs_idx = rng.choice(N, size=n_obs, replace=False)
    
    mask = np.zeros(N)
    mask[obs_idx] = 1.0
    
    mu_obs = mu * mask
    s = mu_obs.sum()
    if s > 1e-12:
        mu_obs /= s
    
    return mu_obs, mask
```

## Context

### Per-node features
[observed_value * mask, mask] — same as Ex12b boundary observation
format. context_dim = 2.

### Global FiLM conditioning
The raw observation vector (60-dim, zeros for unobserved stations).
This is redundant with the per-node context — it contains the same
information. But FiLM processes it through an MLP that can extract
global patterns: total system load, spatial clustering of observations,
whether this looks like a typical or anomalous snapshot. The per-node
context gives local spatial anchoring, FiLM gives global summary.

```python
def build_bike_context(mu_obs, mask, N):
    """
    Build per-node and global context for bike interpolation.
    
    Returns:
        node_context: (N, 2) [observed_value * mask, mask]
        global_cond:  (N,) raw observation vector (zeros for unobserved)
    """
    node_context = np.stack([mu_obs * mask, mask], axis=-1)
    global_cond = mu_obs * mask  # (N,) with zeros for unobserved
    return node_context, global_cond
```

With identity FiLM initialization, the model starts identical to
Ex12b (ignoring global conditioning) and gradually learns to use
the global signal if it helps.

## Dataset

```python
class BikeInterpolationDataset(torch.utils.data.Dataset):
    """
    Training data for bike sharing interpolation.
    
    For each hourly snapshot:
    1. Randomly mask 70% of stations
    2. OT coupling: uniform/Dirichlet -> true distribution
    3. Sample flow: (mu_tau, tau, node_context, global_cond, R_target)
    4. Node context: [observed values, mask]
    5. Global cond: raw observation vector (for FiLM)
    
    Mode 'uniform': all flows start from uniform
    Mode 'dirichlet': flows start from Dirichlet(alpha,...,alpha)
    
    Returns 7-tuple compatible with train_film_conditional:
        (mu_tau, tau, node_context, global_cond, R_target, edge_index, N)
    """
    def __init__(self, R, distributions, obs_fraction=0.3,
                 mode='dirichlet', dirichlet_alpha=1.0,
                 n_starts_per_dist=5, n_samples=15000, seed=42):
        rng = np.random.default_rng(seed)
        N = R.shape[0]
        graph_struct = GraphStructure(R)
        cost = compute_cost_matrix(graph_struct)
        cache = GeodesicCache(graph_struct)
        edge_index = rate_matrix_to_edge_index(R)
        
        all_triples = []
        for mu_clean in distributions:
            # Random mask for this snapshot
            mu_obs, obs_mask = mask_distribution(mu_clean, obs_fraction, rng)
            node_ctx, global_ctx = build_bike_context(mu_obs, obs_mask, N)
            
            n_starts = n_starts_per_dist if mode == 'dirichlet' else 1
            for _ in range(n_starts):
                if mode == 'dirichlet':
                    mu_start = rng.dirichlet(np.full(N, dirichlet_alpha))
                else:
                    mu_start = np.ones(N) / N
                
                pi = compute_ot_coupling(mu_start, mu_clean, cost)
                cache.precompute_for_coupling(pi)
                all_triples.append((mu_clean, node_ctx, global_ctx, pi))
        
        self.samples = []
        for _ in range(n_samples):
            mu_clean, node_ctx, global_ctx, pi = \
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

Note: each training snapshot gets a DIFFERENT random mask. This means
the model must generalize across mask patterns, not memorize a fixed
observation set.

## Model

```python
model = FiLMConditionalGNNRateMatrixPredictor(
    node_context_dim=2,   # [observed_value, mask]
    global_dim=60,        # raw observation vector for FiLM
    hidden_dim=64,        # 60 nodes, moderate size
    n_layers=4)           # dense trip graph, short diameter
# Identity FiLM initialization applied automatically
```

Uses FiLM architecture from Ex13, but with on-graph observations
(like Ex12b) plus global conditioning. Best of both worlds.

## Training

- FiLMConditionalGNNRateMatrixPredictor with node_context_dim=2, global_dim=60
- hidden_dim=64, n_layers=4
- 500 hourly snapshots for training
- Dirichlet mode: 5 starts per snapshot = 2500 OT couplings (60×60 LP, fast)
- 15000 training samples
- 1000 epochs, lr=5e-4, EMA decay=0.999
- Rate KL loss, uniform time weighting
- Gradient clipping max_norm=1.0
- Uses train_film_conditional (7-tuple dataset format)

## Baselines

### 1. Laplacian smoothing (harmonic extension)

Fix observed station values, fill unobserved by harmonic extension
on the trip graph. Same as Ex12.

```python
def baseline_laplacian(mu_obs, mask, R):
    L = -R.copy()
    obs_idx = np.where(mask == 1)[0]
    unobs_idx = np.where(mask == 0)[0]
    L_II = L[np.ix_(unobs_idx, unobs_idx)]
    L_IB = L[np.ix_(unobs_idx, obs_idx)]
    u_B = mu_obs[obs_idx]
    u_I = np.linalg.solve(L_II, -L_IB @ u_B)
    result = mu_obs.copy()
    result[unobs_idx] = u_I
    result = np.clip(result, 0, None)
    result /= result.sum()
    return result
```

### 2. k-NN interpolation

Fill each unobserved station with the average of its k nearest
observed neighbors (geographic distance).

```python
def baseline_knn(mu_obs, mask, positions, k=3):
    result = mu_obs.copy()
    obs_idx = np.where(mask == 1)[0]
    unobs_idx = np.where(mask == 0)[0]
    
    for i in unobs_idx:
        dists = np.linalg.norm(positions[obs_idx] - positions[i], axis=1)
        nearest = obs_idx[np.argsort(dists)[:k]]
        result[i] = mu_obs[nearest].mean()
    
    result = np.clip(result, 0, None)
    result /= result.sum()
    return result
```

### 3. Historical mean

Average distribution at the same hour and day-of-week, computed
from training data. Ignores current observations entirely.

```python
def baseline_historical_mean(hour, dow, train_dists, train_hours, train_dows):
    mask_h = (train_hours == hour) & (train_dows == dow)
    if mask_h.sum() > 0:
        return train_dists[mask_h].mean(axis=0)
    # Fallback: same hour any day
    mask_h = train_hours == hour
    return train_dists[mask_h].mean(axis=0)
```

### 4. Historical mean + Laplacian

Hybrid: use historical mean as prior, then adjust based on
observed stations via Laplacian smoothing. Strong baseline.

```python
def baseline_hist_laplacian(mu_obs, mask, R, hist_mean):
    """
    Laplacian smoothing from observed values, but use historical
    mean as the initial guess for unobserved stations instead of
    pure harmonic extension.
    """
    result = hist_mean.copy()
    obs_idx = np.where(mask == 1)[0]
    result[obs_idx] = mu_obs[obs_idx]
    # One round of Laplacian smoothing for unobserved
    # ... or solve the regularized Dirichlet problem
    result = np.clip(result, 0, None)
    result /= result.sum()
    return result
```

## Evaluation

### Test cases

~170 held-out hourly snapshots. For each:
1. Randomly mask 70% of stations (different mask per test case)
2. Run learned model (posterior mean from K=20 Dirichlet starts)
3. Run all baselines
4. Compare to true full distribution

### Metrics

- **Full TV:** all 60 stations
- **Unobserved TV:** only the ~42 unobserved stations
- **Station-level correlation:** Pearson r between true and reconstructed
  (across all test cases, per station)
- **Calibration:** posterior std vs |error| correlation
- **Diversity:** mean pairwise TV between posterior samples

### Split by conditions

- By hour of day (morning rush vs midday vs evening vs night)
- By day of week (weekday vs weekend)
- By observation fraction (try 0.1, 0.2, 0.3, 0.5 at test time)

## Plots

### Figure 1: Main results (2×3)

- **Panel A:** Training loss curve
- **Panel B:** Map of Manhattan with interpolation example.
  Show: observed stations (large dots with true values), unobserved
  stations (small dots colored by reconstructed value), and true
  values as faint outline. Side by side: learned vs Laplacian vs true.
- **Panel C:** TV comparison bar chart (all methods)
- **Panel D:** Unobserved station TV comparison
- **Panel E:** TV vs observation fraction (0.1 to 0.5). Lines for
  each method. Shows graceful degradation.
- **Panel F:** Calibration scatter (posterior std vs error)

### Figure 2: Posterior visualization (the "aha" figure)

Full page figure showing Manhattan station map:
- True distribution
- Posterior mean
- Posterior uncertainty (std as circle size or color intensity)
- 4 individual posterior samples

Each as a separate map panel. The uncertainty should be highest
at unobserved stations far from any observed station, and lowest
at observed stations (which are known exactly).

## Console Output

```
=== Experiment 15b: Bike Sharing Interpolation ===
Stations: 60, Edges: 1288, Obs fraction: 30%
Train: 500 snapshots, Test: 170 snapshots

Full TV:
  Learned (posterior mean): X.XXXX ± X.XXXX
  Laplacian:                X.XXXX ± X.XXXX
  k-NN:                     X.XXXX ± X.XXXX
  Historical mean:          X.XXXX ± X.XXXX
  Hist + Laplacian:         X.XXXX ± X.XXXX

Unobserved TV:
  Learned: X.XXXX, Laplacian: X.XXXX, k-NN: X.XXXX

Calibration: r = X.XX
Diversity: X.XXXX

By observation fraction:
  10%: learned=X.XX, laplacian=X.XX, hist=X.XX
  20%: learned=X.XX, laplacian=X.XX, hist=X.XX
  30%: learned=X.XX, laplacian=X.XX, hist=X.XX
  50%: learned=X.XX, laplacian=X.XX, hist=X.XX
```

## Expected Outcome

The learned model should beat Laplacian and k-NN because:
- The trip graph captures transport connections that geographic
  distance misses
- The model learns the prior distribution of bike patterns (from
  training data), which purely geometric methods don't have

The historical mean is the toughest baseline — bike patterns are
very regular. The learned model's advantage: it conditions on
CURRENT observations, adapting to anomalies that the historical
mean cannot capture.

The posterior sampling provides uncertainty that no baseline offers.
Stations far from any observed station should have high uncertainty.
Stations near observed stations should have low uncertainty. This
is immediately useful for bike rebalancing operations.

## CLI

```python
parser.add_argument('--data-path', type=str, default='ex15a_citibike_data.npz')
parser.add_argument('--obs-fraction', type=float, default=0.3)
parser.add_argument('--mode', type=str, default='posterior',
                    choices=['point_estimate', 'posterior'])
parser.add_argument('--dirichlet-alpha', type=float, default=1.0)
parser.add_argument('--n-epochs', type=int, default=1000)
parser.add_argument('--hidden-dim', type=int, default=64)
parser.add_argument('--n-layers', type=int, default=4)
parser.add_argument('--posterior-k', type=int, default=20)
parser.add_argument('--loss-type', type=str, default='rate_kl')
parser.add_argument('--regenerate', action='store_true')
```

## Dependencies

No new dependencies beyond what Ex15a already uses.
