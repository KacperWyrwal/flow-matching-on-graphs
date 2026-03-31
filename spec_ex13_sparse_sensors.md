# Experiment 13: `ex13_sparse_sensors.py` — Off-Graph Sparse Sensors

## Motivation

Experiments 12/12b observed all 98 boundary nodes of the cube directly.
Real EEG has far fewer sensors (~20-64) than source locations (~100-1000),
and sensors don't sit on the source graph — they measure linear mixtures
of source activations through volume conduction.

This experiment introduces:
1. **Sparse sensors**: only 20 measurement points (vs 98 boundary nodes)
2. **Mixing matrix**: each sensor measures a weighted average of nearby
   source nodes (not just one node's value)
3. **Underdetermined inverse problem**: 20 measurements, 125 unknowns

This is the last stepping stone before real EEG data.

## Setup

### Graph

5×5×5 cube (125 nodes), same as Ex12b.

### Sensor placement

Place 20 sensors on the boundary surface. Use a roughly uniform spacing:

```python
def place_sensors(size=5, n_sensors=20, seed=42):
    """
    Place sensors at a subset of boundary nodes, approximately uniformly
    distributed across the 6 faces.
    
    Simple approach: take all 98 boundary nodes, run farthest-point sampling
    to select 20 that are well-spread.
    
    Returns: list of boundary node indices (length n_sensors)
    """
    rng = np.random.default_rng(seed)
    mask = cube_boundary_mask(size)
    boundary_nodes = np.where(mask == 1)[0]
    
    # Farthest point sampling on the cube grid
    # Use Manhattan distance between boundary nodes
    selected = [int(rng.choice(boundary_nodes))]
    for _ in range(n_sensors - 1):
        dists = []
        for b in boundary_nodes:
            if b in selected:
                dists.append(0)
            else:
                min_d = min(manhattan_3d(b, s, size) for s in selected)
                dists.append(min_d)
        selected.append(int(boundary_nodes[np.argmax(dists)]))
    return selected


def manhattan_3d(node_a, node_b, size):
    ax, ay, az = node_a // (size*size), (node_a // size) % size, node_a % size
    bx, by, bz = node_b // (size*size), (node_b // size) % size, node_b % size
    return abs(ax-bx) + abs(ay-by) + abs(az-bz)
```

### Mixing matrix (leadfield)

Each sensor measures a weighted average of source nodes, with weights
decaying with distance:

```python
def compute_mixing_matrix(sensor_nodes, N, size, sigma=1.5):
    """
    Compute the M × N mixing matrix A.
    A[m, n] = weight of source node n on sensor m.
    
    Weight = exp(-d(sensor_m, source_n)^2 / (2*sigma^2))
    where d is the 3D Euclidean distance on the cube grid.
    
    Each row is normalized to sum to 1.
    
    Args:
        sensor_nodes: list of M sensor node indices
        N: total number of source nodes
        size: cube side length
        sigma: spatial spread of each sensor's sensitivity
    
    Returns: (M, N) mixing matrix
    """
    M = len(sensor_nodes)
    A = np.zeros((M, N))
    for m, s_node in enumerate(sensor_nodes):
        sx, sy, sz = s_node // (size*size), (s_node // size) % size, s_node % size
        for n in range(N):
            nx, ny, nz = n // (size*size), (n // size) % size, n % size
            d_sq = (sx-nx)**2 + (sy-ny)**2 + (sz-nz)**2
            A[m, n] = np.exp(-d_sq / (2 * sigma**2))
        A[m] /= A[m].sum()  # normalize
    return A
```

The sigma parameter controls how spread out each sensor's sensitivity is.
sigma=1.5 means each sensor is most sensitive to its location and the
~6 nearest neighbors, with decay beyond that.

### Observation model

Given a source distribution mu_source:
1. Diffuse: mu_diffused = mu_source @ expm(tau_diff * R)
2. Measure: y = A @ mu_diffused (M-dimensional sensor readings)
3. The model receives y, must recover mu_source

### Context: backprojection

The sensor readings y are M-dimensional (M=20), not N-dimensional (N=125).
To feed them to the GNN as per-node features, backproject onto the graph:

```python
mu_backproj = A.T @ y  # (N,) — pseudoinverse-like backprojection
mu_backproj /= mu_backproj.sum()  # normalize to distribution
```

This gives a blurry, smeared version of the source — each node gets
a weighted average of the sensor readings. The GNN's job is to sharpen
this backprojection using graph structure and learned priors.

Context per node: [mu_backproj(a), tau_diff]
context_dim = 2.

Note: we don't need the mask anymore since all nodes get a backprojected
value (none are exactly zero). The backprojection itself encodes the
sensor geometry.

### Flow

Start from UNIFORM. OT coupling: uniform → source. Conditioning provides
the backprojected sensor readings and diffusion time.

## Training

- FlexibleConditionalGNNRateMatrixPredictor with context_dim=2
- hidden_dim=128, n_layers=6
- Source distributions: 1-3 peaks (300 training, 100 per peak count)
- tau_diff sampled from [0.5, 2.0]
- 15000 training samples
- 1000 epochs, lr=5e-4, gradient clipping
- 1/(1-t) factorization, loss normalization

### CLI

```python
parser.add_argument('--n-sensors', type=int, default=20)
parser.add_argument('--sensor-sigma', type=float, default=1.5)
parser.add_argument('--n-epochs', type=int, default=1000)
parser.add_argument('--hidden-dim', type=int, default=128)
parser.add_argument('--n-layers', type=int, default=6)
parser.add_argument('--lr', type=float, default=5e-4)
```

## Baselines

### 1. Backprojection (naive)

Just use A^T y, normalized. The simplest possible reconstruction.

```python
def baseline_backproj(y, A):
    mu = A.T @ y
    mu = np.clip(mu, 0, None)
    mu /= mu.sum()
    return mu
```

### 2. Minimum Norm Estimate (MNE)

Regularized pseudo-inverse:

```python
def baseline_mne(y, A, lam=1e-3):
    """mu_hat = A^T (A A^T + lambda I)^{-1} y"""
    M = A.shape[0]
    mu = A.T @ np.linalg.solve(A @ A.T + lam * np.eye(M), y)
    mu = np.clip(mu, 0, None)
    mu /= mu.sum()
    return mu
```

### 3. L1/LASSO (sparse reconstruction)

The strongest classical baseline for peaked sources:

```python
def baseline_lasso(y, A, alpha=0.01):
    """Sparse reconstruction via LASSO."""
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=alpha, positive=True, max_iter=5000)
    model.fit(A, y)
    mu = model.coef_
    mu = np.clip(mu, 0, None)
    s = mu.sum()
    return mu / s if s > 1e-12 else np.ones(len(mu)) / len(mu)
```

### Baseline regularization tuning

For MNE and LASSO, the regularization parameters (lambda, alpha) matter.
For a fair comparison, tune them on a small validation set:

```python
# Try a few values, pick the one with lowest mean TV on 20 validation cases
lambdas = [1e-4, 1e-3, 1e-2, 1e-1]
alphas = [1e-4, 1e-3, 1e-2, 1e-1]
```

Report the best-tuned baseline results.

## Evaluation

### Test cases

90 test cases: 10 per (n_peaks, tau_diff) combination.
n_peaks in {1, 2, 3}, tau_diff in {0.5, 1.0, 1.5}.

### Metrics

- **Full TV**: all 125 nodes
- **Interior TV**: 27 interior nodes
- **Peak recovery (top-k)**: overall and by boundary/interior
- **Depth analysis**: MAE at depth 0, 1, 2

### Comparison structure

Four methods: Learned, MNE, LASSO, Backprojection.

The key comparison is **Learned vs LASSO**: both can produce sparse solutions,
but the learned model also uses graph structure. If the learned model beats
LASSO on interior peaks, it demonstrates the value of graph-aware inversion.

## Plots (single figure, 2×3 grid)

- **Panel A**: Training loss curve.

- **Panel B**: Example reconstruction for one test case. Show the 125-node
  distribution as 3 cube slices (z=0, z=2, z=4) for: backprojection (input
  context), learned, LASSO, true source. The backprojection should be blurry,
  LASSO should be sparse but potentially wrong locations, learned should be
  sparse and correctly located.

- **Panel C**: Full TV by method and n_peaks. Grouped bars: 4 methods ×
  3 peak counts.

- **Panel D**: Interior TV by method. Single set of bars (4 methods),
  averaged across all test cases.

- **Panel E**: Peak recovery by method and peak depth. Grouped bars:
  4 methods × 3 depth categories (boundary, depth-1, depth-2).
  The key comparison: Learned vs LASSO at interior depths.

- **Panel F**: TV vs number of sensors. Run evaluation at n_sensors =
  {10, 15, 20, 30, 50} for the learned model only (retrain or use same
  model with different A). Shows how reconstruction quality degrades with
  fewer sensors. This is directly relevant to EEG experimental design.

  Note: For the sensor sweep, retraining per sensor count is expensive.
  Alternative: train on 20 sensors, evaluate at different counts by
  recomputing A and backprojection. The model might still work reasonably
  since the backprojection adapts to the sensor geometry.
  
  Actually, this won't work without retraining since the model was trained
  on 20-sensor backprojections. Skip the sensor sweep for now, or make it
  a stretch goal.

## Validation Checks

```
=== Experiment 13: Sparse Sensors on 5×5×5 Cube ===
Sensors: 20 (of 98 boundary nodes), sigma=1.5
Measurements: 20-dimensional, sources: 125-dimensional

Baseline tuning:
  Best MNE lambda: X.XXX
  Best LASSO alpha: X.XXX

Full TV:
  Learned:       X.XXXX ± X.XXXX
  LASSO:         X.XXXX ± X.XXXX
  MNE:           X.XXXX ± X.XXXX
  Backprojection: X.XXXX ± X.XXXX

Interior TV:
  Learned:       X.XXXX ± X.XXXX
  LASSO:         X.XXXX ± X.XXXX
  MNE:           X.XXXX ± X.XXXX
  Backprojection: X.XXXX ± X.XXXX

Peak recovery by depth:
  Boundary:   learned=XX%, lasso=XX%, mne=XX%
  Depth-1:    learned=XX%, lasso=XX%, mne=XX%
  Depth-2:    learned=XX%, lasso=XX%, mne=XX%
```

## Expected Outcome

- **Backprojection**: blurry, worst TV, poor peak recovery
- **MNE**: smoother than backprojection but still smeared, moderate TV
- **LASSO**: sparse, good peak recovery on boundary, may struggle with
  interior peaks (L1 doesn't know about graph structure)
- **Learned**: sparse AND graph-aware, best interior peak recovery

The learned model's advantage should be most visible on interior peaks
at depth 1-2, where graph structure helps propagate information from
sparse boundary sensors to deep interior nodes. LASSO treats all 125
nodes independently and doesn't know that interior node (2,2,2) is
connected to its neighbors.

If the learned model matches LASSO on boundary peaks and beats it on
interior peaks, that demonstrates the value of the graph-aware approach.
This would be a convincing result for the EEG source imaging community.

## Dependencies

Add scikit-learn for LASSO baseline:
```
scikit-learn>=1.0
```
