# Experiment 12b: `ex12b_cube5_boundary.py` — 5×5×5 Cube, Multi-Peak

## Motivation

Experiment 12 (v3) demonstrated strong interior reconstruction on a 4×4×4 cube
with single interior peaks: 100% peak recovery, 10x better interior TV than
the Laplacian baseline. Now scale up:
- 5×5×5 cube (125 nodes, 98 boundary, 27 interior, depths 0-2)
- 1-3 peaks, placed anywhere (boundary or interior)
- Variable diffusion time as conditioning

This tests whether the approach scales to deeper interiors (depth 2 = center
of cube, 2 hops from nearest boundary) and multi-peak reconstruction.

## Setup

### Graph

```python
R = make_cube_graph(size=5)       # 125 nodes
mask = cube_boundary_mask(size=5)  # 98 boundary, 27 interior
depth = cube_node_depth(size=5)    # depths 0, 1, 2
```

Interior breakdown:
- Depth 1: 18 nodes (one layer in from boundary)
- Depth 2: 9 nodes (the 3×3 center plane... actually the center of
  a 5×5×5 cube at depth 2 is a single node (2,2,2) plus its depth-2
  neighbors)

Wait, let me recount. Interior = all coordinates in {1,2,3}.
- Depth 1: min(x, 4-x, y, 4-y, z, 4-z) = 1. These have at least one
  coordinate equal to 1 or 3. Count: 27 - (3-2)^3 = 27 - 1 = 26.
  Actually: depth 2 nodes have all coordinates in {2}, so only (2,2,2).
  Depth 1 = 27 - 1 = 26 nodes. Depth 2 = 1 node (the very center).

So depth 2 is a single node. That's actually fine — recovering the center
of the cube from boundary observations is the ultimate test.

### Source distributions

1-3 peaks placed at random nodes (anywhere on the 125-node cube):

```python
def make_cube_multipeak(N, n_peaks, rng):
    peak_nodes = rng.choice(N, size=n_peaks, replace=False).tolist()
    weights = rng.dirichlet(np.full(n_peaks, 2.0))
    dist = np.ones(N) * 0.2 / N
    for node, w in zip(peak_nodes, weights):
        dist[node] += 0.8 * w
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist, peak_nodes
```

Peaks can land on boundary or interior nodes. Track which peaks are
interior for separate evaluation.

### Diffusion and observation

```python
tau_diff sampled uniformly from [0.5, 2.0] during training.
Test at tau_diff in {0.5, 1.0, 1.5}.
```

Wider range than Ex12 (which used fixed tau_diff=1.0) to test the model's
ability to account for different diffusion amounts.

Context per node: [mu_obs(a), mask(a), tau_diff]
context_dim = 3 (added tau_diff compared to Ex12).

### Flow

Start from UNIFORM. OT coupling: uniform → source. Conditioning provides
the diffused boundary observation, mask, and diffusion time.

## Training

- FlexibleConditionalGNNRateMatrixPredictor with context_dim=3
- hidden_dim=128 (larger graph, more complex task)
- n_layers=6 (need depth for 2-hop interior)
- 300 source distributions (100 per peak count)
- Variable tau_diff per source
- 15000 training samples
- 1000 epochs, lr=5e-4, gradient clipping max_norm=1.0
- 1/(1-t) factorization, loss normalization by N*(N-1)

### CLI

```python
parser.add_argument('--loss-weighting', type=str, default='uniform')
parser.add_argument('--n-epochs', type=int, default=1000)
parser.add_argument('--hidden-dim', type=int, default=128)
parser.add_argument('--n-layers', type=int, default=6)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--cube-size', type=int, default=5)
```

## Dataset

Extend CubeBoundaryDataset to accept precomputed (source, observation,
tau_diff) triples:

```python
# In main():
train_data = []
for _ in range(n_train):
    n_peaks = int(rng.integers(1, 4))
    mu_source, peak_nodes = make_cube_multipeak(N, n_peaks, rng)
    tau_diff = float(rng.uniform(0.5, 2.0))
    mu_diffused = mu_source @ expm(tau_diff * R)
    mu_obs = (mu_diffused * mask)
    mu_obs = np.clip(mu_obs, 1e-12, None)
    mu_obs /= mu_obs.sum()
    train_data.append({
        'mu_source': mu_source,
        'mu_obs': mu_obs,
        'tau_diff': tau_diff,
        'peak_nodes': peak_nodes,
    })

# Dataset generates: uniform → source, conditioned on (obs, mask, tau_diff)
```

Context per node:
```python
context = np.stack([mu_obs, mask, np.full(N, tau_diff)], axis=-1)  # (N, 3)
```

## Baselines

Same as Ex12: Laplacian harmonic extension and naive (uniform interior fill),
both using the diffused boundary values.

## Evaluation

### Test matrix

For each combination of:
- tau_diff in {0.5, 1.0, 1.5}
- n_peaks in {1, 2, 3}
- 10 test cases per combination
Total: 90 test cases

Track for each test case:
- Which peaks are on boundary vs interior
- Depth of each peak

### Metrics

- **Full TV**: all 125 nodes
- **Interior TV**: 27 interior nodes
- **Peak recovery (top-k)**: overall and split by boundary/interior peaks
- **Depth analysis**: MAE at depth 0, 1, 2
- **Center node recovery**: for cases with a peak at (2,2,2) specifically

### Comparison to Ex12 (4×4×4)

Report how metrics change from 4×4×4 to 5×5×5 at matched conditions
(single interior peak, tau_diff=1.0). Shows the cost of scaling.

## Plots (single figure, 2×3 grid)

- **Panel A**: Training loss curve.

- **Panel B**: Interior node bar chart for 2 example cases (like Ex12 Panel B
  but now 27 interior nodes). One case with a depth-1 interior peak,
  one with the depth-2 center peak. Show true, learned, Laplacian.

- **Panel C**: Full TV grouped by n_peaks. Bars for learned, Laplacian, naive.

- **Panel D**: Interior TV grouped by n_peaks. Same layout.

- **Panel E**: Reconstruction error vs depth. Lines for learned, Laplacian,
  naive. Now 3 depth levels (0, 1, 2) instead of Ex12's 2. The depth-2
  point is the key — can the model reach the center?

- **Panel F**: Peak recovery split by location. Grouped bars:
  "Boundary peaks" vs "Interior depth-1 peaks" vs "Center peak (depth 2)".
  For each group: learned vs Laplacian. Shows how recovery degrades with
  depth.

## Validation Checks

```
=== Experiment 12b: 5×5×5 Cube, Multi-Peak ===
Graph: 5×5×5 cube (125 nodes, 98 boundary, 27 interior)
Depths: 0 (98 nodes), 1 (26 nodes), 2 (1 node)

Full TV by n_peaks:
  1 peak: learned=X.XX, laplacian=X.XX
  2 peaks: learned=X.XX, laplacian=X.XX
  3 peaks: learned=X.XX, laplacian=X.XX

Interior TV by n_peaks:
  1 peak: learned=X.XX, laplacian=X.XX
  2 peaks: learned=X.XX, laplacian=X.XX
  3 peaks: learned=X.XX, laplacian=X.XX

Peak recovery by location:
  Boundary peaks: learned=XX%, laplacian=XX%
  Depth-1 peaks:  learned=XX%, laplacian=XX%
  Depth-2 peaks:  learned=XX%, laplacian=XX%

Depth MAE:
  Depth 0: learned=X.XXXX, laplacian=X.XXXX
  Depth 1: learned=X.XXXX, laplacian=X.XXXX
  Depth 2: learned=X.XXXX, laplacian=X.XXXX

Comparison to 4×4×4 (single interior peak, tau=1.0):
  4×4×4 interior TV: X.XXX
  5×5×5 interior TV: X.XXX
```

## Expected Outcome

- Depth-1 interior peaks: should work well (similar to 4×4×4 results)
- Depth-2 center peak: harder but the model has 6 layers of message
  passing, enough to propagate information from boundary to center.
  May have lower peak recovery but still beat Laplacian.
- Multi-peak: graceful degradation with peak count, similar to Ex8.2/Ex11.
- Variable tau_diff: model should handle different diffusion levels using
  the tau_diff conditioning.

The depth-2 recovery is the stretch goal. If the model recovers the center
node from boundary-only observations, that's impressive and directly
analogous to deep source localization in EEG.

## Dependencies

No new dependencies.
