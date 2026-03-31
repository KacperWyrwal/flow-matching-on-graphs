# Experiment 12: `ex12_cube_boundary.py` — Boundary Observation on 3D Cube

## Motivation

Build toward inverse EEG by testing two capabilities simultaneously:
1. Scaling to ~125 nodes (our largest graph, 5x previous max)
2. Structured observation: observe the boundary of a 3D volume, infer the
   interior — the same geometric setup as EEG (sensors on scalp, sources
   inside brain)

A 5×5×5 cube graph has 125 nodes. The 98 boundary nodes (at least one
coordinate is 0 or 4) are observed. The 27 interior nodes (the 3×3×3
inner cube, all coordinates in {1,2,3}) are hidden. The model must
reconstruct the full distribution given only boundary values.

---

## New Utilities

### `graph_ot_fm/utils.py` — add two functions

```python
def make_cube_graph(size: int = 5) -> np.ndarray:
    """
    3D grid graph of size × size × size.
    Node (x,y,z) -> index x*size*size + y*size + z.
    Edges to 6-neighbors within bounds.
    Returns: (N, N) rate matrix with N = size^3.
    """
    N = size ** 3
    R = np.zeros((N, N))
    for x in range(size):
        for y in range(size):
            for z in range(size):
                i = x * size * size + y * size + z
                for dx, dy, dz in [
                    (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)
                ]:
                    nx_, ny_, nz_ = x+dx, y+dy, z+dz
                    if 0 <= nx_ < size and 0 <= ny_ < size and 0 <= nz_ < size:
                        j = nx_ * size * size + ny_ * size + nz_
                        R[i, j] = 1.0
    np.fill_diagonal(R, -R.sum(axis=1))
    return R


def cube_boundary_mask(size: int = 5) -> np.ndarray:
    """
    Returns binary mask: 1 for boundary nodes, 0 for interior.
    Boundary = any coordinate is 0 or (size-1).
    Interior = all coordinates in {1, ..., size-2}.
    """
    N = size ** 3
    mask = np.zeros(N)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                i = x * size * size + y * size + z
                if x == 0 or x == size-1 or y == 0 or y == size-1 or z == 0 or z == size-1:
                    mask[i] = 1.0
    return mask


def cube_node_depth(size: int = 5) -> np.ndarray:
    """
    Returns depth of each node: minimum distance to any boundary face.
    Boundary nodes have depth 0, next layer depth 1, center depth 2.
    For size=5: depths are 0, 1, or 2.
    """
    N = size ** 3
    depth = np.zeros(N)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                i = x * size * size + y * size + z
                d = min(x, size-1-x, y, size-1-y, z, size-1-z)
                depth[i] = d
    return depth
```

Export `make_cube_graph`, `cube_boundary_mask`, `cube_node_depth` from
`graph_ot_fm/__init__.py`.

---

## Dataset

Use a new `CubeBoundaryDataset` class, or adapt `InpaintingDataset` with
the fixed boundary mask. The cleaner approach is a dedicated class since
the mask is always the same:

```python
class CubeBoundaryDataset(torch.utils.data.Dataset):
    """
    Training data for boundary observation on a cube graph.
    
    Each sample: OT flow from boundary-only observation to the full
    clean distribution, conditioned on the boundary mask.
    
    The mask is FIXED (always boundary vs interior), unlike InpaintingDataset
    where masks are random. This simplifies precomputation.
    
    Constructor args:
        graph: GraphStructure (from make_cube_graph)
        clean_distributions: list of np.ndarray (N,)
        mask: np.ndarray (N,) — binary, 1=observed (boundary), 0=hidden (interior)
        n_samples: int = 10000
        seed: int = 42
    
    __getitem__ returns:
        mu:       (N,)      distribution at flow time tau
        tau:      (1,)      flow time
        context:  (N, 2)    [observed_value(a), mask(a)] per node
        R_target: (N, N)    factorized target rate matrix
        edge_index: (2, E)  graph edges
        n_nodes:  int       = 125
    """
```

This returns edge_index and n_nodes per sample (like TopologyGeneralizationDataset)
so it works with `train_flexible_conditional`. Even though the graph is fixed
across samples, using the flexible training loop avoids needing yet another
training function.

### Corrupting a distribution

```python
def apply_boundary_mask(mu_clean, mask):
    """Zero out interior nodes, renormalize boundary."""
    mu_obs = mu_clean * mask  # zero interior
    s = mu_obs.sum()
    if s > 1e-12:
        mu_obs /= s
    else:
        mu_obs = mask / mask.sum()  # uniform on boundary
    return mu_obs
```

---

## Source Distributions

Multi-peak (1-3 peaks) on the 125-node cube. Same generation as previous
experiments.

Important: generate peaks at both boundary AND interior locations. Track
where each peak is so we can evaluate interior vs boundary peak recovery
separately.

```python
def make_cube_multipeak(N, n_peaks, rng, mask=None):
    """
    Returns (dist, peak_nodes, n_interior_peaks).
    n_interior_peaks = count of peaks where mask[node] == 0.
    """
    peak_nodes = rng.choice(N, size=n_peaks, replace=False).tolist()
    weights = rng.dirichlet(np.full(n_peaks, 2.0))
    dist = np.ones(N) * 0.2 / N
    for node, w in zip(peak_nodes, weights):
        dist[node] += 0.8 * w
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    
    n_interior = sum(1 for p in peak_nodes if mask is not None and mask[p] == 0)
    return dist, peak_nodes, n_interior
```

Generate 300 source distributions for training. With random peak placement
on 125 nodes (27 interior, 98 boundary), roughly 22% of peaks will land
in the interior by chance.

---

## Training

- FlexibleConditionalGNNRateMatrixPredictor with context_dim=2
- hidden_dim=64 (start here, increase to 128 if needed)
- n_layers=6 (cube has diameter 12 in graph distance; 6 layers of message
  passing gives receptive field of 6 hops, reaching center from nearest
  face at depth 2. Should be sufficient.)
- 10000 training samples
- 1000 epochs, lr=5e-4, gradient clipping max_norm=1.0
- 1/(1-t) factorization applied
- loss_weighting='uniform' (default, configurable via CLI)

### CLI arguments

```python
parser.add_argument('--loss-weighting', type=str, default='uniform',
                    choices=['original', 'uniform', 'linear'])
parser.add_argument('--n-epochs', type=int, default=1000)
parser.add_argument('--hidden-dim', type=int, default=64)
parser.add_argument('--n-layers', type=int, default=6)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--cube-size', type=int, default=5)
parser.add_argument('--n-train-dists', type=int, default=300)
parser.add_argument('--n-samples', type=int, default=10000)
```

---

## Baselines

### 1. Boundary-only (naive)

Set interior nodes to the mean of observed boundary values, renormalize.

```python
def baseline_naive(mu_obs, mask):
    result = mu_obs.copy()
    obs_mean = mu_obs[mask == 1].mean()
    result[mask == 0] = obs_mean
    result /= result.sum()
    return result
```

### 2. Laplacian smoothing (Dirichlet problem)

Fix boundary values, fill interior by harmonic extension. This minimizes
the graph Laplacian energy subject to boundary constraints — the classic
physics-based solution.

```python
def baseline_laplacian(mu_obs, mask, R):
    """
    Solve the Dirichlet problem: find u_I that minimizes
    sum_{edges} (u_i - u_j)^2 subject to u_B = observed values.
    
    Solution: L_II * u_I = -L_IB * u_B
    where L = -R (the graph Laplacian, since R has negative diagonal).
    """
    N = len(mu_obs)
    L = -R.copy()  # graph Laplacian
    
    boundary_idx = np.where(mask == 1)[0]
    interior_idx = np.where(mask == 0)[0]
    
    L_II = L[np.ix_(interior_idx, interior_idx)]
    L_IB = L[np.ix_(interior_idx, boundary_idx)]
    u_B = mu_obs[boundary_idx]
    
    # Solve L_II * u_I = -L_IB * u_B
    u_I = np.linalg.solve(L_II, -L_IB @ u_B)
    
    result = mu_obs.copy()
    result[interior_idx] = u_I
    result = np.clip(result, 0, None)
    result /= result.sum()
    return result
```

---

## Evaluation

### Test cases

50 held-out source distributions. Categorize by interior peak status:
- Cases with all peaks on boundary (easiest)
- Cases with at least one interior peak (harder)
- Cases with dominant peak in interior (hardest)

For each test case:
1. Apply boundary mask to get observation.
2. Run learned model forward from observation, conditioned on (observation, mask).
3. Compute baselines (naive, Laplacian).
4. Compare all three to true source.

### Metrics

- **Full TV**: on all 125 nodes
- **Interior TV**: on 27 interior nodes only (isolates reconstruction quality)
- **Peak recovery (top-k)**: fraction of true peaks found
- **Interior peak recovery**: for peaks at interior nodes, are they recovered?
- **Depth analysis**: mean absolute error at depth 0 (boundary), 1, 2 (center)

---

## Plots (single figure, 2×3 grid)

- **Panel A**: Training loss curve.

- **Panel B**: Example reconstruction. Show 3 slices through the cube
  (z=0 bottom, z=2 middle, z=4 top) for 4 columns: observation,
  learned, Laplacian, true source. The z=2 slice is all interior — shows
  pure reconstruction quality. Layout: 3 rows × 4 columns of 5×5 heatmaps.

- **Panel C**: Full TV comparison. Bar chart: learned vs Laplacian vs naive.
  Split by "all peaks boundary" and "has interior peak" for each method.

- **Panel D**: Interior TV comparison. Same grouping as Panel C but
  computed only on the 27 interior nodes.

- **Panel E**: Reconstruction error vs depth. Line plot with depth on
  x-axis (0, 1, 2), mean absolute error on y-axis. Three lines: learned,
  Laplacian, naive. The learned model should degrade less steeply with
  depth than baselines.

- **Panel F**: Interior peak recovery. For cases with at least one interior
  peak: bar chart comparing top-k peak recovery for learned vs Laplacian
  vs naive. This is the headline metric — can the model find peaks that
  are completely hidden from direct observation?

---

## Validation Checks (print to console)

```
=== Experiment 12: Cube Boundary Observation ===
Graph: 5×5×5 cube (125 nodes, 98 boundary, 27 interior)
Training: initial loss = X.XXXX, final loss = X.XXXX

Full TV:
  Learned:     X.XXXX ± X.XXXX
  Laplacian:   X.XXXX ± X.XXXX
  Naive:       X.XXXX ± X.XXXX

Interior TV:
  Learned:     X.XXXX ± X.XXXX
  Laplacian:   X.XXXX ± X.XXXX
  Naive:       X.XXXX ± X.XXXX

Peak recovery (top-k, all cases):
  Learned:     XX%
  Laplacian:   XX%

Interior peak recovery (cases with interior peaks):
  Learned:     XX%
  Laplacian:   XX%
  Naive:       XX%

Depth analysis (mean abs error per node):
  Depth 0 (boundary): learned=X.XXXX, laplacian=X.XXXX, naive=X.XXXX
  Depth 1:            learned=X.XXXX, laplacian=X.XXXX, naive=X.XXXX
  Depth 2 (center):   learned=X.XXXX, laplacian=X.XXXX, naive=X.XXXX
```

---

## Expected Outcome

The Laplacian baseline is strong for smooth distributions but cannot recover
sharp interior features. Harmonic functions on a graph are smooth by
definition — they average over neighbors. A peak in the deep interior
gets smeared away by harmonic extension.

The learned model should:
- Match or trail the Laplacian on smooth cases (where harmonic extension
  is near-optimal)
- Beat the Laplacian on peaked cases, especially interior peaks
- Show less depth-dependent degradation than baselines

If the learned model beats the Laplacian on interior peak recovery, that
demonstrates it has learned a useful prior beyond smoothness — the "hold up"
result we're aiming for.

---

## Scaling Concerns

At 125 nodes, the exact solver computes:
- 125×125 cost matrix (trivial)
- OT coupling via LP (125×125 — still fast with POT)
- Geodesic cache: 125^2 pairs, matrix powers up to diameter 12.
  This may be slow. Monitor dataset generation time.
  If too slow, use the coupling-aware caching (only precompute pairs
  in the OT coupling support, which has at most 249 entries).

The GNN processes 125-node graphs, which is still small by GNN standards.
No scaling issues expected for the model itself.

---

## Dependencies

No new dependencies. Uses scipy for Laplacian solve (np.linalg.solve),
which is already available.
