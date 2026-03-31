# Experiment 12 (v3): Diffusion-Based Boundary Observation on 4×4×4 Cube

## What Was Wrong

The previous version read boundary values directly from the source distribution.
Since peaks are in the interior with ~80% mass on one node and ~20% spread
thinly across 124 others, the boundary values are nearly identical regardless
of which interior node has the peak. The context was uninformative — the model
couldn't learn because the input carried no signal about peak location.

## The Fix: Diffusion-Based Observation

A physical diffusion process propagates interior structure to the boundary.
The pipeline becomes:

1. Generate source distribution with interior peak
2. Diffuse: mu_diffused = mu_source @ expm(tau_diff * R)
3. Observe boundary only: mu_obs = boundary values of mu_diffused
4. Task: from boundary observations, recover the original source

Now the boundary signal is informative: diffusion spreads mass outward from
the peak, and the nearest boundary face receives more mass. The model can
learn the mapping from boundary diffusion patterns to interior peak locations.

This is exactly the inverse EEG setup: sources create signals on the surface
through physical propagation, and we invert.

## Why 4×4×4 First

The 5×5×5 cube has 27 interior nodes at depths 1-2 from the boundary.
The 4×4×4 cube has 8 interior nodes (a 2×2×2 inner cube), ALL at depth 1
from the boundary. This is the easiest nontrivial case:
- Every interior node is adjacent to boundary nodes
- The diffusion signal reaches the boundary in one hop
- Only 8 possible interior peak locations

If the model can't solve this, there's a fundamental issue. If it works,
scale to 5×5×5.

4×4×4 cube: 64 nodes, 56 boundary, 8 interior.

## Setup

### Graph

```python
R = make_cube_graph(size=4)  # 64 nodes
mask = cube_boundary_mask(size=4)  # 56 boundary, 8 interior
```

### Source distributions

Single peak at a random interior node:

```python
def make_interior_peak_dist(N, rng, interior_idx):
    """Single peak at a random interior node."""
    peak_node = int(rng.choice(interior_idx))
    dist = np.ones(N) * 0.2 / (N - 1)
    dist[peak_node] = 0.8
    dist += rng.normal(0, 0.01, N)
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist, peak_node
```

### Diffusion and observation

```python
def compute_boundary_observation(mu_source, R, mask, tau_diff=1.0):
    """
    Diffuse source, then observe boundary only.
    Returns the boundary observation (renormalized).
    """
    mu_diffused = mu_source @ expm(tau_diff * R)
    mu_obs = mu_diffused * mask  # zero out interior
    mu_obs = np.clip(mu_obs, 1e-12, None)
    mu_obs /= mu_obs.sum()
    return mu_obs
```

Use tau_diff = 1.0 as default. This gives moderate diffusion on a 4×4×4
cube — enough to spread the signal to the boundary faces without completely
homogenizing.

### Context

Per node: [mu_obs(a), mask(a)]
- boundary nodes: [diffused boundary value, 1.0]
- interior nodes: [0.0, 0.0]

context_dim = 2.

### Flow

Start from UNIFORM distribution (1/64 everywhere).
OT coupling: uniform → source.
Conditioning provides the boundary observation.

```python
mu_start = np.ones(N) / N
pi = compute_ot_coupling(mu_start, mu_source, cost)
```

## Dataset

```python
class CubeBoundaryDataset:
    def __init__(self, R, mask, clean_distributions, boundary_observations,
                 n_samples=10000, seed=42):
        """
        Args:
            R: rate matrix
            mask: boundary mask
            clean_distributions: list of source distributions
            boundary_observations: list of diffused boundary observations
                (one per clean distribution, precomputed)
            n_samples: training samples to generate
        """
        # For each pair:
        #   mu_start = uniform
        #   mu_target = clean_distributions[k]
        #   context = [boundary_observations[k], mask]
        #   OT coupling: uniform → clean
        #   Sample (mu_tau, tau, context, R_target) along the flow
```

Precompute boundary observations outside the dataset:

```python
# In main():
tau_diff = 1.0
interior_idx = np.where(mask == 0)[0]
train_sources = []
train_obs = []
for _ in range(n_train):
    mu_source, peak_node = make_interior_peak_dist(N, rng, interior_idx)
    mu_obs = compute_boundary_observation(mu_source, R, mask, tau_diff)
    train_sources.append(mu_source)
    train_obs.append(mu_obs)

dataset = CubeBoundaryDataset(
    R=R, mask=mask,
    clean_distributions=train_sources,
    boundary_observations=train_obs,
    n_samples=10000, seed=42)
```

## Training

- FlexibleConditionalGNNRateMatrixPredictor with context_dim=2
- hidden_dim=64, n_layers=4 (4×4×4 diameter is 9, but interior is
  only depth 1 — 4 layers is plenty)
- 1000 epochs, lr=5e-4, gradient clipping
- 200 training distributions (all interior peaks)
- 10000 training samples
- 1/(1-t) factorization
- Loss normalization by N*(N-1)

## Baselines

### 1. Naive baseline

Fill interior with mean boundary value:

```python
def baseline_naive(mu_obs, mask):
    result = mu_obs.copy()
    result[mask == 0] = mu_obs[mask == 1].mean()
    result /= result.sum()
    return result
```

### 2. Laplacian baseline

Harmonic extension from diffused boundary values. Note: this fills in
the interior smoothly based on the boundary pattern, which now carries
real signal. The Laplacian baseline should be MUCH better than in v2
because the boundary values are informative.

```python
def baseline_laplacian(mu_obs, mask, R):
    L = -R.copy()
    boundary_idx = np.where(mask == 1)[0]
    interior_idx = np.where(mask == 0)[0]
    L_II = L[np.ix_(interior_idx, interior_idx)]
    L_IB = L[np.ix_(interior_idx, boundary_idx)]
    u_B = mu_obs[boundary_idx]
    u_I = np.linalg.solve(L_II, -L_IB @ u_B)
    result = mu_obs.copy()
    result[interior_idx] = u_I
    result = np.clip(result, 0, None)
    result /= result.sum()
    return result
```

### 3. Exact inverse (new baseline)

Since the observation is diffused, we can try to invert the diffusion:
mu_recovered = mu_obs_full @ expm(-tau_diff * R). But we only observe
boundary values — the interior is missing. So the exact inverse isn't
directly applicable. We could:
- Fill interior with Laplacian, then apply exact inverse to the full
  distribution
- Or just skip this baseline (it requires knowing the full diffused
  distribution, which we don't have)

Skip exact inverse for now. The Laplacian is the meaningful baseline.

## Evaluation

### Test cases

50 held-out source distributions, all with interior peaks.

For each test case:
1. Compute boundary observation via diffusion
2. Start from uniform, condition on (observation, mask)
3. Run learned flow forward
4. Compare to true source, Laplacian baseline, naive baseline

### Metrics

- **Full TV**: all 64 nodes
- **Interior TV**: 8 interior nodes only
- **Peak recovery (top-1)**: does the single highest node in the recovered
  distribution match the true peak? With 8 interior nodes and 1 peak,
  top-1 is the natural metric.
- **Peak location (adjacent)**: is the recovered top-1 node within
  Manhattan distance 1 of the true peak?
- **Per-interior-node error**: with only 8 interior nodes, we can show
  the error at each one individually

## Plots (single figure, 2×3 grid)

- **Panel A**: Training loss curve.

- **Panel B**: Example reconstructions. Show 2 horizontal slices (z=0
  boundary, z=1 interior, z=2 interior, z=3 boundary) for: observation,
  learned, Laplacian, true. The z=1 and z=2 slices contain interior nodes.
  Layout: 4 rows × 4 cols of 4×4 heatmaps.

  OR simpler: show the 8 interior node values as a bar chart for one
  test case — true vs learned vs Laplacian. This directly shows whether
  the model finds the right peak.

- **Panel C**: Full TV comparison bar chart (learned vs Laplacian vs naive).

- **Panel D**: Interior TV comparison bar chart.

- **Panel E**: Peak recovery (top-1 accuracy) bar chart for all three methods.
  This is the headline metric: can the model identify which of the 8 interior
  nodes is the source?

- **Panel F**: Interior node values for 3 test cases. For each: 8 bars
  showing true (dark), learned (blue), Laplacian (green) values at the
  8 interior nodes. Shows whether the learned model produces sharper
  peaks than the Laplacian.

## Validation Checks (print to console)

```
=== Experiment 12: Cube Boundary (4×4×4, diffusion-based) ===
Graph: 4×4×4 cube (64 nodes, 56 boundary, 8 interior)
Diffusion time: tau_diff=1.0

Full TV:
  Learned:     X.XXXX ± X.XXXX
  Laplacian:   X.XXXX ± X.XXXX
  Naive:       X.XXXX ± X.XXXX

Interior TV:
  Learned:     X.XXXX ± X.XXXX
  Laplacian:   X.XXXX ± X.XXXX
  Naive:       X.XXXX ± X.XXXX

Peak recovery (top-1 = correct interior node):
  Learned:     XX%
  Laplacian:   XX%
  Naive:       XX%

Peak location (top-1 within Manhattan dist 1):
  Learned:     XX%
  Laplacian:   XX%
  Naive:       XX%
```

## Expected Outcome

With diffusion-based boundary observations:
- Boundary values carry real signal about interior peak location
- Laplacian baseline should do reasonably well (smooth interpolation
  of an informative boundary signal) — maybe 30-50% peak recovery
- Learned model should beat Laplacian by producing sharper interior
  reconstructions — maybe 60-80% peak recovery
- Naive baseline (uniform interior fill) should be worst

If the learned model beats the Laplacian on peak recovery, we have the
"hold up" result. If it doesn't, the 4×4×4 cube with 8 interior nodes
is simple enough to diagnose exactly what's going wrong.

## After Validation

If 4×4×4 works:
1. Scale to 5×5×5 (deeper interior, harder problem)
2. Add variable diffusion time as conditioning (like Ex8)
3. Add multi-peak sources
4. Move to real cortical mesh for inverse EEG

## Dependencies

No new dependencies.
