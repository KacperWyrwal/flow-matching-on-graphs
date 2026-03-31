# Amendment: Experiment 12 — Diagnosis and Fixes

## Observed Issues

The learned model performs worse than the Laplacian baseline, especially for
interior peaks. Several possible causes:

### 1. Starting from masked observation (most likely issue)

The flow starts from the masked observation: boundary values present, interior
zeroed. This creates a pathological starting point:
- Boundary nodes have inflated mass (renormalized after zeroing 27/125 nodes)
- Interior nodes have exactly zero mass
- The flow must CREATE mass at interior nodes from nothing

This is the concentrating direction problem again. Even with the 1/(1-t)
factorization, moving mass from populated boundary nodes to empty interior
nodes requires large rate matrix entries on the boundary-interior edges.

**Fix: Start from uniform.** The flow starts from uniform (1/125 everywhere),
and the conditioning (observation + mask) provides all the information about
where mass should go. The flow redistributes from uniform to the target,
which is a balanced transport problem — some nodes gain mass, others lose it,
but no node starts at zero.

This mirrors the generative modeling setup (Ex7) which worked well:
uniform → structured distribution, conditioned on context.

### 2. Only 2 message-passing layers (contributing factor)

The cube has diameter 12. With 2 layers, the receptive field is 2 hops.
Information from boundary nodes can only reach 2 steps into the interior.
The deep interior (depth 2, center of cube) is 2 hops from the nearest
boundary face — barely within reach of 2 layers.

**Fix: Use 6 layers** as originally specified. This was reduced to 2 for
the initial run but the cube geometry demands more depth.

### 3. Problem complexity with mixed peak counts

Training on 1-3 peaks simultaneously with 300 distributions may spread
the model too thin, especially since interior peaks are rare (~22% of
random placements land in the 27/125 interior nodes).

**Fix: Start with single-peak, interior-only sources** to validate the
architecture can recover interior structure at all. Then scale up.

### 4. Loss scale

The loss plateaus around 15-20, much higher than previous experiments.
This could indicate the target rate matrices are particularly large for
this problem (mass transport from boundary to deep interior over multiple
hops), even with factorization.

**Fix: Check the magnitude of the factorized targets.** If they're much
larger than in Ex8/Ex11, may need to increase model capacity or adjust
the loss weighting.

---

## Revised Experiment: `ex12_cube_boundary.py` (v2)

### Changes from v1

1. **Start from uniform distribution** instead of masked observation.
   The flow transports uniform → clean, conditioned on (observation, mask).

2. **Single interior peak only** for initial validation.
   All sources have exactly 1 peak, and the peak is always in the interior
   (one of the 27 interior nodes). This makes the task clear and focused:
   given boundary observations, find the interior peak.

3. **6 message-passing layers** (matching original spec).

4. **After validation**, scale up to mixed peak counts and locations.

### Dataset changes

```python
class CubeBoundaryDataset:
    def __init__(self, R, mask, clean_distributions, n_samples=10000, seed=42):
        # ...
        
        for mu_clean in clean_distributions:
            # Compute the boundary observation (for context)
            mu_obs = apply_boundary_mask(mu_clean, mask)
            
            # Starting point: UNIFORM, not the masked observation
            mu_start = np.ones(N) / N
            
            # OT coupling: uniform → clean (NOT observation → clean)
            pi = compute_ot_coupling(mu_start, mu_clean, cost)
            cache.precompute_for_coupling(pi)
            pairs.append((mu_obs, mu_clean, mu_start, pi))
        
        for _ in range(n_samples):
            mu_obs, mu_clean, mu_start, pi = pairs[...]
            tau = float(rng.uniform(0.0, 0.999))
            
            # Flow goes from uniform to clean
            mu_tau = marginal_distribution_fast(cache, pi, tau)
            R_target_np = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi, tau)
            
            # Context: boundary observation + mask (provides the signal)
            context = np.stack([mu_obs, mask], axis=-1)
            
            self.samples.append((mu_tau, tau, context, R_target, edge_index, N))
```

Key change: the OT coupling is between UNIFORM and CLEAN, not between
OBSERVATION and CLEAN. The observation only appears in the context.

### Inference changes

```python
# Start from uniform, not from observation
mu_start = np.ones(N) / N
context = np.stack([mu_obs, mask], axis=-1)

_, traj = sample_trajectory_flexible(
    model, mu_start, context, edge_index, n_steps=200, device=device)
mu_recovered = traj[-1]
```

### Source generation: single interior peak

```python
def make_interior_peak_dist(N, rng, mask, interior_idx):
    """
    Generate a distribution with exactly 1 peak at a random interior node.
    """
    peak_node = int(rng.choice(interior_idx))
    dist = np.ones(N) * 0.2 / (N - 1)
    dist[peak_node] = 0.8
    dist += rng.normal(0, 0.01, N)
    dist = np.clip(dist, 1e-6, None)
    dist /= dist.sum()
    return dist, [peak_node]
```

Generate 200 training distributions, all with interior peaks.

### Evaluation

Same metrics as before but simpler since all peaks are interior:
- Full TV and Interior TV
- Peak recovery (does top-1 match the true interior peak?)
- Depth analysis (is the center harder than depth 1?)
- Comparison to Laplacian and naive baselines

### CLI

```python
parser.add_argument('--start-from', type=str, default='uniform',
                    choices=['uniform', 'observation'],
                    help='Starting distribution for the flow')
parser.add_argument('--peak-mode', type=str, default='interior',
                    choices=['interior', 'any', 'mixed'],
                    help='Where to place peaks: interior only, anywhere, or 1-3 mixed')
```

This allows comparing uniform vs observation start, and interior-only vs
mixed peaks, without separate experiment scripts.

### Expected outcome

With uniform start and interior-only peaks:
- The flow is balanced (uniform → peaked), which is what works in Ex7
- The model only needs to learn one thing: where to concentrate mass,
  guided by the boundary observations
- The Laplacian baseline cannot recover interior peaks (harmonic functions
  have no interior extrema)
- Even a modest learned model should beat Laplacian on this task

If this works, scale up to mixed peaks. If it doesn't, the problem is
deeper (architecture, scale, training signal quality).
