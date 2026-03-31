# Experiment 12c: `ex12c_cube_posterior.py` — Posterior Sampling on Cube Boundary

## Motivation

Experiment 12b demonstrated strong interior reconstruction from boundary
observations on the 5×5×5 cube (beating the Laplacian baseline). Experiment
13b attempted posterior sampling on the sparse sensor setup but the
underlying point estimate wasn't validated.

This experiment adds posterior sampling to the setup we KNOW works (12b).
Same cube, same boundary observation, same diffusion — but trained with
Dirichlet(1,...,1) starts. This isolates the posterior sampling question
from any issues with the sparse sensor setup.

## Setup

Identical to Experiment 12b except:
- **Starting distribution**: Dirichlet(1,...,1) samples instead of uniform
- **Training**: each sample starts from a DIFFERENT Dirichlet draw
- **Inference**: K=20 Dirichlet starts per test case → K posterior samples

### Physical setup (same as 12b)
- 5×5×5 cube graph (125 nodes, 98 boundary, 27 interior)
- Source distributions: 1-3 peaks anywhere on the cube
- Diffusion: mu_diffused = mu_source @ expm(tau_diff * R), tau_diff in [0.5, 2.0]
- Observation: boundary values of diffused distribution
- Context per node: [mu_obs(a), mask(a), tau_diff] — context_dim = 3

## Dataset

Adapt the CubeBoundaryDataset to use Dirichlet starts:

```python
class CubePosteriorDataset(torch.utils.data.Dataset):
    """
    Like CubeBoundaryDataset but each sample starts from a Dirichlet(1,...,1)
    draw instead of a fixed uniform distribution.
    
    Multiple Dirichlet starts per source ensure the model sees diverse
    starting points for the same target.
    """
    def __init__(self, R, mask, source_obs_pairs, n_starts_per_pair=10,
                 n_samples=15000, seed=42):
        rng = np.random.default_rng(seed)
        N = R.shape[0]
        graph_struct = GraphStructure(R)
        cost = compute_cost_matrix(graph_struct)
        cache = GeodesicCache(graph_struct)
        edge_index = rate_matrix_to_edge_index(R)
        
        # For each source-obs pair, precompute OT couplings from
        # multiple Dirichlet starts
        all_triples = []
        for pair in source_obs_pairs:
            mu_source = pair['mu_source']
            mu_obs = pair['mu_obs']
            tau_diff = pair['tau_diff']
            
            for _ in range(n_starts_per_pair):
                mu_start = rng.dirichlet(np.ones(N))
                pi = compute_ot_coupling(mu_start, mu_source, cost)
                cache.precompute_for_coupling(pi)
                all_triples.append({
                    'mu_start': mu_start,
                    'mu_source': mu_source,
                    'mu_obs': mu_obs,
                    'tau_diff': tau_diff,
                    'coupling': pi,
                })
        
        self.samples = []
        for _ in range(n_samples):
            triple = all_triples[int(rng.integers(len(all_triples)))]
            tau = float(rng.uniform(0.0, 0.999))
            
            mu_tau = marginal_distribution_fast(cache, triple['coupling'], tau)
            R_target = (1.0 - tau) * marginal_rate_matrix_fast(
                cache, triple['coupling'], tau)
            
            context = np.stack([
                triple['mu_obs'],
                mask,
                np.full(N, triple['tau_diff'])
            ], axis=-1)  # (N, 3)
            
            self.samples.append((
                torch.tensor(mu_tau, dtype=torch.float32),
                torch.tensor([tau], dtype=torch.float32),
                torch.tensor(context, dtype=torch.float32),
                torch.tensor(R_target, dtype=torch.float32),
                edge_index,
                N,
            ))
```

### Scaling concern

With 300 source distributions × 10 Dirichlet starts = 3000 OT couplings
to precompute on a 125-node graph. Each coupling is a 125×125 LP solve.
This may take several minutes. Monitor and report dataset generation time.

If too slow, reduce to 5 starts per pair (1500 couplings) or 200 source
distributions.

## Training

- FlexibleConditionalGNNRateMatrixPredictor with context_dim=3
- hidden_dim=128, n_layers=6 (same as 12b)
- 300 source distributions, 10 Dirichlet starts each
- 15000 training samples
- 1000 epochs, lr=5e-4, gradient clipping
- 1/(1-t) factorization, loss normalization by N*(N-1)

## Inference

```python
def sample_posterior_cube(model, mu_obs, mask, tau_diff, edge_index, N,
                          K=20, n_steps=200, device='cpu', seed=42):
    rng = np.random.default_rng(seed)
    context = np.stack([mu_obs, mask, np.full(N, tau_diff)], axis=-1)
    
    samples = []
    for k in range(K):
        mu_start = rng.dirichlet(np.ones(N))
        _, traj = sample_trajectory_flexible(
            model, mu_start, context, edge_index,
            n_steps=n_steps, device=device)
        samples.append(traj[-1])
    
    return samples
```

## Evaluation

### Test cases

50 held-out source distributions (same mix as 12b). For each:
1. Compute boundary observation via diffusion
2. Generate K=20 posterior samples
3. Compute summary statistics
4. Compare to 12b point estimate and Laplacian baseline

### Metrics

**Reconstruction quality:**
- TV of posterior mean vs true source
- Compare to 12b uniform-start point estimate (should be similar or better)
- Compare to Laplacian baseline

**Calibration:**
- Per-node correlation between posterior std and |posterior mean - true|
- Split by depth (boundary, depth 1, depth 2)
- Calibration should be strongest at interior nodes where uncertainty is
  highest and most meaningful

**Diversity:**
- Mean pairwise TV between posterior samples
- Should be moderate (not collapsed like MC dropout, not random like
  unconditioned Dirichlet)
- Compare diversity for cases with boundary peaks (low ambiguity, lower
  diversity expected) vs interior peaks (high ambiguity, higher diversity)

**Conditional probabilities:**
- Define 8 octant regions on the cube
- For each test case: does P(correct octant active) rank highest?
- For interior peaks: what is P(peak at depth 2 | peak in correct octant)?

**Interior reconstruction specifically:**
- Interior TV of posterior mean vs 12b point estimate vs Laplacian
- This is where 12b already excels — posterior mean should maintain
  or improve this advantage

## Plots (single figure, 2×3 grid)

- **Panel A**: Training loss curve.

- **Panel B**: Posterior visualization for one test case with interior peak.
  Show cube middle slice (z=2) for: true source, posterior mean, posterior
  std, and 3 individual samples. The std should be highest near the peak
  location (multiple plausible positions) and low far from it.

- **Panel C**: Calibration scatter plot. Per-node std vs |mean - true|,
  colored by depth. Report correlation r. Compare to 12b (which has no
  uncertainty) — the posterior provides calibrated uncertainty where the
  point estimate gives no information about reliability.

- **Panel D**: Reconstruction comparison. Bar chart: posterior mean TV,
  12b point estimate TV, Laplacian TV. Split by "boundary peaks" and
  "interior peaks." Posterior mean should match or beat 12b.

- **Panel E**: Diversity vs ambiguity. Scatter plot: x-axis = true peak
  depth (0, 1, 2), y-axis = mean pairwise TV between posterior samples.
  Deeper peaks should produce more diverse posteriors (more ambiguity
  in reconstruction). This validates that the diversity is meaningful,
  not just noise.

- **Panel F**: Conditional probability analysis. For 3 test cases with
  interior peaks: bar chart of P(octant active) across 8 octants, with
  the true octant marked. The posterior should assign highest probability
  to the correct octant.

## Validation Checks

```
=== Experiment 12c: Posterior Sampling on Cube Boundary ===
K=20 posterior samples from Dirichlet(1,...,1) prior

Reconstruction (posterior mean vs baselines):
  Posterior mean TV:    X.XXXX ± X.XXXX
  12b point est TV:     X.XXXX  [reference from 12b]
  Laplacian TV:         X.XXXX

Interior TV:
  Posterior mean:       X.XXXX ± X.XXXX
  12b point estimate:   X.XXXX
  Laplacian:            X.XXXX

Calibration (correlation of std vs |error|):
  Overall:    r = X.XX
  Boundary:   r = X.XX
  Depth 1:    r = X.XX
  Depth 2:    r = X.XX

Diversity (mean pairwise TV):
  Boundary peak cases:  X.XXXX
  Interior peak cases:  X.XXXX
  (interior should be higher — more ambiguity)

Conditional probability (correct octant identified):
  All cases:           XX%
  Interior peak cases: XX%
```

## Expected Outcome

Since 12b worked well with uniform starts, the posterior mean from
Dirichlet starts should achieve similar reconstruction quality. The
additional value is the uncertainty information:

- Calibrated std maps (r > 0.3) showing where the reconstruction is
  reliable vs uncertain
- Higher diversity for interior peaks than boundary peaks (meaningful
  uncertainty that reflects actual ambiguity)
- Correct octant identification from conditional probabilities

If the posterior mean TV is much worse than 12b's point estimate, the
Dirichlet starts are too diverse and the model can't converge them all
to the same target. In that case, try Dirichlet(5,...,5) or
Dirichlet(10,...,10) for starts closer to uniform.

## Dependencies

No new dependencies.
