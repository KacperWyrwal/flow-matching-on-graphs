# Experiment 13b: `ex13b_posterior_sampling.py`

## Motivation

Experiment 13 produces point estimates of source distributions from sparse
sensor measurements. But the inverse problem is underdetermined (20 sensors,
125 unknowns) — multiple source configurations can explain the same
measurements. A point estimate hides this ambiguity.

Experiment 13b produces posterior samples: multiple plausible source
reconstructions for the same observation. This enables uncertainty
quantification, conditional probability computation, and hierarchical
reasoning about source location — capabilities critical for clinical
decision making.

The key idea: train the flow from Dirichlet(1,...,1) starts (uniform
distribution over the simplex) to source distributions, conditioned on
sensor observations. At inference, sample K different starting points from
Dirichlet(1,...,1), run the conditioned flow from each, and collect K
posterior samples.

## Relationship to Meta-Level Framework

This is the meta-level framework in action. Each posterior sample is a
distribution on the graph. The collection of K samples is an empirical
distribution over distributions — a point on the meta-simplex. The flow
transports the prior over distributions (Dirichlet) to the posterior over
distributions (conditioned on observations).

## Setup

Same physical setup as Experiment 13:
- 5×5×5 cube graph (125 nodes)
- 20 sparse sensors on the boundary
- Mixing matrix A with Gaussian spatial decay (sigma=1.5)
- Source distributions with 1-3 interior peaks
- Diffusion + sensor measurement: y = A @ (mu_source @ expm(tau * R))
- Backprojection as context: mu_backproj = A^T @ y, normalized

### Prior distribution

Dirichlet(1, ..., 1) over the 125-node simplex. This is the uniform
distribution over all possible distributions on the graph — the maximum
entropy, uninformative prior.

```python
def sample_dirichlet_prior(N, rng):
    """Sample from Dir(1,...,1) = uniform over simplex."""
    return rng.dirichlet(np.ones(N))
```

## Training

### Training data generation

For each training sample:
1. Pick a source distribution mu_source (multi-peak, as in Ex13)
2. Compute sensor observation y and backprojection mu_backproj
3. Sample a starting point mu_start ~ Dirichlet(1, ..., 1)
4. Compute OT coupling: mu_start → mu_source
5. Sample along the flow: (mu_tau, tau, context, R_target)
6. Context = [mu_backproj(a), tau_diff] per node (same as Ex13)

Key difference from Ex13: each training sample has a DIFFERENT starting
point drawn from the Dirichlet prior. The model must learn to transport
from any point on the simplex to the correct source, conditioned on the
observation.

```python
class PosteriorSamplingDataset(torch.utils.data.Dataset):
    """
    Training data for posterior sampling via flow matching.
    
    Each sample starts from a Dirichlet(1,...,1) draw and flows toward
    the source distribution, conditioned on the sensor observation.
    
    Multiple Dirichlet starts per source distribution ensure the model
    sees diverse starting points for the same target.
    """
    def __init__(self, R, sensor_nodes, mixing_matrix, mask,
                 source_obs_pairs, n_starts_per_pair=10,
                 n_samples=15000, seed=42):
        rng = np.random.default_rng(seed)
        N = R.shape[0]
        graph_struct = GraphStructure(R)
        cost = compute_cost_matrix(graph_struct)
        cache = GeodesicCache(graph_struct)
        edge_index = rate_matrix_to_edge_index(R)
        
        # For each source-obs pair, precompute OT couplings from
        # multiple Dirichlet starts
        all_triples = []  # (mu_start, mu_source, mu_backproj, tau_diff, coupling)
        for pair in source_obs_pairs:
            mu_source = pair['mu_source']
            mu_backproj = pair['mu_backproj']
            tau_diff = pair['tau_diff']
            
            for _ in range(n_starts_per_pair):
                mu_start = rng.dirichlet(np.ones(N))
                pi = compute_ot_coupling(mu_start, mu_source, cost)
                cache.precompute_for_coupling(pi)
                all_triples.append((mu_start, mu_source, mu_backproj,
                                    tau_diff, pi))
        
        # Sample training tuples
        self.samples = []
        for _ in range(n_samples):
            mu_start, mu_source, mu_backproj, tau_diff, pi = \
                all_triples[int(rng.integers(len(all_triples)))]
            tau = float(rng.uniform(0.0, 0.999))
            
            mu_tau = marginal_distribution_fast(cache, pi, tau)
            R_target = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi, tau)
            
            context = np.stack([mu_backproj, np.full(N, tau_diff)], axis=-1)
            
            self.samples.append((
                torch.tensor(mu_tau, dtype=torch.float32),
                torch.tensor([tau], dtype=torch.float32),
                torch.tensor(context, dtype=torch.float32),
                torch.tensor(R_target, dtype=torch.float32),
                edge_index,
                N,
            ))
```

### Training parameters

- FlexibleConditionalGNNRateMatrixPredictor with context_dim=2
- hidden_dim=128, n_layers=6 (same as Ex13)
- 200 source distributions, 10 Dirichlet starts per source = 2000 triples
- 15000 training samples
- 1000 epochs, lr=5e-4, gradient clipping
- 1/(1-t) factorization, loss normalization

## Inference: Posterior Sampling

```python
def sample_posterior(model, mu_backproj, tau_diff, edge_index, N,
                     K=20, n_steps=200, device='cpu', seed=42):
    """
    Generate K posterior samples for a given observation.
    
    Args:
        model: trained flow matching model
        mu_backproj: (N,) backprojected sensor readings
        tau_diff: diffusion time
        K: number of posterior samples
        
    Returns:
        samples: list of K np.ndarray (N,) — posterior source estimates
    """
    rng = np.random.default_rng(seed)
    context = np.stack([mu_backproj, np.full(N, tau_diff)], axis=-1)
    
    samples = []
    for k in range(K):
        mu_start = rng.dirichlet(np.ones(N))
        _, traj = sample_trajectory_flexible(
            model, mu_start, context, edge_index,
            n_steps=n_steps, device=device)
        samples.append(traj[-1])
    
    return samples
```

## Posterior Analysis

### Summary statistics

```python
def posterior_summary(samples):
    """
    Compute mean, std, and quantiles from K posterior samples.
    
    Returns:
        mean: (N,) — posterior mean distribution
        std: (N,) — per-node standard deviation
        q05: (N,) — 5th percentile per node
        q95: (N,) — 95th percentile per node
    """
    S = np.array(samples)  # (K, N)
    return {
        'mean': S.mean(axis=0),
        'std': S.std(axis=0),
        'q05': np.percentile(S, 5, axis=0),
        'q95': np.percentile(S, 95, axis=0),
    }
```

### Conditional probabilities

```python
def conditional_analysis(samples, regions):
    """
    Compute conditional probabilities between regions.
    
    Args:
        samples: list of K posterior samples (N,)
        regions: dict mapping region_name → list of node indices
    
    Returns:
        activation_probs: P(region active) for each region
        conditional_probs: P(region B active | region A active)
    """
    K = len(samples)
    threshold = 1.5 / N  # node is "active" if mass > 1.5x uniform
    
    # Per-sample, per-region activation
    activations = {}
    for name, nodes in regions.items():
        activations[name] = [
            sum(s[n] > threshold for n in nodes) > 0
            for s in samples
        ]
    
    # Marginal probabilities
    activation_probs = {
        name: np.mean(acts) for name, acts in activations.items()
    }
    
    # Conditional probabilities: P(B | A)
    conditional_probs = {}
    for a_name in regions:
        for b_name in regions:
            if a_name == b_name:
                continue
            a_active = activations[a_name]
            b_active = activations[b_name]
            n_a = sum(a_active)
            if n_a > 0:
                n_ab = sum(a and b for a, b in zip(a_active, b_active))
                conditional_probs[(a_name, b_name)] = n_ab / n_a
    
    return activation_probs, conditional_probs
```

### Region definitions for the cube

```python
# Define regions as octants of the 5×5×5 cube
REGIONS = {
    'front_top_left':     nodes where x<2.5, y<2.5, z>=2.5,
    'front_top_right':    nodes where x<2.5, y>=2.5, z>=2.5,
    'front_bottom_left':  nodes where x<2.5, y<2.5, z<2.5,
    'front_bottom_right': nodes where x<2.5, y>=2.5, z<2.5,
    'back_top_left':      nodes where x>=2.5, y<2.5, z>=2.5,
    'back_top_right':     nodes where x>=2.5, y>=2.5, z>=2.5,
    'back_bottom_left':   nodes where x>=2.5, y<2.5, z<2.5,
    'back_bottom_right':  nodes where x>=2.5, y>=2.5, z<2.5,
}

# Or simpler: depth-based regions
DEPTH_REGIONS = {
    'boundary': nodes at depth 0,
    'shallow':  nodes at depth 1,
    'center':   nodes at depth 2,
}
```

## Comparison: GNN + Dropout

As a baseline for uncertainty quantification, use MC dropout on the
GNN+softmax model from Ex13:

```python
def mc_dropout_samples(model, context, edge_index, K=20):
    """
    Run model K times with dropout enabled at test time.
    """
    model.train()  # enable dropout
    samples = []
    for _ in range(K):
        with torch.no_grad():
            mu = model(context, edge_index)
        samples.append(mu.cpu().numpy())
    model.eval()
    return samples
```

Note: the GNN+softmax model needs dropout layers added during training
for this to work. Add dropout (p=0.1) after each message passing layer.

## Evaluation

### Test cases

30 test cases (10 per peak count). For each:
- Generate K=20 posterior samples from the flow model
- Generate K=20 MC dropout samples from the GNN+softmax model
- Compute summary statistics and conditional probabilities

### Metrics

**Reconstruction quality (mean):**
- TV of posterior mean vs true source
- Compare to Ex13 point estimate — posterior mean should be comparable or better

**Calibration:**
- Per-node: plot std vs absolute error. If calibrated, high std should
  correspond to high error.
- Compute calibration score: correlation between per-node std and per-node
  |mean - true|.
- Compare calibration: flow samples vs MC dropout

**Diversity:**
- Mean pairwise TV between posterior samples. Higher = more diverse.
  Too low means the model is collapsing to a point (no uncertainty).
  Too high means the model hasn't converged to a useful posterior.
- Compare diversity: flow vs MC dropout

**Conditional probabilities:**
- For cases with known ground truth (peak in a specific octant), check
  whether the conditional probabilities correctly identify the octant.
- Example: if true peak is in front_top_left, P(front_top_left active)
  should be high, and P(back_bottom_right active | front_top_left active)
  should be low (peaks don't co-occur there in the prior).

## Plots (single figure, 2×3 grid)

- **Panel A**: Training loss curve.

- **Panel B**: Posterior visualization for one test case. Show on the
  5×5×5 cube (middle slice z=2):
  - True source
  - Posterior mean
  - Posterior std (uncertainty map)
  - 4 individual posterior samples
  Layout: 2 rows × 4 cols of 5×5 heatmaps. The std map should highlight
  the region around the true peak (high uncertainty where the peak could
  be) and be low far from it.

- **Panel C**: Calibration plot. Scatter plot of per-node std vs per-node
  |mean - true|, aggregated across all test cases. Color points by depth
  (boundary vs interior). Include correlation coefficient. Compare flow
  model (blue) vs MC dropout (orange).

- **Panel D**: Sample diversity. Box plot of pairwise TV between posterior
  samples, for flow model vs MC dropout. The flow model should show
  meaningful diversity (not collapsed, not random).

- **Panel E**: Conditional probability example. For one test case with
  a peak in a specific octant: bar chart showing P(region active) for
  each of the 8 octants. The true octant should have highest probability.
  Show for both flow model and MC dropout.

- **Panel F**: Reconstruction quality comparison. Bar chart: TV of
  posterior mean (flow), TV of MC dropout mean, TV of point estimates
  from Ex13 (learned flow, GNN+softmax, LASSO). Shows whether posterior
  mean is competitive with or better than point estimates.

## Validation Checks

```
=== Experiment 13b: Posterior Sampling ===
K=20 posterior samples per test case

Reconstruction (posterior mean):
  Flow posterior mean TV:     X.XXXX ± X.XXXX
  MC dropout mean TV:         X.XXXX ± X.XXXX
  (Ex13 flow point estimate): X.XXXX  [reference]
  (Ex13 LASSO):               X.XXXX  [reference]

Calibration (correlation of std vs |error|):
  Flow model:    r = X.XX
  MC dropout:    r = X.XX

Diversity (mean pairwise TV between samples):
  Flow model:    X.XXXX
  MC dropout:    X.XXXX

Conditional probability accuracy (correct octant identified):
  Flow model:    XX% of test cases
  MC dropout:    XX% of test cases
```

## Expected Outcome

- **Posterior mean** should match or slightly beat the Ex13 point estimate
  (averaging reduces noise).
- **Calibration** should be positive (std correlates with error) for the
  flow model. MC dropout calibration is often poor.
- **Diversity** should be moderate — enough to capture genuine ambiguity
  but not random. Flow diversity should be more structured (spatially
  coherent variations) than MC dropout (random node-level fluctuations).
- **Conditional probabilities** should correctly identify the source octant
  in most cases, demonstrating that the posterior samples carry meaningful
  structural information.

The headline result: the flow model produces geometrically meaningful
uncertainty estimates (smooth, graph-aware, spatially coherent) while
MC dropout produces noisy, uncorrelated uncertainty. This justifies the
additional complexity of the flow matching approach.

## Dependencies

scikit-learn (already added for Ex13 LASSO baseline).
No other new dependencies.
