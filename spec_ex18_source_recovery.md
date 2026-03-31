# Experiment 18: Diffusion Source Recovery — Broad Generalization

## Motivation

Upgrade the diffusion source recovery experiment (Ex10-11) to test
generalization across three axes simultaneously:
1. **Graph topology:** Train on diverse graphs, test on unseen types
2. **Graph size:** Train on N=15-60, test on N=8-150
3. **Diffusion time:** Train on τ ∈ [0.3, 1.5], test on τ ∈ [0.1, 3.0]

Uses the graph zoo from Ex17 and arbitrary source distributions
(not parameterized by peak count). Evaluation groups by difficulty
(TV between source and observation) rather than by peak count.

## Task

**Forward (training data generation):** Given source distribution μ on
graph G, diffuse to get observation: obs = μ @ expm(τR).

**Inverse (model task):** Given observation obs, graph G, and τ,
recover the source μ.

## Graph Zoo

Same as Ex17 (see that spec for all graph constructors).

### Training graphs (~30 graphs, sizes 15-60)
Grids, cycles, random geometric, Barabasi-Albert, Watts-Strogatz,
random trees, mesh shapes (square, circle, triangle), barbell graphs.

### Test: unseen topologies (sizes 15-60)
New grid sizes, ladder, mesh from unseen shapes (L_shape, star, annulus),
stochastic block model, Petersen graph.

### Test: unseen sizes (N=8-150)
Small: grid 2×4, cycle 8, rgg 10, tree 10.
Large: grid 8×10, grid 10×10, rgg 100, rgg 150, mesh 100, BA 100.

## Source Distributions

General distributions — not parameterized by number of peaks.
The variety should cover the full entropy spectrum.

```python
def generate_source_distribution(N, points, rng):
    """
    Generate a random source distribution with varied structure.
    
    Types (sampled uniformly):
    - single_peak: one dominant node
    - multi_peak: 2-5 peaks with random weights
    - gradient: linear gradient in random direction
    - smooth_random: sum of 2-8 Gaussian bumps
    - clustered: mass concentrated in a random subgraph neighborhood
    - near_uniform: slightly perturbed uniform (high entropy)
    - power_law: power-law-like mass assignment (few heavy, many light)
    """
    ic_type = rng.choice([
        'single_peak', 'multi_peak', 'gradient', 'smooth_random',
        'clustered', 'near_uniform', 'power_law',
    ])
    
    if ic_type == 'single_peak':
        mu = np.ones(N) * 0.01
        mu[rng.integers(N)] += 1.0
    
    elif ic_type == 'multi_peak':
        n_peaks = int(rng.integers(2, 6))
        peaks = rng.choice(N, size=n_peaks, replace=False)
        weights = rng.dirichlet(np.ones(n_peaks))
        mu = np.ones(N) * 1e-3
        for p, w in zip(peaks, weights):
            mu[p] += w
    
    elif ic_type == 'gradient':
        direction = rng.standard_normal(2)
        direction /= np.linalg.norm(direction) + 1e-10
        proj = points @ direction
        proj = proj - proj.min()
        mu = proj / (proj.max() + 1e-10) + 0.05
    
    elif ic_type == 'smooth_random':
        n_bumps = int(rng.integers(2, 9))
        mu = np.zeros(N)
        for _ in range(n_bumps):
            center = rng.uniform(points.min(0), points.max(0))
            sigma = rng.uniform(0.05, 0.3)
            dists = np.linalg.norm(points - center, axis=1)
            mu += rng.uniform(0.5, 2.0) * np.exp(-0.5 * (dists/sigma)**2)
        mu += 1e-3
    
    elif ic_type == 'clustered':
        # Pick a seed node, spread mass to its k-hop neighborhood
        seed = int(rng.integers(N))
        k_hops = int(rng.integers(1, 4))
        # Simple BFS to find neighborhood
        visited = {seed}
        frontier = {seed}
        for _ in range(k_hops):
            new_frontier = set()
            for node in frontier:
                # Find neighbors from rate matrix (nonzero off-diagonal)
                # This will be passed as adjacency info
                pass  # implemented via graph structure
            frontier = new_frontier - visited
            visited |= frontier
        mu = np.ones(N) * 0.01
        for node in visited:
            mu[node] = rng.uniform(0.1, 1.0)
    
    elif ic_type == 'near_uniform':
        mu = np.ones(N) + rng.normal(0, 0.1, size=N)
        mu = np.clip(mu, 0.01, None)
    
    elif ic_type == 'power_law':
        mu = rng.pareto(a=1.5, size=N) + 0.01
    
    mu = np.clip(mu, 1e-6, None)
    mu /= mu.sum()
    return mu.astype(np.float32)
```

## Diffusion Time Ranges

### Training
τ ∈ [0.3, 1.5] — moderate diffusion. The observation is neither
trivially close to the source nor completely uniform.

### Testing
- **In-range:** τ ∈ [0.3, 1.5] (same as training)
- **Short τ (extrapolation):** τ ∈ [0.1, 0.3) — less diffusion than
  seen in training. Source is more visible in observation.
- **Long τ (extrapolation):** τ ∈ (1.5, 3.0] — more diffusion than
  seen in training. Source is heavily washed out.

The model receives τ as conditioning (via FiLM) and must adapt its
reconstruction behavior accordingly.

## Observation Model

```python
def generate_observation(mu_source, R, tau):
    """Diffuse source on graph to produce observation."""
    from scipy.linalg import expm
    obs = mu_source @ expm(tau * R)
    obs = np.clip(obs, 0, None)
    obs /= obs.sum() + 1e-15
    return obs
```

## Difficulty Measure

```python
def compute_difficulty(mu_source, obs):
    """
    Difficulty = TV(source, observation).
    
    Low TV: observation ≈ source, easy recovery.
    High TV: observation far from source, hard recovery.
    """
    return 0.5 * np.abs(mu_source - obs).sum()
```

This is computed for every test case and used to bin results.
Bins: easy (TV < 0.2), medium (0.2 ≤ TV < 0.5), hard (TV ≥ 0.5).

## Dataset

```python
class DiffusionSourceDataset(torch.utils.data.Dataset):
    """
    Training data for diffusion source recovery across topologies.
    
    For each sample:
    1. Pick a random graph
    2. Generate random source distribution
    3. Pick random τ from training range
    4. Diffuse source to get observation
    5. OT coupling: observation → source (inverse direction!)
       Actually: uniform/Dirichlet → source, conditioned on observation
    6. Sample flow matching tuples
    
    Context:
        node_context: (N, 2) [obs(a), is_boundary(a)]
            Observation value and boundary flag per node.
        global_cond: (1,) [τ]
            Diffusion time — tells model how much to "undo."
    
    Uses FiLM conditioning for τ.
    Returns 7-tuple for train_film_conditional.
    """
    def __init__(self, graphs, n_pairs_per_graph=20,
                 tau_range=(0.3, 1.5), mode='dirichlet',
                 dirichlet_alpha=1.0, n_starts_per_pair=5,
                 n_samples=20000, seed=42):
        rng = np.random.default_rng(seed)
        all_items = []
        self.all_pairs = []
        
        for name, R, pos in graphs:
            N = R.shape[0]
            
            graph_struct = GraphStructure(R)
            cost = compute_cost_matrix(graph_struct)
            geo_cache = GeodesicCache(graph_struct)
            edge_index = rate_matrix_to_edge_index(R)
            
            for pair_idx in range(n_pairs_per_graph):
                mu_source = generate_source_distribution(N, pos, rng)
                tau = float(rng.uniform(*tau_range))
                
                obs = generate_observation(mu_source, R, tau)
                difficulty = compute_difficulty(mu_source, obs)
                
                # Context
                node_ctx = np.stack([obs, np.zeros(N)], axis=1).astype(np.float32)
                global_ctx = np.array([tau], dtype=np.float32)
                
                self.all_pairs.append({
                    'name': name,
                    'N': N,
                    'R': R,
                    'positions': pos,
                    'edge_index': edge_index,
                    'mu_source': mu_source,
                    'obs': obs,
                    'tau': tau,
                    'difficulty': difficulty,
                })
                
                # OT couplings with Dirichlet starts
                n_starts = n_starts_per_pair if mode == 'dirichlet' else 1
                for _ in range(n_starts):
                    if mode == 'dirichlet':
                        mu_start = rng.dirichlet(
                            np.full(N, dirichlet_alpha)).astype(np.float32)
                    else:
                        mu_start = (np.ones(N) / N).astype(np.float32)
                    
                    coupling = compute_ot_coupling(mu_start, mu_source, cost)
                    geo_cache.precompute_for_coupling(coupling)
                    
                    n_per = max(1, n_samples // (
                        len(graphs) * n_pairs_per_graph * n_starts))
                    
                    for _ in range(n_per):
                        flow_tau = float(rng.uniform(0.0, 0.999))
                        mu_tau = marginal_distribution_fast(
                            geo_cache, coupling, flow_tau)
                        R_target = marginal_rate_matrix_fast(
                            geo_cache, coupling, flow_tau)
                        u_tilde = R_target * (1.0 - flow_tau)
                        
                        all_items.append((
                            torch.tensor(mu_tau, dtype=torch.float32),
                            torch.tensor([flow_tau], dtype=torch.float32),
                            torch.tensor(node_ctx, dtype=torch.float32),
                            torch.tensor(global_ctx, dtype=torch.float32),
                            torch.tensor(u_tilde, dtype=torch.float32),
                            edge_index, N,
                        ))
        
        idx = rng.permutation(len(all_items))
        self.samples = [all_items[i] for i in idx[:n_samples]]
```

## Model

```python
model = FiLMConditionalGNNRateMatrixPredictor(
    node_context_dim=2,   # [obs(a), is_boundary(a)]
    global_dim=1,         # [τ]
    hidden_dim=64,
    n_layers=4)
```

FiLM is needed here: τ determines how much diffusion to undo, and
this is a global property that affects all nodes uniformly. The model
must learn: "at τ=0.3, sharpen slightly; at τ=2.0, sharpen aggressively."

## Baselines

### 1. Direct GNN (no flow matching)
Predicts μ_source directly from (obs, τ, graph).
Context = [obs(a), τ_broadcast] per node.

### 2. Backprojection
μ_hat = obs (just return the observation unchanged).
This is the trivial baseline — works well at small τ, fails at large τ.

### 3. Laplacian sharpening
Inverse diffusion: μ_hat = obs @ expm(-τR_approx) where R_approx is
estimated from the graph. Unstable for large τ (inverse diffusion
amplifies noise) but principled.

### 4. Posterior mean (our model, K=20 Dirichlet starts)
Run K=20 independent flows from Dirichlet starts, average the results.

## Test Cases

Generate test cases for each condition:

```python
def generate_test_cases(graphs, tau_values, n_per_graph=10, seed=None):
    """
    Generate test cases at specific tau values.
    
    Returns list of dicts with mu_source, obs, tau, difficulty, etc.
    """
    rng = np.random.default_rng(seed)
    cases = []
    for name, R, pos in graphs:
        N = R.shape[0]
        edge_index = rate_matrix_to_edge_index(R)
        for tau in tau_values:
            for _ in range(n_per_graph):
                mu_source = generate_source_distribution(N, pos, rng)
                obs = generate_observation(mu_source, R, tau)
                difficulty = compute_difficulty(mu_source, obs)
                cases.append({
                    'name': name, 'N': N, 'R': R,
                    'positions': pos, 'edge_index': edge_index,
                    'mu_source': mu_source, 'obs': obs,
                    'tau': tau, 'difficulty': difficulty,
                })
    return cases

# Test tau values
TAU_IN_RANGE   = [0.3, 0.5, 0.8, 1.0, 1.5]        # in training range
TAU_SHORT      = [0.1, 0.15, 0.2]                    # extrapolation: less diffusion
TAU_LONG       = [2.0, 2.5, 3.0]                     # extrapolation: more diffusion
```

## Evaluation

### Test conditions (3 × 3 = 9 cells)

Topology axis:
- ID: training graphs, new pairs
- OOD-topo: unseen topologies, similar sizes
- OOD-size: unseen sizes (small + large)

Tau axis:
- In-range: τ ∈ [0.3, 1.5]
- Short (extrapolation): τ ∈ [0.1, 0.3)
- Long (extrapolation): τ ∈ (1.5, 3.0]

### Metrics

- **Full TV** between reconstruction and true source
- **Peak recovery:** does the argmax of reconstruction match argmax of source?
- **Calibration** (posterior mode): r between posterior std and |error|

### Grouping by difficulty

Instead of grouping by number of peaks, group by
difficulty = TV(source, observation):
- Easy: difficulty < 0.2
- Medium: 0.2 ≤ difficulty < 0.5
- Hard: difficulty ≥ 0.5

## Data Visualization

### Plot 1: `ex18_graph_zoo.png`
Same as Ex17 — show all training and test graphs.

### Plot 2: `ex18_difficulty_spectrum.png`
Histogram of difficulty values across training data.
Show examples at different difficulty levels:
- Easy (low TV): source and observation look similar
- Medium: source structure partially visible in observation
- Hard (high TV): observation is nearly uniform

For each difficulty level, show 2 examples: (graph, source, observation)
side by side.

### Plot 3: `ex18_tau_effect.png`
For one graph, fix a source distribution and show the observation at
τ = 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0. Shows how diffusion
progressively washes out the source. Annotate each with difficulty value.

## Results Plots

### Plot 4: `ex18_results.png` (2×3 grid)

- **Panel A:** Training loss curve

- **Panel B:** TV vs difficulty scatter plot.
  X = difficulty (TV between source and observation).
  Y = reconstruction TV.
  One cloud per method (FM posterior mean, DirectGNN, backprojection).
  Perfect recovery = points on y=0. Backprojection = points on y=x
  diagonal (returning the observation). Our model should be well below
  the diagonal.

- **Panel C:** TV by τ range, grouped bars.
  Three groups (short τ, in-range τ, long τ).
  Bars for each method. Shows τ generalization.

- **Panel D:** TV by topology condition, grouped bars.
  Three groups (ID, OOD-topo, OOD-size).
  Bars for each method. Shows topology/size generalization.

- **Panel E:** TV vs graph size scatter.
  X = N (number of nodes). Y = reconstruction TV.
  Shows scaling behavior from N=8 to N=150.

- **Panel F:** Calibration plot (posterior mode).
  Scatter of posterior std vs |error|, report r.

### Plot 5: `ex18_reconstruction_gallery.png`

For 6 test cases spanning different topologies and difficulties:
- Row per test case
- Columns: graph structure, source (true), observation (diffused),
  reconstruction (our model), reconstruction (DirectGNN baseline)
- Annotate with difficulty, τ, TV for each method

### Plot 6: `ex18_tau_generalization.png`

For one OOD-topo graph, fix a source and show reconstructions at
τ = 0.1, 0.5, 1.0, 2.0, 3.0 (spanning the full test range including
extrapolation). Row 1: observation. Row 2: our reconstruction.
Row 3: DirectGNN. Row 4: true source (repeated for reference).

## Console Output

```
=== Experiment 18: Diffusion Source Recovery ===
Training: 30 graphs (N=15-60), τ ∈ [0.3, 1.5]
Testing: ID + OOD-topo + OOD-size, τ ∈ [0.1, 3.0]

                 ID          OOD-topo    OOD-size
  τ in-range:
    FM post:   X.XXXX       X.XXXX       X.XXXX
    DirectGNN: X.XXXX       X.XXXX       X.XXXX
    Backproj:  X.XXXX       X.XXXX       X.XXXX

  τ short (extrapolation):
    FM post:   X.XXXX       X.XXXX       X.XXXX
    DirectGNN: X.XXXX       X.XXXX       X.XXXX

  τ long (extrapolation):
    FM post:   X.XXXX       X.XXXX       X.XXXX
    DirectGNN: X.XXXX       X.XXXX       X.XXXX

By difficulty:
              Easy(<0.2)   Medium(0.2-0.5)  Hard(>0.5)
  FM post:   X.XXXX       X.XXXX           X.XXXX
  DirectGNN: X.XXXX       X.XXXX           X.XXXX
  Backproj:  X.XXXX       X.XXXX           X.XXXX

Calibration: r = X.XX
Peak recovery: FM XX%, DirectGNN XX%, Backproj XX%
```

## Expected Outcome

**In-range τ, ID topology:** Our model should excel — this is exactly
what it's trained on.

**OOD topology:** Small generalization gap expected. The GNN learns
local diffusion-inversion rules that transfer across topologies.

**OOD size:** Moderate gap for very large graphs (N=150) due to
deeper message passing needed. Small graphs (N=8) should be easy.

**Short τ (extrapolation):** Both methods should do well — less
diffusion means the observation is close to the source (low difficulty).

**Long τ (extrapolation):** This is the hard test. At τ=3.0, the
observation may be nearly uniform. Our posterior sampling should
provide meaningful uncertainty (high diversity at high τ), while
the DirectGNN gives a single (possibly poor) answer.

**Difficulty analysis:** The scatter plot (Panel B) should show our
model consistently below the y=x diagonal (better than backprojection)
with the gap largest at medium difficulty. At extreme difficulty
(observation ≈ uniform), no method can recover the source.

## CLI

```python
parser.add_argument('--tau-train-range', type=float, nargs=2, default=[0.3, 1.5])
parser.add_argument('--n-pairs-per-graph', type=int, default=20)
parser.add_argument('--n-starts-per-pair', type=int, default=5)
parser.add_argument('--n-samples', type=int, default=20000)
parser.add_argument('--n-epochs', type=int, default=1000)
parser.add_argument('--hidden-dim', type=int, default=64)
parser.add_argument('--n-layers', type=int, default=4)
parser.add_argument('--mode', type=str, default='posterior',
                    choices=['point_estimate', 'posterior'])
parser.add_argument('--posterior-k', type=int, default=20)
parser.add_argument('--loss-type', type=str, default='rate_kl')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint-every', type=int, default=50)
```
