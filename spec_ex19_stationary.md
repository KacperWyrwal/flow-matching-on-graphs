# Experiment 19: Stationary Distribution Prediction

## Motivation

A clean forward graph-level task: given a graph with asymmetric edge
weights, predict its stationary distribution. No target distribution
provided — the model must infer the steady state purely from graph
structure and edge weights.

For symmetric (reversible) graphs, π(a) ∝ degree(a) — trivial.
For asymmetric graphs, π solves πR = 0 and depends on the global
cycle structure. This is genuinely hard for local message passing.

The flow matching framing: transport uniform → π. The intermediate
flow shows how mass redistributes toward steady state, revealing
the graph's mixing structure (bottlenecks, communities, cycles).

## Task

**Input:** Graph with asymmetric edge weights (encoded as edge features).
**Output:** Stationary distribution π where πR = 0.

The model starts from uniform μ₀ = 1/N and flows to π. At test time
on an unseen graph, it integrates the learned ODE to produce π̂.

## Asymmetric Graph Construction

Start from a base graph (from the Ex17 zoo), then assign asymmetric
weights to create a non-reversible Markov chain.

```python
def make_asymmetric_rate_matrix(R_base, asymmetry=1.0, rng=None):
    """
    Given a symmetric base rate matrix, add asymmetry.
    
    For each edge (a,b), the forward and backward rates are:
        R_ab = R_base_ab * exp(+w_ab)
        R_ba = R_base_ab * exp(-w_ab)
    where w_ab ~ Uniform(-asymmetry, +asymmetry).
    
    This preserves the graph topology but makes the chain
    non-reversible. Higher asymmetry = more directional flow
    = stationary distribution further from degree-proportional.
    
    Args:
        R_base: (N, N) symmetric rate matrix
        asymmetry: controls strength of asymmetry
        rng: random generator
    
    Returns:
        R_asym: (N, N) asymmetric rate matrix (rows sum to zero)
        edge_weights: dict mapping (a,b) -> weight for edge features
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N = R_base.shape[0]
    R_asym = np.zeros((N, N))
    edge_weights = {}
    
    # Find edges (upper triangle of symmetric R)
    for a in range(N):
        for b in range(a+1, N):
            if R_base[a, b] > 0:
                w = rng.uniform(-asymmetry, asymmetry)
                R_asym[a, b] = R_base[a, b] * np.exp(w)
                R_asym[b, a] = R_base[a, b] * np.exp(-w)
                edge_weights[(a, b)] = w
                edge_weights[(b, a)] = -w
    
    # Set diagonal
    np.fill_diagonal(R_asym, 0)
    np.fill_diagonal(R_asym, -R_asym.sum(axis=1))
    
    return R_asym, edge_weights


def compute_stationary(R):
    """
    Compute stationary distribution π where πR = 0.
    
    Method: π is the left eigenvector of R corresponding to
    eigenvalue 0, normalized to sum to 1.
    """
    N = R.shape[0]
    # Solve π R = 0 subject to sum(π) = 1
    # Equivalent: R^T π^T = 0
    # Replace last equation with normalization
    A = R.T.copy()
    A[-1, :] = 1.0
    b = np.zeros(N)
    b[-1] = 1.0
    
    pi = np.linalg.solve(A, b)
    pi = np.clip(pi, 0, None)
    pi /= pi.sum()
    return pi.astype(np.float32)
```

## Edge Features

The GNN needs to see the asymmetric edge weights. Each directed edge
(a, b) gets a feature encoding its weight relative to the reverse edge:

```python
def build_edge_features(R, edge_index):
    """
    Build per-edge features encoding asymmetry.
    
    For each directed edge (a, b) in edge_index:
        feature = [R_ab, R_ba, log(R_ab / R_ba)]
    
    The log-ratio is the key asymmetry signal: positive means
    flow prefers a→b over b→a.
    
    Returns: (E, 3) tensor of edge features
    """
    src, dst = edge_index[0], edge_index[1]
    E = len(src)
    feats = np.zeros((E, 3), dtype=np.float32)
    
    for e in range(E):
        a, b = int(src[e]), int(dst[e])
        r_ab = R[a, b]
        r_ba = R[b, a]
        feats[e, 0] = r_ab
        feats[e, 1] = r_ba
        feats[e, 2] = np.log(r_ab / (r_ba + 1e-8) + 1e-8)
    
    return feats
```

Note: the current GNN architecture takes node features and edge_index
but not edge features. We may need to extend the message passing to
incorporate edge features, or encode them as additional node context
(e.g., per-node summary of incoming/outgoing rate asymmetry).

Alternative (simpler): encode asymmetry as per-node features:

```python
def build_node_asymmetry_features(R):
    """
    Per-node summary of directional flow.
    
    For each node a:
        in_strength  = sum_b R_ba  (total incoming rate)
        out_strength = sum_b R_ab  (total outgoing rate)
        net_flow     = in_strength - out_strength
        asymmetry    = log(in_strength / out_strength)
    
    Nodes with high in_strength relative to out_strength are
    "attractors" — the stationary distribution should be high there.
    """
    N = R.shape[0]
    R_off = R.copy()
    np.fill_diagonal(R_off, 0)
    
    in_strength = R_off.sum(axis=0)     # sum of column = incoming
    out_strength = R_off.sum(axis=1)    # sum of row = outgoing
    net_flow = in_strength - out_strength
    log_ratio = np.log((in_strength + 1e-8) / (out_strength + 1e-8))
    
    features = np.stack([in_strength, out_strength, net_flow, log_ratio],
                         axis=1).astype(np.float32)
    return features  # (N, 4)
```

## Graph Zoo

Same as Ex17 but with asymmetric weights applied to each graph.
Multiple asymmetry levels per graph for variety.

### Training
~30 base graphs × 3 asymmetry levels (0.5, 1.0, 2.0) = ~90 instances.

### Test: unseen topologies
~10 base graphs × 3 asymmetry levels = ~30 instances.

### Test: unseen sizes
~12 base graphs × 3 asymmetry levels = ~36 instances.

### Test: unseen asymmetry levels
Training graphs at asymmetry = 0.25 (weaker) and 3.0 (stronger).
Tests whether the model generalizes to different degrees of
non-reversibility.

## Dataset

```python
class StationaryDistDataset(torch.utils.data.Dataset):
    """
    Training data for stationary distribution prediction.
    
    For each graph instance:
    1. Compute stationary distribution π
    2. OT coupling: uniform (1/N) → π
    3. Sample flow matching tuples
    
    Context:
        node_context: (N, 4) [in_strength, out_strength, net_flow, log_ratio]
            Per-node asymmetry features derived from rate matrix.
    
    No global conditioning (no FiLM). The model must infer everything
    from the per-node asymmetry features and graph topology.
    
    Returns 6-tuple for train_flexible_conditional:
        (mu_tau, tau, node_context, R_target, edge_index, N)
    """
    def __init__(self, graph_instances, mode='dirichlet',
                 dirichlet_alpha=1.0, n_starts_per_graph=5,
                 n_samples=20000, seed=42):
        """
        graph_instances: list of (name, R_asym, positions) tuples
        mode: 'uniform' (single start) or 'dirichlet' (multiple starts
              for epistemic uncertainty quantification)
        """
        rng = np.random.default_rng(seed)
        all_items = []
        self.all_pairs = []
        
        for name, R, pos in graph_instances:
            N = R.shape[0]
            
            # Compute stationary distribution
            pi = compute_stationary(R)
            
            # Node context: asymmetry features
            node_ctx = build_node_asymmetry_features(R)  # (N, 4)
            
            # OT structures
            graph_struct = GraphStructure(R)
            cost = compute_cost_matrix(graph_struct)
            geo_cache = GeodesicCache(graph_struct)
            edge_index = rate_matrix_to_edge_index(R)
            
            self.all_pairs.append({
                'name': name,
                'N': N,
                'R': R,
                'positions': pos,
                'edge_index': edge_index,
                'pi': pi,
                'node_ctx': node_ctx,
            })
            
            # Multiple starts for epistemic uncertainty
            n_starts = n_starts_per_graph if mode == 'dirichlet' else 1
            for _ in range(n_starts):
                if mode == 'dirichlet':
                    mu_start = rng.dirichlet(
                        np.full(N, dirichlet_alpha)).astype(np.float32)
                else:
                    mu_start = (np.ones(N) / N).astype(np.float32)
                
                coupling = compute_ot_coupling(mu_start, pi, cost)
                geo_cache.precompute_for_coupling(coupling)
                
                n_per = max(1, n_samples // (
                    len(graph_instances) * n_starts))
                
                for _ in range(n_per):
                    tau = float(rng.uniform(0.0, 0.999))
                    mu_tau = marginal_distribution_fast(
                        geo_cache, coupling, tau)
                    R_target = marginal_rate_matrix_fast(
                        geo_cache, coupling, tau)
                    u_tilde = R_target * (1.0 - tau)
                    
                    all_items.append((
                        torch.tensor(mu_tau, dtype=torch.float32),
                        torch.tensor([tau], dtype=torch.float32),
                        torch.tensor(node_ctx, dtype=torch.float32),
                        torch.tensor(u_tilde, dtype=torch.float32),
                        edge_index, N,
                    ))
        
        idx = rng.permutation(len(all_items))
        self.samples = [all_items[i] for i in idx[:n_samples]]
```

## Model

```python
model = FlexibleConditionalGNNRateMatrixPredictor(
    context_dim=4,        # [in_strength, out_strength, net_flow, log_ratio]
    hidden_dim=64,
    n_layers=6)           # more layers for global information propagation
```

No FiLM — the model infers the stationary distribution entirely from
per-node asymmetry features and graph structure. 6 layers instead of 4
because predicting π requires global information (cycle structure),
which needs more message passing rounds.

## Baselines

### 1. Degree-proportional (reversible approximation)
π̂(a) ∝ in_strength(a). This is exact for reversible chains and
a reasonable approximation for mildly asymmetric chains. Fails as
asymmetry increases.

```python
def baseline_degree_proportional(R):
    R_off = R.copy()
    np.fill_diagonal(R_off, 0)
    in_str = R_off.sum(axis=0)
    pi = in_str / in_str.sum()
    return pi
```

### 2. Power iteration
Run π ← π @ expm(dt * R) from uniform for many steps until
convergence. This is exact given enough iterations but requires
access to R at test time (which the model doesn't use directly).
Serves as an upper bound.

```python
def baseline_power_iteration(R, n_steps=1000, dt=0.01):
    N = R.shape[0]
    pi = np.ones(N) / N
    P = expm(dt * R)
    for _ in range(n_steps):
        pi = pi @ P
        pi = np.clip(pi, 0, None)
        pi /= pi.sum()
    return pi
```

### 3. Direct GNN
Predicts π directly from node features in one forward pass.
Context = [in_strength, out_strength, net_flow, log_ratio] per node.

### 4. Exact solution (ground truth)
Solve πR = 0 directly. This is the reference.

## Evaluation

### Test conditions

Four axes:
1. **ID:** Training graphs, training asymmetry levels
2. **OOD-topo:** Unseen graph types, same sizes
3. **OOD-size:** Unseen sizes (smaller N=8-12, larger N=80-150)
4. **OOD-asymmetry:** Training graphs at unseen asymmetry (0.25, 3.0)

### Metrics

- **TV** between predicted π̂ and true π
  - For FM: posterior mean from K=20 Dirichlet starts
  - For baselines: single prediction
- **KL divergence** KL(π ‖ π̂)
- **Rank correlation:** Spearman ρ between π̂ and π (does the model
  get the ordering right, even if magnitudes are off?)
- **Top-k accuracy:** Are the k nodes with highest π̂ among the
  k nodes with highest true π?
- **Calibration:** Pearson r between posterior std per node and
  |posterior mean - true π| per node. High r means the model knows
  where it's uncertain. This is epistemic uncertainty — the model
  is uncertain about its prediction, not about the ground truth.
- **Diversity:** Mean pairwise TV between the K posterior samples.
  High diversity on OOD graphs (model is uncertain) and low diversity
  on ID graphs (model is confident) is the expected pattern.

## Data Visualization

### Plot 1: `ex19_asymmetry_examples.png`

For 4 graphs, show:
- Column 1: Graph with edge thickness proportional to rate (arrows
  showing direction for asymmetric edges)
- Column 2: Degree-proportional estimate (color on nodes)
- Column 3: True stationary distribution (color on nodes)
- Column 4: Difference (where degree-proportional fails)

Shows that asymmetry makes π non-trivial.

### Plot 2: `ex19_asymmetry_spectrum.png`

For one graph (e.g., cycle_20), show π at asymmetry = 0.0, 0.5, 1.0,
2.0, 3.0. At asymmetry 0, π is uniform (cycle is symmetric). As
asymmetry increases, π concentrates on "attractor" nodes. Shows
how the problem difficulty increases with asymmetry.

### Plot 3: `ex19_flow_to_stationary.png`

For one graph, show the flow from uniform to π at
t = 0, 0.25, 0.5, 0.75, 1.0. Mass gradually concentrates at
high-π nodes. On a graph with a bottleneck, mass accumulates on
the "receiving" side of the bottleneck.

## Results Plots

### Plot 4: `ex19_results.png` (2×3 grid)

- **Panel A:** Training loss curve

- **Panel B:** TV comparison bars grouped by test condition
  (ID, OOD-topo, OOD-size, OOD-asymmetry). Bars for FM, DirectGNN,
  degree-proportional.

- **Panel C:** TV vs asymmetry level. X = asymmetry (0.25 to 3.0).
  Lines for each method. The degree-proportional baseline should
  degrade with increasing asymmetry. Our model should maintain
  performance.

- **Panel D:** Prediction gallery. For 3 OOD test graphs: true π,
  predicted π̂ (our model), predicted (DirectGNN), degree-proportional.
  Each as colored nodes on the graph.

- **Panel E:** TV vs graph size (N=8 to N=150).

- **Panel F:** Calibration plot. Scatter of posterior std vs |error|
  per node, across all test cases. Report Pearson r. Color by test
  condition (ID vs OOD). OOD points should have higher std AND higher
  error — the model should be more uncertain on unfamiliar graphs.
  
  Additionally: box plot of posterior diversity (mean pairwise TV)
  grouped by ID / OOD-topo / OOD-size / OOD-asymmetry. Diversity
  should increase for OOD conditions (epistemic uncertainty).

## Console Output

```
=== Experiment 19: Stationary Distribution Prediction ===
Training: 90 graph instances (30 base × 3 asymmetry levels)
Testing: ID, OOD-topo, OOD-size, OOD-asymmetry

TV Results:
                    ID       OOD-topo  OOD-size  OOD-asym
  FM (Learned):   X.XXXX    X.XXXX    X.XXXX    X.XXXX
  DirectGNN:      X.XXXX    X.XXXX    X.XXXX    X.XXXX
  Degree-prop:    X.XXXX    X.XXXX    X.XXXX    X.XXXX

TV by asymmetry:
              0.25     0.5      1.0      2.0      3.0
  FM:       X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
  DirectGNN:X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX
  Deg-prop: X.XXXX   X.XXXX   X.XXXX   X.XXXX   X.XXXX

Rank correlation (Spearman ρ):
  FM: X.XX, DirectGNN: X.XX, Degree-prop: X.XX
```

## Expected Outcome

**Low asymmetry (0.25-0.5):** All methods do well — π is close to
degree-proportional. Our model should match DirectGNN.

**Medium asymmetry (1.0-2.0):** Degree-proportional degrades. FM and
DirectGNN should remain strong if they've learned the global balance.

**High asymmetry (3.0, extrapolation):** Hardest test. π concentrates
strongly on attractor nodes. The model must extrapolate beyond
training asymmetry levels. Degree-proportional fails badly.

**Size generalization:** The key challenge for large graphs — the
stationary distribution depends on global cycle structure, which
requires information to propagate across the entire graph. More
GNN layers help but may not be sufficient for N=150. This is an
honest limitation to report.

**The flow visualization** is unique to our approach: showing HOW
mass redistributes from uniform toward the stationary distribution
reveals the graph's mixing dynamics — which nodes fill first, where
bottlenecks slow the flow. DirectGNN gives only the answer, not
the process.

## CLI

```python
parser.add_argument('--asymmetry-levels', type=float, nargs='+',
                    default=[0.5, 1.0, 2.0])
parser.add_argument('--test-asymmetry', type=float, nargs='+',
                    default=[0.25, 3.0])
parser.add_argument('--n-samples', type=int, default=20000)
parser.add_argument('--n-epochs', type=int, default=1000)
parser.add_argument('--hidden-dim', type=int, default=64)
parser.add_argument('--n-layers', type=int, default=6)
parser.add_argument('--loss-type', type=str, default='rate_kl')
parser.add_argument('--mode', type=str, default='dirichlet',
                    choices=['uniform', 'dirichlet'])
parser.add_argument('--dirichlet-alpha', type=float, default=1.0)
parser.add_argument('--n-starts-per-graph', type=int, default=5,
                    help='Dirichlet starts per graph during training')
parser.add_argument('--posterior-k', type=int, default=20,
                    help='Number of posterior samples at test time')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint-every', type=int, default=50)
```

## Note on Edge Features

The current GNN architecture propagates information via node features
along edges but doesn't use explicit edge features. The asymmetry
information is encoded as per-node summaries (in_strength, out_strength,
etc.). This loses directionality information — the model knows each
node's total inflow/outflow but not which specific edges are strong.

A future improvement: extend the GNN to use edge features directly
in the message passing (as in MeshGraphNets). This would allow the
model to reason about specific directional flows, potentially
improving performance on highly asymmetric graphs.

For this experiment, per-node asymmetry summaries should suffice for
moderate asymmetry. At high asymmetry, edge features may become
necessary — this is a diagnostic to watch for.
