# Implementation Spec: Optimal Transport Flow Matching on Graphs

## Project Overview

Implement a two-level optimal transport flow matching framework on graphs:
1. **Graph-level**: Exact OT flow between distributions on a finite graph (no learning)
2. **Meta-level**: Learned flow matching on the space of distributions, using graph-level solutions as training signal

The codebase should be clean, minimal, research-quality Python. PyTorch for the meta-level neural network; NumPy/SciPy for the graph-level solver.

---

## Repository Structure

```
graph-ot-fm/
├── README.md
├── requirements.txt
├── graph_ot_fm/
│   ├── __init__.py
│   ├── graph.py              # Graph data structure and geodesic computations
│   ├── ot_solver.py          # OT coupling solver
│   ├── conditional.py        # Conditional rate matrices and marginals
│   ├── flow.py               # Marginal rate matrix assembly (exact solution)
│   └── utils.py              # Helpers (e.g., distribution evolution)
├── meta_fm/
│   ├── __init__.py
│   ├── dataset.py            # Training data generation from graph-level solver
│   ├── model.py              # Neural network for rate matrix prediction
│   ├── train.py              # Training loop
│   └── sample.py             # Sampling / inference at meta-level
├── experiments/
│   ├── ex1_cycle_graph.py    # Graph-level: exact OT flow on cycle graph
│   ├── ex2_grid_graph.py     # Graph-level: exact OT flow on grid graph
│   ├── ex3_meta_level.py     # Meta-level: learned flow on space of distributions
│   └── plotting.py           # Shared visualization utilities
├── notebooks/
│   └── walkthrough.ipynb     # Interactive walkthrough of the full framework
└── tests/
    ├── test_graph.py
    ├── test_conditional.py
    ├── test_flow.py
    └── test_meta.py
```

---

## Dependencies

```
# requirements.txt
numpy>=1.24
scipy>=1.10
torch>=2.0
matplotlib>=3.7
networkx>=3.0            # Graph construction and shortest paths
pot>=0.9                  # Python Optimal Transport library
jupyter                   # For notebook
pytest                    # For tests
```

---

## Module 1: `graph_ot_fm/` — Graph-Level Exact Solver

### `graph.py` — Graph Structure and Geodesics

```python
class GraphStructure:
    """
    Precomputes and caches all geodesic information needed for flow matching.
    
    Constructor args:
        rate_matrix: np.ndarray of shape (N, N)
            Reference rate matrix R. R[i,j] > 0 iff edge i->j exists.
            Diagonal entries are ignored (recomputed as -sum of off-diag row).
            
    Attributes:
        N: int — number of nodes
        R: np.ndarray (N, N) — cleaned rate matrix (diagonal = -sum of off-diag)
        dist: np.ndarray (N, N) — shortest path distances d(i,j)
            Use scipy.sparse.csgraph.shortest_path on the adjacency (hop-count).
            For weighted graphs: edge weight for shortest path = 1 for all edges
            (we use hop distance, not weight-distance). Set dist[i,j] = np.inf
            if j is unreachable from i.
        geodesic_count: np.ndarray (N, N) — N_a values
            geodesic_count[a, j] = (R^{d(a,j)})_{aj}
            Compute via matrix power: for each target j, for each distance level d,
            compute R^d and extract entry [a,j] for all a with dist[a,j] == d.
            Optimization: compute R^1, R^2, ..., R^{dmax} incrementally.
        closer_neighbors: dict
            closer_neighbors[(a, j)] = list of nodes b where:
              R[a,b] > 0 AND dist[b,j] == dist[a,j] - 1
            These are the allowed transitions in the conditional process.
    
    Methods:
        branching_probs(a, j) -> dict[int, float]:
            Returns {b: R[a,b] * N_b / N_a} for b in closer_neighbors[(a,j)]
            These are the geodesic random walk transition probabilities.
            Must sum to 1.0 (verify this as assertion).
    
    Implementation notes:
        - Use networkx or scipy for shortest paths
        - For geodesic_count, IMPORTANT: use only the off-diagonal part of R 
          when computing matrix powers. That is, define R_offdiag = R.copy() with
          diagonal set to 0, then R_offdiag^d gives the weighted path counts.
        - Cache everything at construction time; this object is immutable.
        - Graphs are assumed undirected for now (R symmetric off-diagonal).
```

### `ot_solver.py` — Optimal Transport Coupling

```python
def compute_cost_matrix(graph: GraphStructure) -> np.ndarray:
    """
    Compute the pairwise OT cost c(i,j) for all node pairs.
    
    For nodes i, j with dist[i,j] < inf:
        c(i,j) = E[ sum_{k} -log R_{X_k, X_{k+1}} | geodesic walk i -> j ]
    
    Implementation:
        Use dynamic programming backward from j.
        Define V(a) = expected cost-to-go from a to j along geodesic walk.
        Base case: V(j) = 0.
        Recursion: V(a) = sum_{b in N^-(a)} P_geo(a->b|j) * (-log R[a,b] + V(b))
        where P_geo(a->b|j) = R[a,b] * N_b / N_a (from graph.branching_probs).
        Then c(i,j) = V(i).
    
    For unreachable pairs: c(i,j) = inf.
    
    For unweighted graphs where all R[i,j] in {0,1}: all -log terms are 0,
    so c(i,j) = 0 for all reachable pairs. In this case, FALL BACK to
    c(i,j) = d(i,j)^2 (squared hop distance) as the cost.
    
    Returns: np.ndarray (N, N) cost matrix.
    """

def compute_ot_coupling(
    mu0: np.ndarray,  # shape (N,), source distribution
    mu1: np.ndarray,  # shape (N,), target distribution
    cost: np.ndarray,  # shape (N, N), cost matrix
) -> np.ndarray:
    """
    Solve the discrete OT problem for coupling pi(i,j).
    
    Use the POT library: ot.emd(mu0, mu1, cost)
    
    Returns: np.ndarray (N, N) optimal coupling matrix.
    """
```

### `conditional.py` — Conditional Paths and Rate Matrices

```python
def conditional_marginal(
    graph: GraphStructure,
    i: int,           # source node
    j: int,           # target node
    t: float,         # time in [0, 1)
) -> np.ndarray:
    """
    Compute p_t(x | i, j) for all nodes x.
    
    p_t(x | i,j) = sum_{k=0}^{d} Binom(k; d, t) * w_k(x | i, j)
    
    where d = dist[i,j].
    
    Computing w_k(x | i, j) — the spatial weights:
        w_0(x) = delta(x, i)
        w_{k+1}(x) = sum_{a: x in N^-(a)} w_k(a) * P_geo(a -> x | j)
        
        In other words, start with all mass at i, then repeatedly apply
        the geodesic random walk transition matrix. This is a simple
        matrix-vector product if we construct the transition matrix P_geo
        restricted to geodesic nodes.
    
    Combine with Binom(k; d, t) from scipy.stats.binom.pmf(k, d, t).
    
    Edge case: if i == j, return delta(x, i) for all t.
    Edge case: t == 0, return delta(x, i).
    Edge case: t == 1, return delta(x, j).
    
    Returns: np.ndarray (N,) probability distribution over nodes.
    """

def conditional_rate_matrix(
    graph: GraphStructure,
    i: int,           # source node
    j: int,           # target node
    t: float,         # time in [0, 1)
) -> np.ndarray:
    """
    Compute the conditional rate matrix R_t^{i->j}(a, b) for all a, b.
    
    For each node a on a geodesic from i to j (i.e., d(i,a) + d(a,j) == d(i,j)):
        For each b in N^-(a) (closer to j):
            R_t[a, b] = d(a,j) / (1-t) * R[a,b] * N_b / N_a
        R_t[a, a] = -d(a,j) / (1-t)
    
    All other entries are 0.
    
    IMPORTANT: Only nodes a that lie on SOME geodesic from i to j should
    have nonzero rates. A node a is on a geodesic from i to j iff
    d(i,a) + d(a,j) == d(i,j).
    
    Edge case: if i == j, return zero matrix.
    Edge case: as t -> 1, rates diverge; caller should not evaluate at t=1.
    
    Returns: np.ndarray (N, N) rate matrix.
    """

def sample_conditional_state(
    graph: GraphStructure,
    i: int,
    j: int,
    t: float,
    rng: np.random.Generator,
) -> int:
    """
    Sample x ~ p_t(·|i,j).
    
    Algorithm:
        1. d = dist[i,j]
        2. k ~ Binom(d, t)
        3. Run k steps of geodesic random walk from i toward j
        4. Return final node
    
    This is more efficient than computing the full marginal for sampling.
    """
```

### `flow.py` — Marginal Rate Matrix (Exact Solution)

```python
def marginal_rate_matrix(
    graph: GraphStructure,
    coupling: np.ndarray,   # shape (N, N), OT coupling pi(i,j)
    t: float,               # time in [0, 1)
) -> np.ndarray:
    """
    Compute the exact marginal rate matrix u_t(a, b).
    
    u_t(a, b) = sum_{i,j} pi(i,j) * p_t(a|i,j) / p_t(a) * R_t^{i->j}(a,b)
    
    Implementation:
        1. Compute p_t(a) = sum_{i,j} pi(i,j) * p_t(a|i,j) — the marginal at time t.
        2. For each (i,j) with pi(i,j) > 0:
            a. Compute p_t(·|i,j) via conditional_marginal
            b. Compute R_t^{i->j} via conditional_rate_matrix
            c. Accumulate: u_t += pi(i,j) * outer_weight * R_t^{i->j}
               where outer_weight[a] = p_t(a|i,j) / p_t(a)
        3. Ensure diagonal: u_t[a,a] = -sum_{b!=a} u_t[a,b]
    
    Returns: np.ndarray (N, N) rate matrix.
    """

def marginal_distribution(
    graph: GraphStructure,
    coupling: np.ndarray,   # OT coupling
    t: float,
) -> np.ndarray:
    """
    Compute p_t(a) = sum_{i,j} pi(i,j) * p_t(a|i,j).
    
    Returns: np.ndarray (N,) probability distribution.
    """

def evolve_distribution(
    p0: np.ndarray,         # initial distribution (N,)
    rate_matrix_fn,         # callable: t -> (N, N) rate matrix
    t_span: tuple[float, float],
    n_steps: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Numerically evolve dp/dt = p @ R(t) from t_span[0] to t_span[1].
    
    Use simple Euler method or scipy.integrate.solve_ivp.
    The ODE is: dp/dt = p @ R(t), i.e., row vector @ matrix.
    
    Returns:
        times: np.ndarray (n_steps,)
        distributions: np.ndarray (n_steps, N) — distribution at each time
    """
```

### `utils.py` — Helpers

```python
def make_cycle_graph(n: int, weighted: bool = False) -> np.ndarray:
    """
    Create rate matrix for cycle graph on n nodes.
    Node i connects to (i-1) % n and (i+1) % n.
    If weighted: R[i, (i+1)%n] = random weight > 0.
    If not weighted: all edge rates = 1.
    Returns: (N, N) rate matrix.
    """

def make_grid_graph(rows: int, cols: int, weighted: bool = False) -> np.ndarray:
    """
    Create rate matrix for 2D grid graph (rows x cols).
    Node (r,c) has index r*cols + c.
    Edges to 4-neighbors (up, down, left, right) within bounds.
    Returns: (N, N) rate matrix.
    """

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q), handling zeros gracefully."""

def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """TV distance = 0.5 * sum |p - q|."""
```

---

## Module 2: `meta_fm/` — Meta-Level Learned Flow Matching

### `dataset.py` — Training Data Generation

```python
class MetaFlowMatchingDataset(torch.utils.data.Dataset):
    """
    Generates training data for the meta-level flow matching.
    
    Constructor args:
        graph: GraphStructure
        source_distributions: list of np.ndarray, each shape (N,)
            Samples from Pi_0 (source meta-distribution)
        target_distributions: list of np.ndarray, each shape (N,)
            Samples from Pi_1 (target meta-distribution)
        n_samples: int
            Number of training samples to pre-generate.
        cost_type: str = 'wasserstein'
            How to compute W(mu, nu) for the meta-OT coupling.
            Use the graph-level OT cost: for each pair (mu_i, nu_j),
            compute the OT distance using the graph cost matrix.
    
    Pre-generation procedure:
        1. Compute the graph-level cost matrix c(i,j) once.
        2. For each pair (mu_s, nu_t) of source/target distributions:
            a. Compute the OT coupling pi^{s,t}(i,j) on the graph
            b. Compute the OT distance W(mu_s, nu_t) = sum pi^{s,t}(i,j) * c(i,j)
        3. Form the meta-cost matrix W_meta[s,t] = W(mu_s, nu_t)
        4. Solve the meta-OT coupling: Pi_meta = ot.emd(uniform, uniform, W_meta)
           (or with appropriate meta-source/target weights)
        5. Pre-generate training tuples:
            For each sample:
                a. Draw (s, t) from Pi_meta
                b. Draw time tau ~ U[0, 1]
                c. Compute mu_tau by evolving mu_s under the exact graph-level
                   flow toward nu_t, evaluated at time tau.
                   
                   Specifically: mu_tau = marginal_distribution(graph, pi^{s,t}, tau)
                   where pi^{s,t} is the graph-level OT coupling for the pair (mu_s, nu_t).
                   
                d. Compute the target rate matrix:
                   R_target = marginal_rate_matrix(graph, pi^{s,t}, tau)
                e. Store (mu_tau, tau, R_target)
    
    __getitem__ returns:
        mu: torch.Tensor (N,) — the distribution at time tau
        tau: torch.Tensor (1,) — the time
        R_target: torch.Tensor (N, N) — the target rate matrix
    """
```

### `model.py` — Neural Network

```python
class RateMatrixPredictor(nn.Module):
    """
    Predicts a rate matrix given a distribution and time.
    
    Architecture:
        Input: concatenation of [mu (N dims), t (1 dim)] -> N+1 dims
        Hidden: 3 hidden layers, each 256 units, ReLU activation
        Output: N*N dims, reshaped to (N, N)
        
        Post-processing to ensure valid rate matrix:
            1. Mask diagonal to 0
            2. Apply softplus to off-diagonal entries (ensures non-negative rates)
            3. Set diagonal = -sum of off-diagonal in each row
    
    Constructor args:
        n_nodes: int — number of graph nodes
        hidden_dim: int = 256
        n_layers: int = 3
    
    forward(mu, t):
        mu: (batch, N)
        t: (batch, 1)
        returns: (batch, N, N) valid rate matrices
    """
```

### `train.py` — Training Loop

```python
def train(
    model: RateMatrixPredictor,
    dataset: MetaFlowMatchingDataset,
    n_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> dict:
    """
    Train the rate matrix predictor.
    
    Loss function:
        MSE between predicted and target rate matrices, computed on
        OFF-DIAGONAL entries only (diagonal is determined by off-diagonal).
        
        loss = mean over batch of:
            sum_{a != b} (u_theta(a,b | mu, t) - R_target(a,b))^2
    
    Optimizer: Adam
    
    Logging:
        Track loss per epoch. Print every 50 epochs.
    
    Returns: dict with 'losses' key containing per-epoch loss values.
    """
```

### `sample.py` — Inference

```python
def sample_trajectory(
    model: RateMatrixPredictor,
    mu_start: np.ndarray,     # starting distribution (N,)
    n_steps: int = 200,
    device: str = 'cpu',
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a trajectory by integrating the learned rate matrix field.
    
    Uses Euler method:
        mu_{k+1} = mu_k + dt * mu_k @ R_theta(mu_k, t_k)
        
    After each step, clip to non-negative and renormalize to sum to 1.
    
    Returns:
        times: np.ndarray (n_steps,)
        trajectory: np.ndarray (n_steps, N)
    """
```

---

## Experiments

### Experiment 1: `ex1_cycle_graph.py` — Cycle Graph Exact Flow

**Setup:**
- Cycle graph with N=8 nodes, unweighted (all R[i,j] = 1 for edges)
- Source: peaked at node 0 (e.g., [0.7, 0.1, 0.05, 0.05, 0.02, 0.02, 0.03, 0.03])
- Target: peaked at node 4 (the opposite side of the cycle)
- Use cost c(i,j) = d(i,j)^2

**Procedure:**
1. Construct GraphStructure
2. Compute cost matrix and OT coupling
3. Compute marginal rate matrix u_t at a grid of times t in [0, 0.99]
4. Evolve the source distribution forward using the exact rate matrices
5. Verify: the evolved distribution at t=1 should match the target

**Plots (single figure with subplots):**
- **Panel A**: Bar plots of p_t(x) at t = 0, 0.25, 0.5, 0.75, 0.99 (5 snapshots)
- **Panel B**: Heatmap of p_t(x) over (t, x) — time on x-axis, node on y-axis, color = probability
- **Panel C**: Total variation distance between evolved distribution and exact marginal (should be ~0, validating the numerics)

**Validation checks (print to console):**
- p_t sums to 1 at each time step
- All p_t entries non-negative  
- Final distribution close to mu_1 (TV < 0.01)
- Rate matrices have zero row sums at each time step

### Experiment 2: `ex2_grid_graph.py` — Grid Graph Exact Flow

**Setup:**
- 4x4 grid graph (16 nodes), unweighted
- Source: peaked at top-left corner (node 0)
- Target: peaked at bottom-right corner (node 15)
- These are at maximum graph distance d=6 from each other
- Use cost c(i,j) = d(i,j)^2

**Procedure:** Same as Experiment 1.

**Plots (single figure with subplots):**
- **Panel A**: Grid visualizations of p_t at t = 0, 0.25, 0.5, 0.75, 0.99
  Show the 4x4 grid with node color/size proportional to probability.
  Use matplotlib with circles at grid positions, sized by probability.
- **Panel B**: Heatmap of p_t(x) over time (same as Ex1 but 16 nodes)
- **Panel C**: Verify that mass flows along geodesics (shortest paths from
  corner to corner go through the diagonal of the grid).

**Additional demonstration:**
- Second source/target pair: uniform distribution -> peaked at center node
- Shows the framework handles different transport patterns

### Experiment 3: `ex3_meta_level.py` — Meta-Level Learned Flow

**Setup:**
- Use the cycle graph with N=6 nodes (small for fast training)
- Source meta-distribution Pi_0: 50 distributions, each "peaked" at a single
  random node. Generate by: pick a node, put mass 0.6 there, distribute 0.4
  uniformly among the rest, then add small noise and renormalize.
- Target meta-distribution Pi_1: 50 distributions, each approximately uniform.
  Generate by: start from uniform (1/6, ..., 1/6), add Gaussian noise with
  std=0.05, clip, renormalize.
- The meta-level task: learn to transport peaked distributions to near-uniform ones.

**Procedure:**
1. Generate source and target distribution sets
2. Compute meta-cost matrix (W(mu_s, nu_t) for all pairs)
3. Solve meta-OT coupling
4. Pre-generate training dataset (5000 samples)
5. Train RateMatrixPredictor for 500 epochs
6. Evaluate: take a held-out peaked distribution, run the learned flow,
   check if it reaches a near-uniform distribution

**Plots (single figure with subplots):**
- **Panel A**: Training loss curve
- **Panel B**: Trajectories for 3 test distributions — show bar plots at
  t = 0, 0.5, 1.0 for each (3x3 grid of bar plots)
- **Panel C**: Compare learned trajectory vs exact trajectory for one test pair.
  Plot TV distance between learned and exact at each time step.

---

## Notebook: `walkthrough.ipynb`

### Structure

**Cell 1: Introduction**
Markdown explaining the framework at a high level. Reference the PDF summary.

**Cell 2: Graph setup**
Build a cycle graph with N=6. Visualize with networkx.
Show the rate matrix R, distance matrix d(i,j), geodesic counts N_a.

**Cell 3: Conditional components**
Pick a source-target pair (i=0, j=3 on cycle of 6).
Show the conditional marginal p_t(x|0,3) at several times — demonstrate the
binomial * geodesic walk factorization visually.
Show the conditional rate matrix at one time value.

**Cell 4: Cost and OT coupling**
Compute and display the cost matrix. Solve OT for two distributions.
Visualize the coupling matrix as a heatmap.

**Cell 5: Exact flow**
Compute and evolve the marginal rate matrix. Animate or show snapshots.
Verify convergence to target distribution.

**Cell 6: Meta-level setup**
Generate source and target distribution sets. Show a few examples.
Compute meta-cost matrix and meta-OT coupling.

**Cell 7: Meta-level training**
Train the network. Show loss curve.

**Cell 8: Meta-level evaluation**
Run the learned flow on test distributions. Compare to exact.

---

## Tests

### `test_graph.py`
- Test that distance matrix is correct for cycle and grid graphs
- Test that geodesic_count satisfies the recursion N_a = sum R[a,b]*N_b
- Test that branching probabilities sum to 1
- Test symmetry: dist[i,j] == dist[j,i] for undirected graphs

### `test_conditional.py`
- Test that conditional_marginal sums to 1
- Test that conditional_marginal at t=0 is delta_i, at t->1 is delta_j
- Test the time derivative: dp/dt should equal p @ R_t (finite difference check)
- Test that conditional_rate_matrix has zero row sums
- Test that only geodesic nodes have nonzero rates
- Test d(i,j)=1 special case: rate = 1/(1-t)
- Test d(i,j)=2 case: check binomial marginal with known geodesics

### `test_flow.py`
- Test that marginal_distribution sums to 1
- Test that marginal_rate_matrix has zero row sums
- Test that evolving mu_0 under u_t reaches mu_1 (TV < 0.01)
- Test with trivial coupling (i=j for all): flow should be zero

### `test_meta.py`
- Test that MetaFlowMatchingDataset generates valid data (distributions sum to 1,
  rate matrices have zero row sums)
- Test that RateMatrixPredictor output is a valid rate matrix
- Test that training loss decreases over epochs (on a tiny problem)

---

## Key Implementation Notes

1. **Numerical stability near t=1**: The factor 1/(1-t) diverges. In practice,
   clamp t to [0, 0.999] in all computations. This is standard in flow matching.

2. **Geodesic count computation**: When computing R^d for the geodesic counts,
   use the off-diagonal part of R only (zero out diagonal). This ensures we count
   paths of exactly d distinct edges.

3. **Sparse coupling**: The OT coupling pi(i,j) is typically very sparse for
   discrete distributions. Iterate only over nonzero entries for efficiency.

4. **Rate matrix validity**: After every computation, assert that off-diagonal
   entries are non-negative and each row sums to 0 (up to numerical tolerance).

5. **The cost fallback**: For unweighted graphs, -log(R[i,j]) = 0 for all edges,
   making the cost degenerate. Use c(i,j) = d(i,j)^2 as the fallback cost.

6. **Meta-level distribution evolution**: When computing mu_t for training data,
   use the closed-form marginal_distribution (which is a finite sum over the
   coupling entries), NOT numerical ODE integration. The closed form is exact.

7. **Reproducibility**: Use seeded random number generators throughout. Default
   seed = 42.
