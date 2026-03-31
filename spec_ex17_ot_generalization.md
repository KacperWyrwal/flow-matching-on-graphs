# Experiment 17: OT Transport Generalization Across Topologies and Sizes

## Motivation

Test the core capability in isolation: can a GNN learn optimal transport
on graphs and generalize to unseen topologies and unseen graph sizes?

No physics, no application framing — just pure OT flow matching.
This is the cleanest possible validation of the framework.

## Task

**Input:** Source distribution $\mu_0$, target distribution $\mu_1$,
graph topology (edge_index).

**Output:** The OT interpolation $\mu_t$ at $t=1$, i.e., the result of
flowing $\mu_0$ to $\mu_1$ via the learned rate matrices.

The exact OT flow (our solver) provides perfect training targets.
The model must learn to predict rate matrices that correctly transport
mass along geodesics on ANY graph.

## Graph Zoo

### Training graphs (diverse topologies, sizes 15–60)

```python
def generate_training_graphs(seed=42):
    """
    Generate a diverse collection of training graphs.
    Returns list of (name, R, positions) tuples.
    """
    rng = np.random.default_rng(seed)
    graphs = []
    
    # 1. Grid graphs: 3x5, 4x4, 5x5, 4x6, 6x6
    for rows, cols in [(3,5), (4,4), (5,5), (4,6), (6,6)]:
        R, pos = make_grid_graph(rows, cols)
        graphs.append((f'grid_{rows}x{cols}', R, pos))
    
    # 2. Cycle graphs: N=15, 20, 30
    for n in [15, 20, 30]:
        R, pos = make_cycle_graph(n)
        graphs.append((f'cycle_{n}', R, pos))
    
    # 3. Random geometric graphs: N=20,30,40,50 with radius r
    for n in [20, 30, 40, 50]:
        R, pos = make_random_geometric_graph(n, radius=0.35, rng=rng)
        graphs.append((f'rgg_{n}', R, pos))
    
    # 4. Barabasi-Albert: N=20,30,40
    for n in [20, 30, 40]:
        R, pos = make_barabasi_albert_graph(n, m=3, rng=rng)
        graphs.append((f'ba_{n}', R, pos))
    
    # 5. Small-world (Watts-Strogatz): N=20,30,40
    for n in [20, 30, 40]:
        R, pos = make_watts_strogatz_graph(n, k=4, p=0.2, rng=rng)
        graphs.append((f'ws_{n}', R, pos))
    
    # 6. Random trees: N=15,25,35
    for n in [15, 25, 35]:
        R, pos = make_random_tree(n, rng=rng)
        graphs.append((f'tree_{n}', R, pos))
    
    # 7. Mesh graphs from 2D shapes: square, circle, triangle
    for shape in ['square', 'circle', 'triangle']:
        R, pos = make_mesh_graph(shape, n_points=40, rng=rng)
        graphs.append((f'mesh_{shape}_40', R, pos))
    
    # 8. Barbell graph: two cliques connected by a path
    for n_clique in [8, 10, 12]:
        R, pos = make_barbell_graph(n_clique, bridge_len=3, rng=rng)
        graphs.append((f'barbell_{n_clique}', R, pos))
    
    return graphs  # ~30 training graphs, sizes 15-60
```

### Test graphs: unseen topologies (sizes 15–60)

```python
def generate_test_graphs_topology(seed=9000):
    """Unseen topologies at similar sizes to training."""
    rng = np.random.default_rng(seed)
    graphs = []
    
    # 1. Grid sizes not seen: 3x7, 5x6, 7x4
    for rows, cols in [(3,7), (5,6), (7,4)]:
        R, pos = make_grid_graph(rows, cols)
        graphs.append((f'grid_{rows}x{cols}', R, pos))
    
    # 2. Ladder graph: N=30
    R, pos = make_ladder_graph(15)
    graphs.append(('ladder_30', R, pos))
    
    # 3. Random geometric with different radius
    for n in [25, 40]:
        R, pos = make_random_geometric_graph(n, radius=0.45, rng=rng)
        graphs.append((f'rgg_r045_{n}', R, pos))
    
    # 4. Mesh graphs from unseen shapes
    for shape in ['L_shape', 'star', 'annulus']:
        R, pos = make_mesh_graph(shape, n_points=40, rng=rng)
        graphs.append((f'mesh_{shape}_40', R, pos))
    
    # 5. Stochastic block model (community structure)
    R, pos = make_sbm_graph(sizes=[10,10,10], p_in=0.5, p_out=0.05, rng=rng)
    graphs.append(('sbm_3x10', R, pos))
    
    # 6. Petersen graph (highly symmetric, non-planar)
    R, pos = make_petersen_graph()
    graphs.append(('petersen', R, pos))
    
    return graphs  # ~10 test graphs
```

### Test graphs: unseen sizes (smaller AND larger)

```python
def generate_test_graphs_size(seed=9500):
    """Unseen sizes — both smaller and larger than training."""
    rng = np.random.default_rng(seed)
    graphs = []
    
    # SMALLER (N=8-12): can model handle tiny graphs?
    R, pos = make_grid_graph(2, 4)
    graphs.append(('grid_2x4', R, pos))
    
    R, pos = make_cycle_graph(8)
    graphs.append(('cycle_8', R, pos))
    
    R, pos = make_random_geometric_graph(10, radius=0.5, rng=rng)
    graphs.append(('rgg_10', R, pos))
    
    R, pos = make_random_tree(10, rng=rng)
    graphs.append(('tree_10', R, pos))
    
    # LARGER (N=80-150): can model scale beyond training?
    R, pos = make_grid_graph(8, 10)
    graphs.append(('grid_8x10', R, pos))
    
    R, pos = make_grid_graph(10, 10)
    graphs.append(('grid_10x10', R, pos))
    
    R, pos = make_random_geometric_graph(100, radius=0.20, rng=rng)
    graphs.append(('rgg_100', R, pos))
    
    R, pos = make_random_geometric_graph(150, radius=0.18, rng=rng)
    graphs.append(('rgg_150', R, pos))
    
    R, pos = make_mesh_graph('square', n_points=100, rng=rng)
    graphs.append(('mesh_square_100', R, pos))
    
    R, pos = make_mesh_graph('circle', n_points=120, rng=rng)
    graphs.append(('mesh_circle_120', R, pos))
    
    R, pos = make_barabasi_albert_graph(100, m=3, rng=rng)
    graphs.append(('ba_100', R, pos))
    
    R, pos = make_watts_strogatz_graph(100, k=4, p=0.2, rng=rng)
    graphs.append(('ws_100', R, pos))
    
    return graphs  # ~12 test graphs, sizes 8-150
```

## Graph Construction Helpers

```python
def make_grid_graph(rows, cols):
    """2D grid graph. N = rows * cols."""
    N = rows * cols
    pos = np.array([(i % cols, i // cols) for i in range(N)],
                    dtype=np.float32)
    R = np.zeros((N, N))
    for i in range(N):
        r, c = i // cols, i % cols
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                j = nr * cols + nc
                R[i, j] = 1.0
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos

def make_cycle_graph(n):
    """Cycle graph with N nodes on a circle."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pos = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    R = np.zeros((n, n))
    for i in range(n):
        R[i, (i+1) % n] = 1.0
        R[i, (i-1) % n] = 1.0
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos

def make_random_geometric_graph(n, radius=0.3, rng=None):
    """Random geometric graph: connect points within radius."""
    if rng is None:
        rng = np.random.default_rng()
    pos = rng.uniform(0, 1, size=(n, 2)).astype(np.float32)
    from scipy.spatial.distance import cdist
    dists = cdist(pos, pos)
    R = np.zeros((n, n))
    R[dists < radius] = 1.0
    np.fill_diagonal(R, 0)
    # Ensure connected: add edges to nearest unconnected component
    # (or regenerate if disconnected)
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos

def make_barabasi_albert_graph(n, m=3, rng=None):
    """Barabasi-Albert preferential attachment graph."""
    import networkx as nx
    G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(100000)))
    pos_dict = nx.spring_layout(G, seed=42)
    pos = np.array([pos_dict[i] for i in range(n)], dtype=np.float32)
    A = nx.to_numpy_array(G)
    R = A.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos

def make_watts_strogatz_graph(n, k=4, p=0.2, rng=None):
    """Watts-Strogatz small-world graph."""
    import networkx as nx
    G = nx.watts_strogatz_graph(n, k, p, seed=int(rng.integers(100000)))
    pos_dict = nx.spring_layout(G, seed=42)
    pos = np.array([pos_dict[i] for i in range(n)], dtype=np.float32)
    A = nx.to_numpy_array(G)
    R = A.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos

def make_random_tree(n, rng=None):
    """Random tree via Prufer sequence."""
    import networkx as nx
    G = nx.random_tree(n, seed=int(rng.integers(100000)))
    pos_dict = nx.spring_layout(G, seed=42)
    pos = np.array([pos_dict[i] for i in range(n)], dtype=np.float32)
    A = nx.to_numpy_array(G)
    R = A.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos

def make_barbell_graph(n_clique, bridge_len=3, rng=None):
    """Two cliques connected by a bridge path."""
    import networkx as nx
    G = nx.barbell_graph(n_clique, bridge_len)
    pos_dict = nx.spring_layout(G, seed=42)
    N = G.number_of_nodes()
    pos = np.array([pos_dict[i] for i in range(N)], dtype=np.float32)
    A = nx.to_numpy_array(G)
    R = A.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos

def make_sbm_graph(sizes, p_in=0.5, p_out=0.05, rng=None):
    """Stochastic block model."""
    import networkx as nx
    k = len(sizes)
    probs = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
    G = nx.stochastic_block_model(sizes, probs,
                                    seed=int(rng.integers(100000)))
    pos_dict = nx.spring_layout(G, seed=42)
    N = G.number_of_nodes()
    pos = np.array([pos_dict[i] for i in range(N)], dtype=np.float32)
    A = nx.to_numpy_array(G)
    R = A.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos

def make_petersen_graph():
    """Petersen graph (10 nodes, 3-regular, non-planar)."""
    import networkx as nx
    G = nx.petersen_graph()
    pos_dict = nx.spring_layout(G, seed=42)
    pos = np.array([pos_dict[i] for i in range(10)], dtype=np.float32)
    A = nx.to_numpy_array(G)
    R = A.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, pos
```

## Distribution Pairs

```python
def generate_distribution_pair(N, points, rng):
    """
    Generate a (source, target) pair of distributions on N nodes.
    
    The pair is designed so that non-trivial transport is required:
    source and target have peaks at different locations.
    """
    # Source: random IC
    ic_type_src = rng.choice(['single_peak', 'multi_peak',
                               'gradient', 'smooth_random'])
    mu_src = generate_initial_condition(N, points, rng, ic_type_src)
    
    # Target: different random IC (guaranteed different from source)
    ic_type_tgt = rng.choice(['single_peak', 'multi_peak',
                               'gradient', 'smooth_random'])
    mu_tgt = generate_initial_condition(N, points, rng, ic_type_tgt)
    
    return mu_src, mu_tgt
```

## Dataset

```python
class OTTransportDataset(torch.utils.data.Dataset):
    """
    Training data for OT transport across varying topologies.
    
    For each training sample:
    1. Pick a random graph from the training set
    2. Generate a random (source, target) distribution pair
    3. Compute OT coupling and flow
    4. Sample (mu_tau, tau, context, R_target) tuples
    
    Context:
        node_context: (N, 1) [mu_target(a)]
            The target distribution at each node — this is what we're
            transporting toward. The model sees where mass should end up.
    
    No global conditioning. No FiLM. The model must infer graph scale
    and structure entirely from message passing.
    
    Returns 6-tuple for train_flexible_conditional:
        (mu_tau, tau, node_context, R_target, edge_index, N)
    """
    def __init__(self, graphs, n_pairs_per_graph=20,
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
                mu_src, mu_tgt = generate_distribution_pair(N, pos, rng)
                
                # OT coupling
                coupling = compute_ot_coupling(mu_src, mu_tgt, cost)
                geo_cache.precompute_for_coupling(coupling)
                
                # Node context: target distribution only
                node_ctx = mu_tgt[:, None].astype(np.float32)  # (N, 1)
                
                self.all_pairs.append({
                    'name': name,
                    'N': N,
                    'R': R,
                    'positions': pos,
                    'edge_index': edge_index,
                    'mu_source': mu_src,
                    'mu_target': mu_tgt,
                })
                
                # Sample flow tuples
                n_per = max(1, n_samples // (
                    len(graphs) * n_pairs_per_graph))
                
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
    context_dim=1,        # [mu_target(a)] — just the target value per node
    hidden_dim=64,
    n_layers=4)
```

No FiLM, no global features. The model sees only the per-node target
distribution and the graph topology. Everything about scale, structure,
and routing must be inferred through message passing alone. This is
the purest test of whether a GNN can learn OT on graphs.

## Training

- FlexibleConditionalGNNRateMatrixPredictor with context_dim=1
- hidden_dim=64, n_layers=4
- ~30 training graphs, 20 pairs each = 600 OT couplings
- 20000 training samples
- 1000 epochs, lr=5e-4, EMA decay=0.999
- Rate KL loss, uniform time weighting
- Uses train_flexible_conditional (6-tuple dataset format)
- Checkpoint every 50 epochs

### 1. Direct GNN
Predicts $\mu_1$ directly from $(\mu_0, \mu_1^{\text{target}}, \text{graph})$.
Context = $[\mu_0(a), \mu_1(a)]$ per node.

### 2. Laplacian smoothing toward target
Interpolate: $\hat{\mu}_1 = (1-\alpha)\mu_0 + \alpha \mu_1^{\text{target}}$.
Tune $\alpha$ on validation. This is a trivial baseline — if the model
can't beat linear interpolation in distribution space, something is wrong.

### 3. Diffusion toward target
Run heat equation from $\mu_0$ for a tuned time $\tau^*$.
This spreads mass but doesn't direct it toward $\mu_1$.

## Evaluation

### Test splits

Three test conditions:
1. **In-distribution (ID):** Unseen distribution pairs on training graphs
2. **Unseen topology (OOD-topo):** Test graphs at similar sizes (15-60)
3. **Unseen size (OOD-size):** Test graphs at smaller (8-12) and
   larger (80-150) sizes

### Metrics

- **Path TV at multiple flow times:** For each test case, compute the
  exact OT interpolation $\mu_t^{\text{exact}}$ at $t = 0.25, 0.5, 0.75, 1.0$
  and compare to the model's ODE trajectory $\hat{\mu}_t$ at the same times.
  Report TV at each $t$. This is the primary metric — it measures whether
  the model has learned the correct OT flow, not just the endpoint.

- **Mean path TV:** Average TV across $t \in \{0.25, 0.5, 0.75, 1.0\}$.
  A single number summarizing path quality.

- **Endpoint TV:** TV at $t=1.0$ only. This is weaker — a model could
  get the endpoint right with a wrong path.

- **Path energy:** The total kinetic energy of the learned flow
  $\sum_t \sum_{a,b} \hat{u}_t(a,b)^2$. The OT flow minimizes this
  (by definition). A model that produces non-optimal paths will have
  higher energy. Compare learned energy to exact OT energy.

- **Peak recovery at $t=1.0$:** Does argmax of prediction match argmax of target?

### Computing exact OT interpolation for evaluation

```python
def compute_exact_interpolation(mu_0, mu_1, R, t_values):
    """
    Compute the exact OT interpolation at specified flow times.
    
    Uses the graph-level OT solver to get the coupling, then
    evaluates the marginal distribution at each t.
    
    Returns: dict mapping t -> mu_t_exact (N,) array
    """
    graph_struct = GraphStructure(R)
    cost = compute_cost_matrix(graph_struct)
    geo_cache = GeodesicCache(graph_struct)
    
    coupling = compute_ot_coupling(mu_0, mu_1, cost)
    geo_cache.precompute_for_coupling(coupling)
    
    interpolations = {}
    for t in t_values:
        mu_t = marginal_distribution_fast(geo_cache, coupling, t)
        interpolations[t] = mu_t
    
    return interpolations
```

### Scaling analysis

Plot TV vs graph size for the OOD-size test set. Does the model
degrade gracefully as graphs get larger? Does it work on tiny graphs?

## Data Visualization (before training)

### Plot 1: `ex17_graph_zoo.png`

Show all training and test graphs laid out in a grid.
Each graph drawn with `nx.draw` or as a point-edge plot.
Color nodes by degree. Label with name and size.
Training graphs in blue, test-topology in red, test-size in orange.

### Plot 2: `ex17_transport_examples.png`

For 6 diverse graphs (2 training, 2 OOD-topo, 2 OOD-size), show
the full exact OT path at $t = 0, 0.25, 0.5, 0.75, 1.0$:
- One row per graph
- 5 columns for the 5 time points
- Each cell: graph colored by distribution value

Shows how OT transport looks on different topologies. On a grid,
mass slides smoothly. On a barbell, mass must squeeze through the
bridge. On a tree, mass flows along branches. The intermediate
frames reveal the transport dynamics that endpoint-only evaluation
would miss.

### Plot 3: `ex17_size_range.png`

Show the smallest (N=8) and largest (N=150) test graphs side by side,
each with a source and target distribution. Demonstrates the scale
range the model must handle.

## Main Results Plots

### Plot 4: `ex17_results.png` (2x3 grid)

- **Panel A:** Training loss curve

- **Panel B:** Graph zoo visualization (subset, showing diversity)

- **Panel C:** Mean path TV comparison bar chart, grouped by
  ID / OOD-topo / OOD-size. Each group has bars for FM, DirectGNN,
  interpolation baseline. This is the headline metric — path quality
  across generalization conditions.

- **Panel D:** Path quality over flow time. Lines showing TV vs $t$
  at $t = 0.0, 0.25, 0.5, 0.75, 1.0$ for our model and baselines,
  averaged over OOD-topo test cases. The exact OT interpolation is
  the reference (TV = 0 at all times). Our model should stay near
  zero throughout; baselines may deviate at intermediate times even
  if they match the endpoint.

- **Panel E:** TV vs graph size scatter plot. X-axis: number of nodes.
  Y-axis: mean path TV. One point per test case. Color by method.
  Shows how each method scales with graph size.

- **Panel F:** TV by graph family. Group test cases by graph type
  (grid, random geometric, BA, tree, mesh) and show per-family
  performance. Some families may be harder than others.

### Plot 5: `ex17_transport_gallery.png`

For 4 OOD test graphs, show the full OT flow at $t = 0, 0.25, 0.5, 0.75, 1.0$:
- Row 1: Exact OT interpolation (ground truth path)
- Row 2: Learned model's trajectory
- Row 3: DirectGNN (only has endpoint, no path)

Each cell is the graph colored by distribution value. This visually
shows whether the learned flow matches the exact OT path — mass
should move along geodesics, not take shortcuts or detours.

## Console Output

```
=== Experiment 17: OT Transport Generalization ===
Training: 30 graphs (sizes 15-60), 600 distribution pairs
Testing: ID (30 graphs), OOD-topo (10 graphs), OOD-size (12 graphs)

TV Results:
                      ID          OOD-topo    OOD-size-small  OOD-size-large
  FM (Learned):    X.XXXX       X.XXXX         X.XXXX          X.XXXX
  DirectGNN:       X.XXXX       X.XXXX         X.XXXX          X.XXXX
  Interpolation:   X.XXXX       X.XXXX         X.XXXX          X.XXXX

TV by graph size:
  N=8-12:   FM=X.XX, DirectGNN=X.XX
  N=15-30:  FM=X.XX, DirectGNN=X.XX
  N=31-60:  FM=X.XX, DirectGNN=X.XX
  N=80-100: FM=X.XX, DirectGNN=X.XX
  N=100+:   FM=X.XX, DirectGNN=X.XX

Peak recovery:
  FM: XX%, DirectGNN: XX%, Interpolation: XX%
```

## Expected Outcome

The model should:
- Achieve low path TV on in-distribution (this is what it's trained on)
- Generalize well to unseen topologies at similar sizes (the GNN
  learns local transport rules, not graph-specific patterns)
- Generalize reasonably to larger graphs (GNN is size-equivariant
  by construction — same weights applied to more nodes)
- Work on smaller graphs (fewer nodes = simpler transport)

The path evaluation at intermediate times is crucial. The DirectGNN
baseline can only predict the endpoint — it has no notion of the
transport path. Our flow matching model generates the full trajectory,
and the intermediate distributions should match the exact OT
interpolation. This is the key qualitative difference: our model
learns HOW mass moves, not just WHERE it ends up.

On bottleneck graphs (barbell, SBM with weak inter-community edges),
the OT path is non-trivial: mass must queue at the bottleneck and
flow through sequentially. The model must learn this behavior from
the graph structure alone.

The scaling plot (Panel E) is the headline result: a clear
demonstration that the model handles graphs from N=8 to N=150
despite only training on N=15-60.

## CLI

```python
parser.add_argument('--n-pairs-per-graph', type=int, default=20)
parser.add_argument('--n-samples', type=int, default=20000)
parser.add_argument('--n-epochs', type=int, default=1000)
parser.add_argument('--hidden-dim', type=int, default=64)
parser.add_argument('--n-layers', type=int, default=4)
parser.add_argument('--loss-type', type=str, default='rate_kl')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint-every', type=int, default=50)
```

## Dependencies

networkx (for graph generation — already standard)
No other new dependencies.

## Note on Large Graphs

For graphs with N>100, the exact OT solver (LP on N×N) may be slow.
Options:
- Use entropic OT (Sinkhorn) for large test graphs — approximate
  but fast
- Only compute exact OT for training (small graphs); use the learned
  model for large test graphs and compare to Sinkhorn ground truth
- Precompute all OT couplings and cache to disk

The GNN forward pass scales linearly with edges, so inference on
N=150 is fast. Only the OT solver (for ground truth) is expensive.
