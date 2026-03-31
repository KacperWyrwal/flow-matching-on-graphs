# Experiment 16: Heat Equation on Varying Meshes

## Motivation

Demonstrate forward physics simulation with topology generalization.
Train on heat equation simulations across multiple mesh geometries,
test on unseen geometries. The heat equation is ideal because:
- The exact solution is our framework (heat kernel = matrix exponential)
- Temperature distribution is naturally non-negative and normalizable
- Different domain shapes give different mesh topologies
- This connects to the large learned-simulator literature (MeshGraphNets)

## Setup

### Mesh generation

Generate 2D meshes for various domain shapes using Delaunay
triangulation. Each mesh becomes a graph.

```python
import numpy as np
from scipy.spatial import Delaunay

def generate_mesh(shape, n_points=80, seed=42):
    """
    Generate a 2D mesh for a given domain shape.
    
    Args:
        shape: str or dict describing the domain
        n_points: approximate number of interior points
        seed: random seed
    
    Returns:
        points: (N, 2) node coordinates
        triangles: (T, 3) triangle indices
        boundary_mask: (N,) boolean, True for boundary nodes
    """
    rng = np.random.default_rng(seed)
    
    if shape == 'square':
        # Unit square [0,1]^2
        # Boundary points
        n_boundary = int(np.sqrt(n_points)) * 4
        boundary = []
        for i in range(n_boundary):
            t = i / n_boundary
            if t < 0.25:
                boundary.append([4*t, 0])
            elif t < 0.5:
                boundary.append([1, 4*(t-0.25)])
            elif t < 0.75:
                boundary.append([1-4*(t-0.5), 1])
            else:
                boundary.append([0, 1-4*(t-0.75)])
        boundary = np.array(boundary)
        # Interior points
        interior = rng.uniform(0.05, 0.95, size=(n_points, 2))
        points = np.vstack([boundary, interior])
        
    elif shape == 'circle':
        # Unit circle centered at (0.5, 0.5)
        n_boundary = int(np.sqrt(n_points)) * 4
        angles = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
        boundary = np.stack([0.5 + 0.5*np.cos(angles),
                              0.5 + 0.5*np.sin(angles)], axis=1)
        # Interior: rejection sampling
        interior = []
        while len(interior) < n_points:
            p = rng.uniform(0, 1, size=(2,))
            if np.linalg.norm(p - 0.5) < 0.45:
                interior.append(p)
        interior = np.array(interior)
        points = np.vstack([boundary, interior])
        
    elif shape == 'L_shape':
        # L-shaped domain: unit square minus upper-right quadrant
        n_boundary = int(np.sqrt(n_points)) * 5
        # Generate boundary of L-shape
        boundary = _l_shape_boundary(n_boundary)
        interior = []
        while len(interior) < n_points:
            p = rng.uniform(0, 1, size=(2,))
            if p[0] < 0.5 or p[1] < 0.5:  # inside L
                interior.append(p)
        interior = np.array(interior)
        points = np.vstack([boundary, interior])
    
    elif shape == 'triangle':
        # Equilateral triangle
        vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
        # ... generate boundary and interior points
        
    elif shape == 'annulus':
        # Ring domain: outer radius 0.5, inner radius 0.2
        # ... generate boundary and interior points
        
    elif shape == 'star':
        # Star-shaped domain with 5 points
        # ... generate boundary and interior points
        
    elif shape == 'rectangle':
        # Rectangle with aspect ratio as parameter
        # ... generate boundary and interior points
    
    # Delaunay triangulation
    tri = Delaunay(points)
    triangles = tri.simplices
    
    # Identify boundary nodes
    boundary_mask = _compute_boundary_mask(points, triangles, shape)
    
    return points, triangles, boundary_mask


def mesh_to_graph(points, triangles):
    """
    Convert a triangular mesh to a graph rate matrix.
    
    Nodes = mesh vertices. Two nodes are connected if they share
    a triangle edge. Edge weight = 1/distance (closer nodes have
    stronger coupling, physically motivated by FEM discretization).
    
    Returns:
        R: (N, N) rate matrix
        adj: (N, N) adjacency matrix
    """
    N = len(points)
    adj = np.zeros((N, N))
    
    for tri in triangles:
        for i in range(3):
            for j in range(3):
                if i != j:
                    a, b = tri[i], tri[j]
                    dist = np.linalg.norm(points[a] - points[b])
                    weight = 1.0 / max(dist, 1e-6)
                    adj[a, b] = max(adj[a, b], weight)
                    adj[b, a] = max(adj[b, a], weight)
    
    R = adj.copy()
    np.fill_diagonal(R, -R.sum(axis=1))
    return R, adj
```

### Domain shapes

Training geometries (8 shapes):
- square, circle, rectangle (2:1), rectangle (1:2)
- triangle, pentagon, hexagon, ellipse

Test geometries (4 shapes, unseen):
- L_shape, annulus, star, trapezoid

Each shape generates a mesh with ~60-100 nodes. Different random
seeds give different meshes for the same shape (point cloud varies).

### Heat equation simulation

The heat equation on the graph is:
    dμ/dt = μR

The exact solution is:
    μ(t) = μ(0) · exp(tR)

For training data, we don't need a numerical solver — the matrix
exponential gives the exact answer.

```python
from scipy.linalg import expm

def simulate_heat(mu_initial, R, dt, n_steps):
    """
    Simulate heat equation on a graph.
    
    Args:
        mu_initial: (N,) initial temperature distribution (normalized)
        R: (N, N) rate matrix (graph Laplacian with negative diagonal)
        dt: time step
        n_steps: number of steps
    
    Returns:
        trajectory: (n_steps+1, N) temperature at each time step
    """
    trajectory = [mu_initial]
    P = expm(dt * R)  # transition matrix for one step
    mu = mu_initial.copy()
    for _ in range(n_steps):
        mu = mu @ P
        mu = np.clip(mu, 0, None)
        mu /= mu.sum()
        trajectory.append(mu)
    return np.array(trajectory)
```

### Initial conditions

Random initial temperature distributions:
- Single hot spot: peaked distribution at a random node
- Multiple hot spots: 2-3 peaks
- Smooth gradient: linear temperature gradient across the domain
- Random smooth: random Fourier modes on the mesh

```python
def generate_initial_condition(N, points, rng, ic_type='random'):
    """
    Generate a random initial temperature distribution.
    """
    if ic_type == 'single_peak':
        peak = int(rng.integers(N))
        mu = np.ones(N) * 0.1 / N
        mu[peak] = 0.9
        mu /= mu.sum()
        
    elif ic_type == 'multi_peak':
        n_peaks = int(rng.integers(2, 4))
        peaks = rng.choice(N, size=n_peaks, replace=False)
        weights = rng.dirichlet(np.full(n_peaks, 2.0))
        mu = np.ones(N) * 0.1 / N
        for p, w in zip(peaks, weights):
            mu[p] += 0.9 * w
        mu /= mu.sum()
        
    elif ic_type == 'gradient':
        # Linear gradient in a random direction
        direction = rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        projections = points @ direction
        mu = projections - projections.min() + 0.01
        mu /= mu.sum()
        
    elif ic_type == 'smooth_random':
        # Sum of a few Gaussian bumps at random locations
        n_bumps = int(rng.integers(2, 6))
        mu = np.ones(N) * 0.01
        for _ in range(n_bumps):
            center = rng.uniform(0, 1, size=2)
            sigma = float(rng.uniform(0.1, 0.3))
            dists = np.linalg.norm(points - center, axis=1)
            mu += np.exp(-0.5 * (dists / sigma) ** 2)
        mu /= mu.sum()
    
    return mu
```

## Task

Given temperature distribution at time $t$, predict distribution at
time $t + \Delta t$.

The model receives:
- Current distribution μ(t) as the starting point of the flow
- Time step Δt as global conditioning (via FiLM)
- Graph topology (edge_index) as input

The OT coupling transports μ(t) to μ(t+Δt) on the mesh graph.
The model learns the rate matrices for this transport.

This is a forward problem — our strongest setting.

## Dataset

```python
class HeatMeshDataset(torch.utils.data.Dataset):
    """
    Training data for heat equation on varying meshes.
    
    For each training sample:
    1. Pick a random geometry and mesh
    2. Generate random initial condition
    3. Simulate heat equation for a random number of steps
    4. Pick a random (t, t+dt) pair from the trajectory
    5. Compute OT coupling: μ(t) -> μ(t+dt)
    6. Sample flow: (mu_tau, tau, global_cond, R_target)
    
    Context:
        node_context: (N, 2) [mu_current(a), is_boundary(a)]
        global_cond: (1,) [dt] — the time step size
        
    The graph varies per sample (topology generalization).
    
    Returns 7-tuple for train_film_conditional:
        (mu_tau, tau, node_context, global_cond, R_target, edge_index, N)
    """
    def __init__(self, geometries, n_meshes_per_geo=10,
                 n_trajectories_per_mesh=5, dt_range=(0.01, 0.1),
                 n_samples=20000, seed=42):
        rng = np.random.default_rng(seed)
        
        # Generate all meshes
        meshes = []
        for geo in geometries:
            for mesh_seed in range(n_meshes_per_geo):
                points, triangles, boundary = generate_mesh(
                    geo, n_points=80, seed=seed + mesh_seed * 1000)
                R, adj = mesh_to_graph(points, triangles)
                edge_index = rate_matrix_to_edge_index(R)
                N = len(points)
                meshes.append({
                    'geometry': geo,
                    'R': R,
                    'edge_index': edge_index,
                    'N': N,
                    'points': points,
                    'boundary': boundary,
                })
        
        # Generate trajectories and OT couplings
        all_pairs = []
        for mesh in meshes:
            R = mesh['R']
            N = mesh['N']
            graph_struct = GraphStructure(R)
            cost = compute_cost_matrix(graph_struct)
            cache = GeodesicCache(graph_struct)
            
            for _ in range(n_trajectories_per_mesh):
                ic_type = rng.choice(['single_peak', 'multi_peak',
                                       'gradient', 'smooth_random'])
                mu_init = generate_initial_condition(
                    N, mesh['points'], rng, ic_type)
                
                dt = float(rng.uniform(*dt_range))
                # Simulate one step
                mu_next = mu_init @ expm(dt * R)
                mu_next = np.clip(mu_next, 1e-10, None)
                mu_next /= mu_next.sum()
                
                # OT coupling: mu_init -> mu_next
                pi = compute_ot_coupling(mu_init, mu_next, cost)
                cache.precompute_for_coupling(pi)
                
                node_ctx = np.stack([mu_init, mesh['boundary'].astype(float)],
                                     axis=-1)
                global_ctx = np.array([dt])
                
                all_pairs.append({
                    'mu_source': mu_init,
                    'mu_target': mu_next,
                    'node_ctx': node_ctx,
                    'global_ctx': global_ctx,
                    'coupling': pi,
                    'cache': cache,
                    'edge_index': mesh['edge_index'],
                    'N': N,
                })
        
        # Sample training tuples
        self.samples = []
        for _ in range(n_samples):
            pair = all_pairs[int(rng.integers(len(all_pairs)))]
            tau = float(rng.uniform(0.0, 0.999))
            
            mu_tau = marginal_distribution_fast(pair['cache'],
                                                 pair['coupling'], tau)
            R_target = (1.0 - tau) * marginal_rate_matrix_fast(
                pair['cache'], pair['coupling'], tau)
            
            self.samples.append((
                torch.tensor(mu_tau, dtype=torch.float32),
                torch.tensor([tau], dtype=torch.float32),
                torch.tensor(pair['node_ctx'], dtype=torch.float32),
                torch.tensor(pair['global_ctx'], dtype=torch.float32),
                torch.tensor(R_target, dtype=torch.float32),
                pair['edge_index'],
                pair['N'],
            ))
```

## Model

```python
model = FiLMConditionalGNNRateMatrixPredictor(
    node_context_dim=2,   # [current_temp, is_boundary]
    global_dim=1,         # dt (time step size)
    hidden_dim=64,
    n_layers=4)
```

Small model because individual meshes are small (~60-100 nodes).
The complexity comes from generalizing across topologies, not from
graph size.

## Training

- 8 training geometries × 10 meshes × 5 trajectories = 400 pairs
- 20000 training samples
- 1000 epochs, lr=5e-4, EMA decay=0.999
- Rate KL loss, uniform time weighting
- train_film_conditional (handles variable graph sizes)

## Baselines

### 1. Exact solution (upper bound)
μ(t+dt) = μ(t) @ expm(dt * R). Perfect, but requires knowing R.
This is the ground truth, not a practical baseline — it shows the
best possible performance.

### 2. MeshGraphNet-style baseline
Standard GNN that predicts node-level updates:
```python
class MeshGraphNetBaseline(nn.Module):
    """Predict Δμ = μ(t+dt) - μ(t) directly."""
    def forward(self, mu, dt, boundary, edge_index):
        # Message passing -> per-node Δμ prediction
        mu_next = mu + delta_mu
        mu_next = softmax(mu_next)  # ensure valid distribution
        return mu_next
```

### 3. GNN + softmax (direct prediction)
Same as our DirectGNNPredictor. Predicts μ(t+dt) directly from
μ(t) in one forward pass.

### 4. Laplacian smoothing
μ(t+dt) ≈ μ(t) @ (I + dt * R). First-order Euler approximation.
Doesn't require learning but is only accurate for small dt.

## Evaluation

### In-distribution test
Generate new meshes for training geometries (different random seeds).
Evaluate prediction quality.

### Out-of-distribution test (TOPOLOGY GENERALIZATION)
Generate meshes for 4 unseen geometries: L_shape, annulus, star,
trapezoid. The model has never seen these mesh topologies.

### Metrics
- **TV between predicted and exact μ(t+dt)** — primary metric
- **Per-node MSE** — standard in the simulation literature
- **Long-horizon rollout:** Apply the model repeatedly for K steps
  and measure error accumulation. OT-based model should accumulate
  less error because each step preserves distribution validity.
- **Conservation error:** Does the predicted distribution sum to 1?
  (Our model guarantees this by construction; baselines may drift.)

### Test matrix
- Geometry: 4 training (in-dist) + 4 test (out-of-dist)
- dt: {0.01, 0.05, 0.1}
- Initial condition type: {single_peak, multi_peak, gradient, smooth}
- 10 cases per combination

## Plots

### Figure 1: Main results (2×3)

- **Panel A:** Training loss curve

- **Panel B:** Mesh visualization. Show 4 training meshes and 4 test
  meshes with their triangulations. Highlight the structural diversity.

- **Panel C:** TV comparison. Grouped bars: learned vs MeshGraphNet vs
  GNN+softmax vs Laplacian. Split by in-distribution vs out-of-distribution.
  THE headline result: does the model generalize to unseen geometries?

- **Panel D:** Heat evolution example. For one test geometry (e.g., L_shape):
  show μ(t), predicted μ(t+dt), and exact μ(t+dt) as color maps on the
  mesh. The predicted should match the exact.

- **Panel E:** Long-horizon rollout. Plot TV vs number of steps (1-20)
  for learned model vs baselines. The learned model should accumulate
  less error due to distribution-preserving transport.

- **Panel F:** Generalization gap. For each test geometry: TV for
  learned model vs in-distribution performance. Small gap = good
  generalization. Compare to MeshGraphNet baseline's gap.

### Figure 2: Posterior sampling (optional)

For the forward heat prediction, posterior sampling shows: "given the
current temperature, here are multiple plausible future states." With
heat equation this is less interesting (the solution is deterministic)
but with noisy measurements of the current state, the posterior would
show how measurement uncertainty propagates through the dynamics.

## Console Output

```
=== Experiment 16: Heat Equation on Varying Meshes ===
Training: 8 geometries, 80 meshes, 400 trajectory pairs
Testing: 4 unseen geometries, 40 meshes

In-distribution TV:
  Learned:        X.XXXX ± X.XXXX
  MeshGraphNet:   X.XXXX ± X.XXXX
  GNN+softmax:    X.XXXX ± X.XXXX
  Laplacian:      X.XXXX ± X.XXXX

Out-of-distribution TV (TOPOLOGY GENERALIZATION):
  Learned:        X.XXXX ± X.XXXX
  MeshGraphNet:   X.XXXX ± X.XXXX
  GNN+softmax:    X.XXXX ± X.XXXX
  Laplacian:      X.XXXX ± X.XXXX

Generalization gap (OOD - ID):
  Learned:        X.XXXX
  MeshGraphNet:   X.XXXX

Long-horizon rollout (20 steps, TV):
  Learned:        X.XXXX
  MeshGraphNet:   X.XXXX
  GNN+softmax:    X.XXXX
```

## Expected Outcome

The learned model should:
- Match or beat MeshGraphNet on in-distribution (both have graph
  structure, but ours has exact OT training signal)
- Generalize better to unseen geometries (OT-derived rates capture
  universal transport principles, not geometry-specific patterns)
- Accumulate less error on long rollouts (distribution preservation
  by construction prevents drift)

The Laplacian baseline (Euler step) should be good for small dt but
degrade at larger dt. The MeshGraphNet baseline predicts deltas and
may violate distribution constraints over long rollouts.

The topology generalization result is the headline: a model trained
on squares and circles predicts heat flow on L-shapes and stars.

## CLI

```python
parser.add_argument('--n-points', type=int, default=80,
                    help='Approximate nodes per mesh')
parser.add_argument('--n-meshes-per-geo', type=int, default=10)
parser.add_argument('--n-traj-per-mesh', type=int, default=5)
parser.add_argument('--dt-range', type=float, nargs=2, default=[0.01, 0.1])
parser.add_argument('--n-samples', type=int, default=20000)
parser.add_argument('--n-epochs', type=int, default=1000)
parser.add_argument('--hidden-dim', type=int, default=64)
parser.add_argument('--n-layers', type=int, default=4)
parser.add_argument('--mode', type=str, default='point_estimate',
                    choices=['point_estimate', 'posterior'])
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint-every', type=int, default=50)
```

## Dependencies

scipy (Delaunay triangulation — already available)
No new dependencies.
