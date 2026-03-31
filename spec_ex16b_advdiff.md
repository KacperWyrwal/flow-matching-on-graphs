# Experiment 16b: Advection-Diffusion on Varying Meshes

## Motivation

Ex16 used the heat equation (pure diffusion, symmetric rates). The
dynamics reduce to a constant rate matrix per mesh — baselines learn
this trivially. Advection-diffusion adds an asymmetric drift component:
mass is carried by a velocity field while simultaneously diffusing. The
rate matrix is asymmetric ($R_{ab} \neq R_{ba}$), and different velocity
fields on the same mesh give different dynamics. The model must learn
both the diffusive and advective components.

Visually: mass spirals, flows, and spreads across the domain. On
different geometries, the same velocity field creates different patterns
(a vortex on an L-shape wraps around the corner; on an annulus it
circulates the ring).

## Rate Matrix Construction

Given a mesh with node positions $\{x_a\}$, a velocity field
$\vec{v}: \mathbb{R}^2 \to \mathbb{R}^2$, diffusion coefficient $D > 0$,
and advection strength $\alpha > 0$:

```python
def build_advection_diffusion_rate_matrix(points, triangles, v_field,
                                           D=1.0, alpha=1.0):
    """
    Build rate matrix for advection-diffusion on a mesh.
    
    R_ab = D * R_sym_ab + alpha * max(0, v(x_a) . (x_b - x_a)) / |x_b - x_a|
    
    The symmetric part (diffusion) spreads mass equally.
    The asymmetric part (advection) pushes mass along the velocity field.
    
    Args:
        points: (N, 2) node positions
        triangles: (T, 3) triangulation
        v_field: callable, v_field(x) -> (2,) velocity at position x
        D: diffusion coefficient
        alpha: advection strength
    
    Returns:
        R: (N, N) rate matrix (asymmetric, rows sum to zero)
    """
    N = len(points)
    R = np.zeros((N, N))
    
    # Build adjacency from triangulation
    edges = set()
    for tri in triangles:
        for i in range(3):
            for j in range(3):
                if i != j:
                    edges.add((tri[i], tri[j]))
    
    for (a, b) in edges:
        dx = points[b] - points[a]
        dist = np.linalg.norm(dx)
        if dist < 1e-10:
            continue
        
        # Diffusion: symmetric, weight = 1/distance
        diff_rate = D / dist
        
        # Advection: asymmetric, proportional to velocity projection
        v_a = v_field(points[a])
        projection = np.dot(v_a, dx) / dist  # component along edge
        adv_rate = alpha * max(0.0, projection) / dist
        
        R[a, b] = diff_rate + adv_rate
    
    # Set diagonal
    np.fill_diagonal(R, 0.0)
    np.fill_diagonal(R, -R.sum(axis=1))
    return R
```

## Velocity Fields

Each velocity field is definable on any 2D domain. The field type and
parameters serve as conditioning for the model.

```python
def make_velocity_field(field_type, params=None):
    """
    Return a callable v(x) -> (2,) for the given field type.
    
    Field types:
        'uniform':  constant direction, v = (cos(theta), sin(theta))
        'vortex':   rotation around center, v = (-(y-cy), (x-cx))
        'source':   radial outflow from center, v = (x-cx, y-cy)
        'sink':     radial inflow to center, v = (-(x-cx), -(y-cy))
        'shear':    v = (y, 0) — horizontal velocity proportional to y
        'dipole':   combination of source and sink
    """
    if params is None:
        params = {}
    cx = params.get('cx', 0.5)
    cy = params.get('cy', 0.5)
    
    if field_type == 'uniform':
        theta = params.get('theta', 0.0)
        vx, vy = np.cos(theta), np.sin(theta)
        return lambda x: np.array([vx, vy])
    
    elif field_type == 'vortex':
        strength = params.get('strength', 1.0)
        def _vortex(x):
            dx, dy = x[0] - cx, x[1] - cy
            return strength * np.array([-dy, dx])
        return _vortex
    
    elif field_type == 'source':
        strength = params.get('strength', 1.0)
        def _source(x):
            dx, dy = x[0] - cx, x[1] - cy
            r = max(np.sqrt(dx**2 + dy**2), 1e-6)
            return strength * np.array([dx, dy]) / r
        return _source
    
    elif field_type == 'sink':
        strength = params.get('strength', 1.0)
        def _sink(x):
            dx, dy = x[0] - cx, x[1] - cy
            r = max(np.sqrt(dx**2 + dy**2), 1e-6)
            return -strength * np.array([dx, dy]) / r
        return _sink
    
    elif field_type == 'shear':
        strength = params.get('strength', 1.0)
        return lambda x: strength * np.array([x[1] - cy, 0.0])
    
    elif field_type == 'dipole':
        # Source at (cx - 0.2, cy), sink at (cx + 0.2, cy)
        src = np.array([cx - 0.2, cy])
        snk = np.array([cx + 0.2, cy])
        strength = params.get('strength', 1.0)
        def _dipole(x):
            x = np.array(x)
            ds = x - src
            dk = x - snk
            rs = max(np.linalg.norm(ds), 0.05)
            rk = max(np.linalg.norm(dk), 0.05)
            return strength * (ds / rs**2 - dk / rk**2)
        return _dipole
    
    else:
        raise ValueError(f"Unknown field type: {field_type}")
```

## Simulation

The exact solution requires the time-ordered exponential, which we
compute by discretizing time:

```python
def simulate_advection_diffusion(mu_init, R, T, n_substeps=100):
    """
    Integrate d mu/dt = mu R from t=0 to t=T using n_substeps Euler steps.
    
    For accuracy, use small substeps. The matrix exponential of each
    substep is well-approximated by (I + dt*R) for small dt.
    
    Alternatively, for a fixed R (time-independent), just use expm(T*R).
    """
    dt = T / n_substeps
    mu = mu_init.copy()
    
    for _ in range(n_substeps):
        # Euler step: mu(t+dt) ≈ mu(t) + mu(t) * R * dt
        # Or more accurately: mu(t+dt) = mu(t) @ expm(dt * R)
        mu = mu + mu @ R * dt
        mu = np.clip(mu, 0, None)
        mu /= mu.sum() + 1e-15
    
    return mu

def simulate_exact(mu_init, R, T):
    """Exact solution for time-independent R."""
    from scipy.linalg import expm
    P = expm(T * R)
    mu = mu_init @ P
    mu = np.clip(mu, 0, None)
    mu /= mu.sum() + 1e-15
    return mu
```

Note: for a fixed velocity field (time-independent), $R$ is constant
and the exact solution is $\mu(T) = \mu(0) e^{TR}$. This IS a single
matrix exponential — same as heat equation but with asymmetric $R$.

To make the dynamics genuinely time-varying, we could rotate the
velocity field over time. But even with fixed $R$, the asymmetry
makes the problem harder than heat: mass doesn't just spread, it flows
directionally, creating non-trivial patterns that depend on the
velocity field and geometry.

For the first version, use fixed $R$ per (mesh, velocity field) pair
but vary the velocity field across training samples. The model must
generalize across:
- Mesh topologies (8 train, 4 test)
- Velocity field types (vortex, source, shear, uniform)
- Velocity field parameters (direction, strength, center)
- Initial conditions (peaked, multi-peak, gradient, smooth)
- Time step T

## Dataset

```python
VELOCITY_FIELDS = ['uniform', 'vortex', 'source', 'sink', 'shear']

class AdvectionDiffusionDataset(torch.utils.data.Dataset):
    """
    Training data for advection-diffusion on varying meshes.
    
    For each training sample:
    1. Pick a random geometry and mesh
    2. Pick a random velocity field type and parameters
    3. Build the advection-diffusion rate matrix
    4. Generate random initial condition
    5. Simulate to get mu(T) for random T
    6. OT coupling: mu(0) -> mu(T)
    7. Sample flow matching tuples
    
    Context:
        node_context: (N, 4) [mu_current, is_boundary, v_x(a), v_y(a)]
        global_cond:  (3,) [T, D, alpha]
    
    The velocity field at each node is part of the per-node context.
    This tells the model the local drift direction.
    
    Returns 7-tuple for train_film_conditional.
    """
    def __init__(self, geometries, n_meshes_per_geo=10,
                 n_fields_per_mesh=3, n_traj_per_field=3,
                 T_range=(0.05, 0.2), D=1.0, alpha_range=(0.5, 2.0),
                 n_samples=20000, n_points=60, seed=42):
        
        rng = np.random.default_rng(seed)
        all_items = []
        self.all_pairs = []
        
        for geo in geometries:
            for mesh_idx in range(n_meshes_per_geo):
                # Generate mesh
                mesh_seed = int(rng.integers(0, 100000))
                points, triangles, boundary_mask = generate_mesh(
                    geo, n_points, seed=mesh_seed)
                N = len(points)
                bnd = boundary_mask.astype(np.float32)
                
                for field_idx in range(n_fields_per_mesh):
                    # Random velocity field
                    field_type = rng.choice(VELOCITY_FIELDS)
                    theta = float(rng.uniform(0, 2*np.pi))
                    strength = float(rng.uniform(0.5, 2.0))
                    cx = float(rng.uniform(0.3, 0.7))
                    cy = float(rng.uniform(0.3, 0.7))
                    params = {'theta': theta, 'strength': strength,
                              'cx': cx, 'cy': cy}
                    
                    v_field = make_velocity_field(field_type, params)
                    alpha = float(rng.uniform(*alpha_range))
                    
                    # Build rate matrix
                    R = build_advection_diffusion_rate_matrix(
                        points, triangles, v_field, D=D, alpha=alpha)
                    
                    # Check connectivity
                    if not _is_connected(R):
                        continue
                    
                    # Precompute OT structures
                    graph_struct = GraphStructure(R)
                    cost = compute_cost_matrix(graph_struct)
                    geo_cache = GeodesicCache(graph_struct)
                    edge_index = rate_matrix_to_edge_index(R)
                    
                    # Compute velocity at each node for context
                    v_nodes = np.array([v_field(points[a]) for a in range(N)])
                    # Normalize for stability
                    v_max = np.abs(v_nodes).max() + 1e-6
                    v_nodes_norm = v_nodes / v_max
                    
                    for traj_idx in range(n_traj_per_field):
                        ic_type = rng.choice(IC_TYPES)
                        mu_init = generate_initial_condition(
                            N, points, rng, ic_type)
                        T_val = float(rng.uniform(*T_range))
                        
                        # Exact solution
                        mu_final = simulate_exact(mu_init, R, T_val)
                        
                        # OT coupling
                        coupling = compute_ot_coupling(
                            mu_init, mu_final, cost)
                        geo_cache.precompute_for_coupling(coupling)
                        
                        # Node context: [mu_init, boundary, v_x, v_y]
                        node_ctx = np.stack([
                            mu_init, bnd,
                            v_nodes_norm[:, 0],
                            v_nodes_norm[:, 1],
                        ], axis=1).astype(np.float32)
                        
                        # Global context: [T, D, alpha]
                        global_ctx = np.array(
                            [T_val, D, alpha], dtype=np.float32)
                        
                        self.all_pairs.append({
                            'geo': geo,
                            'field_type': field_type,
                            'N': N, 'R': R,
                            'points': points,
                            'triangles': triangles,
                            'boundary': bnd,
                            'edge_index': edge_index,
                            'mu_source': mu_init,
                            'mu_target': mu_final,
                            'v_nodes': v_nodes_norm,
                            'T': T_val, 'D': D, 'alpha': alpha,
                        })
                        
                        # Sample flow matching tuples
                        n_per = max(1, n_samples // (
                            len(geometries) * n_meshes_per_geo
                            * n_fields_per_mesh * n_traj_per_field))
                        
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
                                torch.tensor(global_ctx, dtype=torch.float32),
                                torch.tensor(u_tilde, dtype=torch.float32),
                                edge_index, N,
                            ))
        
        # Shuffle and trim
        idx = rng.permutation(len(all_items))
        self.samples = [all_items[i] for i in idx[:n_samples]]
```

## Model

```python
model = FiLMConditionalGNNRateMatrixPredictor(
    node_context_dim=4,   # [mu_init, is_boundary, v_x, v_y]
    global_dim=3,         # [T, D, alpha]
    hidden_dim=64,
    n_layers=4)
```

The per-node velocity field is the key new input. The model sees the
local drift direction at each node, which tells it how mass should
flow. The FiLM conditioning provides the global parameters.

## Baselines

Same as Ex16: MeshGraphNet, DirectGNN, Euler step. All receive the
same velocity field information. The Euler baseline becomes:

```python
def baseline_euler_advdiff(mu, R, T):
    """First-order Euler: mu(T) ≈ mu @ (I + T*R)"""
    mu_next = mu @ (np.eye(len(mu)) + T * R)
    mu_next = np.clip(mu_next, 0, None)
    return mu_next / (mu_next.sum() + 1e-15)
```

## Data Visualization (before training)

Generate diagnostic plots of the training data, similar to Ex16:

### Plot 1: `ex16b_data_viz.png`

For each of the 12 geometries (8 train + 4 test), one row showing:
- Column 1: Mesh with velocity field arrows overlaid
- Column 2: Initial condition (color on mesh)
- Column 3: After advection-diffusion at T=0.1 (color on mesh)

The velocity arrows are the key visual — they show the drift direction
at each node. Use `ax.quiver()` on the node positions.

### Plot 2: `ex16b_field_variety.png`

On a single mesh (e.g., square), show all 5 velocity field types:
- Row 1: velocity arrows for uniform, vortex, source, sink, shear
- Row 2: initial condition (same peaked IC for all)
- Row 3: result after T=0.1

Shows how different fields create different transport patterns on the
same geometry.

### Plot 3: `ex16b_evolution.png`

For one (geometry, field) pair (e.g., vortex on L_shape), show the
evolution at 7 time steps: t=0, 0.02, 0.04, ..., 0.12. Each panel
shows the distribution on the mesh with velocity arrows faintly
overlaid. Shows mass spiraling around the L-shape corner.

## Evaluation

Same structure as Ex16:
- In-distribution: 8 training geometries
- Out-of-distribution: 4 test geometries (topology generalization)
- Long-horizon rollout: apply repeatedly for K steps
- Split by velocity field type

Additional evaluation:
- **Asymmetry test:** Does the model correctly predict directional
  transport? Measure whether mass moves in the correct direction
  for uniform and vortex fields.
- **Field generalization:** Train on {uniform, vortex, source},
  test on {shear, sink}. Can the model generalize to unseen
  velocity field types?

## Plots

### Main figure: `ex16b_advdiff.png` (2x3 grid)

Same layout as Ex16 (training loss, mesh geometries, TV comparison,
example reconstruction, long-horizon TV curve, generalization gap).

Panel D should show velocity field arrows overlaid on the mesh, with
the distribution colored underneath.

### Rollout figure: `ex16b_rollout.png` (separate, detailed)

A multi-row figure showing the full rollout for one OOD test case
(e.g., vortex on L_shape, K=10 steps):

- **Row 1 (Ground truth):** Exact solution at t=0, dt, 2dt, ..., Kdt.
  Each panel shows distribution on mesh with faint velocity arrows.

- **Row 2 (Flow matching):** Our model's rollout at the same time steps.

- **Row 3 (MeshGraphNet):** MeshGraphNet baseline's rollout.

- **Row 4 (DirectGNN):** DirectGNN baseline's rollout.

- **Row 5 (Laplacian/Euler):** Euler baseline's rollout.

Each panel is a small mesh plot colored by distribution value,
with shared colorbar per row. Annotate each panel with the TV
distance from exact at that step.

This figure directly shows WHERE each method goes wrong over time.
The exact solution shows mass spiraling; a good model tracks this;
a bad model either smears mass uniformly or sends it the wrong way.

```python
def plot_rollout_comparison(model, mgn, direct_gnn, test_case,
                             K=10, device='cpu', save_path=None):
    """
    Plot side-by-side rollout for all methods vs exact.
    
    5 rows x (K+1) columns. Each cell is a small mesh plot.
    """
    fig, axes = plt.subplots(5, K+1, figsize=(2.5*(K+1), 12))
    
    row_labels = ['Exact', 'Flow Matching', 'MeshGraphNet',
                  'DirectGNN', 'Euler']
    
    # Compute all rollouts...
    # For each method and each step, plot tripcolor on mesh
    # Annotate with TV from exact
    
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10, rotation=90,
                                  labelpad=15)
    
    for col in range(K+1):
        axes[0, col].set_title(f't={col*dt:.3f}', fontsize=8)
    
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
```

## CLI

```python
# Inherits all Ex16 arguments plus:
parser.add_argument('--alpha-range', type=float, nargs=2, default=[0.5, 2.0],
                    help='Advection strength range')
parser.add_argument('--diffusion-coeff', type=float, default=1.0)
parser.add_argument('--T-range', type=float, nargs=2, default=[0.05, 0.2])
parser.add_argument('--n-fields-per-mesh', type=int, default=3)
```

## Expected Outcome

The asymmetric rates make this harder than heat:
- The Euler baseline should be worse (asymmetric rates amplify
  the first-order error)
- MeshGraphNet and DirectGNN should still be strong (they can
  learn from the velocity context)
- Our model should be competitive on single-step, with advantages
  on long-horizon rollout due to distribution preservation

The topology generalization story carries over from Ex16: train on
square/circle, predict advection-diffusion on L-shape/star with
the same velocity field types.

The velocity field visualization makes this a much more compelling
figure than pure heat diffusion.
