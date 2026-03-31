# Amendment: Experiment 13 — FiLM + Spatial Sensor Conditioning

## Problem

The backprojection A^T y is too blurry to serve as useful per-node context.
The learned model performs worse than all baselines because every node sees
roughly the same context value — no spatial signal for the GNN to work with.

## Fix: Dual Conditioning

Combine two complementary information channels:

### 1. Local: Sensor values at sensor nodes (per-node features)

Place each sensor's reading directly at its graph location. Non-sensor
nodes get zero. Plus a binary mask indicating which nodes are sensors.

Per-node features: [mu(a), t, sensor_value(a) * is_sensor(a), is_sensor(a)]

This gives the GNN sharp, spatially localized information at 20 points.
Message passing propagates it inward through the graph — exactly what
worked in Ex12b with boundary observations.

### 2. Global: FiLM conditioning from raw sensor vector y

The 20-dimensional sensor reading y is injected globally via FiLM
(Feature-wise Linear Modulation) at each message passing layer. This
gives the model access to cross-sensor correlations and global patterns
that the sparse per-node signal doesn't capture.

## Architecture Changes

### FiLM-conditioned message passing layer

```python
class FiLMRateMessagePassing(MessagePassing):
    """
    Message passing layer with FiLM conditioning from a global vector.
    
    After the standard message + update step, modulate node features:
        h_a <- gamma * h_a + beta
    where (gamma, beta) = MLP(global_conditioning)
    """
    def __init__(self, in_dim, hidden_dim, global_dim):
        super().__init__(aggr='add')
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # FiLM: global vector -> (gamma, beta) for modulating hidden features
        self.film_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),  # outputs gamma and beta
        )
    
    def forward(self, x, edge_index, global_cond):
        # Standard message passing
        aggr = self.propagate(edge_index, x=x)
        h = self.update_mlp(torch.cat([x, aggr], dim=-1))
        
        # FiLM modulation
        film_params = self.film_mlp(global_cond)  # (batch, 2*hidden)
        gamma, beta = film_params.chunk(2, dim=-1)  # each (batch, hidden)
        # Broadcast gamma, beta to all nodes
        # gamma: (1, hidden) or (batch, hidden) -> applied to h: (N, hidden)
        h = gamma * h + beta
        
        return h
    
    def message(self, x_i, x_j):
        return self.msg_mlp(torch.cat([x_i, x_j], dim=-1))
```

### Updated model

```python
class FiLMConditionalGNNRateMatrixPredictor(nn.Module):
    """
    GNN with dual conditioning:
    - Per-node context features (sensor values at sensor locations + mask)
    - Global FiLM conditioning from raw sensor vector y
    
    Constructor args:
        node_context_dim: int = 2  (sensor_value * is_sensor, is_sensor)
        global_dim: int = 20  (raw sensor vector dimensionality)
        hidden_dim: int = 128
        n_layers: int = 6
    
    forward(mu, t, node_context, global_cond, edge_index):
        mu: (N,) current distribution
        t: scalar, flow time
        node_context: (N, node_context_dim) per-node features
        global_cond: (global_dim,) raw sensor readings + tau_diff
        edge_index: (2, E)
        returns: (N, N) rate matrix
    """
    def __init__(self, node_context_dim=2, global_dim=21,
                 hidden_dim=128, n_layers=6):
        super().__init__()
        
        # Global conditioning encoder: map raw y + tau_diff to hidden dim
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Message passing layers with FiLM
        input_dim = 2 + node_context_dim  # mu(a), t, node_context
        self.mp_layers = nn.ModuleList()
        self.mp_layers.append(
            FiLMRateMessagePassing(input_dim, hidden_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.mp_layers.append(
                FiLMRateMessagePassing(hidden_dim, hidden_dim, hidden_dim))
        
        # Edge readout (same as before)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, mu, t, node_context, global_cond, edge_index):
        N = mu.shape[0]
        
        # Node features: [mu(a), t, sensor_val*is_sensor, is_sensor]
        t_expanded = torch.full((N,), t, device=mu.device)
        h = torch.cat([mu.unsqueeze(-1), t_expanded.unsqueeze(-1),
                        node_context], dim=-1)  # (N, 2+node_context_dim)
        
        # Encode global conditioning
        g = self.global_encoder(global_cond)  # (hidden_dim,)
        
        # Message passing with FiLM
        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index, g)
        
        # Edge readout (same as before)
        src, dst = edge_index
        edge_features = torch.cat([h[src], h[dst]], dim=-1)
        edge_rates = F.softplus(self.edge_mlp(edge_features).squeeze(-1))
        
        rate_matrix = torch.zeros(N, N, device=mu.device)
        rate_matrix[src, dst] = edge_rates
        rate_matrix[range(N), range(N)] = -rate_matrix.sum(dim=-1)
        
        return rate_matrix
```

## Context Construction

```python
def build_sensor_context(y, sensor_nodes, N, tau_diff):
    """
    Build per-node and global context from sensor readings.
    
    Returns:
        node_context: (N, 2) — [sensor_val * is_sensor, is_sensor]
        global_cond: (21,) — [y (20 dims), tau_diff (1 dim)]
    """
    # Per-node: sensor value at sensor locations, zero elsewhere
    sensor_vals = np.zeros(N)
    is_sensor = np.zeros(N)
    for m, node in enumerate(sensor_nodes):
        sensor_vals[node] = y[m]
        is_sensor[node] = 1.0
    node_context = np.stack([sensor_vals, is_sensor], axis=-1)  # (N, 2)
    
    # Global: raw sensor vector + diffusion time
    global_cond = np.concatenate([y, [tau_diff]])  # (21,)
    
    return node_context, global_cond
```

## Dataset Changes

```python
class SparseSensorFiLMDataset(torch.utils.data.Dataset):
    """
    Training data for sparse sensor reconstruction with FiLM conditioning.
    
    Each sample includes:
        mu_tau: (N,) distribution at flow time
        tau: (1,) flow time
        node_context: (N, 2) sensor values at sensor nodes + mask
        global_cond: (21,) raw sensor vector y + tau_diff
        R_target: (N, N) factorized target rate matrix
        edge_index: (2, E)
        n_nodes: int
    """
    def __init__(self, R, sensor_nodes, mixing_matrix,
                 clean_distributions, tau_diffs,
                 n_samples=15000, seed=42):
        rng = np.random.default_rng(seed)
        N = R.shape[0]
        A = mixing_matrix
        graph_struct = GraphStructure(R)
        cost = compute_cost_matrix(graph_struct)
        cache = GeodesicCache(graph_struct)
        edge_index = rate_matrix_to_edge_index(R)
        
        mu_start = np.ones(N) / N
        
        pairs = []
        for mu_clean, td in zip(clean_distributions, tau_diffs):
            # Sensor observation
            mu_diffused = mu_clean @ expm(td * R)
            y = A @ mu_diffused
            
            # Build contexts
            node_ctx, global_ctx = build_sensor_context(
                y, sensor_nodes, N, td)
            
            # OT coupling: uniform -> source
            pi = compute_ot_coupling(mu_start, mu_clean, cost)
            cache.precompute_for_coupling(pi)
            pairs.append((mu_clean, node_ctx, global_ctx, pi))
        
        self.samples = []
        for _ in range(n_samples):
            mu_clean, node_ctx, global_ctx, pi = \
                pairs[int(rng.integers(len(pairs)))]
            tau = float(rng.uniform(0.0, 0.999))
            
            mu_tau = marginal_distribution_fast(cache, pi, tau)
            R_target = (1.0 - tau) * marginal_rate_matrix_fast(
                cache, pi, tau)
            
            self.samples.append((
                torch.tensor(mu_tau, dtype=torch.float32),
                torch.tensor([tau], dtype=torch.float32),
                torch.tensor(node_ctx, dtype=torch.float32),
                torch.tensor(global_ctx, dtype=torch.float32),
                torch.tensor(R_target, dtype=torch.float32),
                edge_index,
                N,
            ))
```

## Training Loop Changes

The training loop needs to handle the extra global_cond tensor:

```python
def train_film_conditional(model, dataset, n_epochs=1000, batch_size=256,
                            lr=5e-4, device=None, loss_weighting='uniform'):
    """
    Training loop for FiLMConditionalGNNRateMatrixPredictor.
    Dataset returns (mu, tau, node_context, global_cond, R_target,
                     edge_index, n_nodes).
    """
    # Process each sample individually (variable graph sizes)
    # For fixed graph: can batch by stacking mu, tau, node_context,
    # global_cond, R_target and sharing edge_index
```

## Sampling Changes

```python
def sample_trajectory_film(model, mu_start, node_context, global_cond,
                            edge_index, n_steps=200, device='cpu'):
    """
    Integrate flow with FiLM conditioning.
    node_context and global_cond are fixed throughout the trajectory.
    """
    # At each step:
    #   R_tilde = model(mu, t, node_context, global_cond, edge_index)
    #   R = R_tilde / (1 - t)
    #   mu_next = mu + dt * mu @ R
```

## CLI

```python
parser.add_argument('--conditioning', type=str, default='film',
                    choices=['backproj', 'film'],
                    help='Conditioning method: backprojection only or FiLM + spatial')
```

## Why This Should Work

In Ex12b, the model saw boundary values at 98 nodes and reconstructed
27 interior nodes — a 3.6:1 observed-to-hidden ratio. It worked
beautifully because the observations were sharp and spatially localized.

With FiLM + spatial: 20 sensor nodes have sharp values (like Ex12b's
boundary), 105 nodes have zeros (like Ex12b's interior), plus FiLM
provides global context from the full sensor vector. The ratio is worse
(0.16:1) but the FiLM compensates by giving the model cross-sensor
information that pure per-node features can't capture.

The key insight: the GNN's message passing is designed to propagate
local information through the graph. Give it local information (sensor
values at specific nodes) and let it propagate. Don't pre-blur the
information with a backprojection and then ask the GNN to un-blur it.

## Dependencies

No new dependencies.
