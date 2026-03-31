# Fix: Edge Features in GNN + Ex19 Update

## Motivation

Ex19 (stationary distribution prediction) encodes asymmetric graph
structure via per-node summary features (in_strength, out_strength,
etc.). These are lossy — they discard which specific edges are strong
in which direction. Adding edge features to the GNN message passing
allows the model to see the raw rates R_ab per edge, preserving all
directional information.

This is a general architecture improvement that benefits any task
where edge weights matter (Ex16b advection-diffusion, future
directed graph experiments).

## Architecture Change: Edge-Aware Message Passing

### Current message passing (RateMessagePassing)

```python
class RateMessagePassing(nn.Module):
    def forward(self, h, edge_index):
        # Aggregate neighbor features
        src, dst = edge_index
        messages = self.msg_mlp(torch.cat([h[src], h[dst]], dim=-1))
        agg = scatter_add(messages, dst, dim=0, dim_size=N)
        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))
        return h_new
```

### Updated: EdgeAwareMessagePassing

```python
class EdgeAwareMessagePassing(nn.Module):
    """Message passing with optional edge features.
    
    If edge_feat is None, behaves identically to RateMessagePassing.
    If edge_feat is provided, concatenates it to the message input.
    """
    def __init__(self, in_dim, hidden_dim, edge_dim=0):
        super().__init__()
        msg_input_dim = 2 * in_dim + edge_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Residual connection
        self.residual = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
    
    def forward(self, h, edge_index, edge_feat=None):
        """
        h:          (N, in_dim) node features
        edge_index: (2, E) directed edges
        edge_feat:  (E, edge_dim) or None
        """
        src, dst = edge_index
        
        if edge_feat is not None:
            msg_input = torch.cat([h[src], h[dst], edge_feat], dim=-1)
        else:
            msg_input = torch.cat([h[src], h[dst]], dim=-1)
        
        messages = self.msg_mlp(msg_input)  # (E, hidden_dim)
        
        # Aggregate messages to destination nodes
        N = h.shape[0]
        agg = torch.zeros(N, messages.shape[1], device=h.device)
        agg.index_add_(0, dst, messages)
        
        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))
        h_new = h_new + self.residual(h)
        return h_new
```

### Backward compatibility

When edge_dim=0 and edge_feat=None, this behaves identically to the
current RateMessagePassing. All existing experiments continue to work
unchanged.

## Updated Model: FlexibleConditionalGNNRateMatrixPredictor

Add edge_dim parameter:

```python
class FlexibleConditionalGNNRateMatrixPredictor(nn.Module):
    def __init__(self, context_dim=1, hidden_dim=64, n_layers=4,
                 edge_dim=0):
        super().__init__()
        self.edge_dim = edge_dim
        
        # Input projection: mu(a) + tau + context
        self.input_proj = nn.Linear(1 + 1 + context_dim, hidden_dim)
        
        # Message passing layers with edge features
        self.mp_layers = nn.ModuleList([
            EdgeAwareMessagePassing(hidden_dim, hidden_dim, edge_dim)
            for _ in range(n_layers)
        ])
        
        # Edge readout (unchanged)
        self.edge_readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
    
    def forward_single(self, mu, tau, context, edge_index, edge_feat=None):
        """
        mu:         (N,) distribution
        tau:        (1,) flow time
        context:    (N, context_dim) per-node context
        edge_index: (2, E)
        edge_feat:  (E, edge_dim) or None
        """
        N = mu.shape[0]
        tau_expand = tau.expand(N, 1)
        x = torch.cat([mu.unsqueeze(-1), tau_expand, context], dim=-1)
        h = self.input_proj(x)
        
        for mp in self.mp_layers:
            h = mp(h, edge_index, edge_feat)
        
        # Edge readout
        src, dst = edge_index
        edge_h = torch.cat([h[src], h[dst]], dim=-1)
        rates = self.edge_readout(edge_h).squeeze(-1)
        
        # Build rate matrix
        R = torch.zeros(N, N, device=mu.device)
        R[src, dst] = rates
        R.diagonal().copy_(-R.sum(dim=1))
        
        return R
```

## Edge Feature Construction for Ex19

```python
def build_edge_features(R, edge_index):
    """
    Build per-edge features from the rate matrix.
    
    For each directed edge (a, b) in edge_index:
        feature = [R_ab]
    
    Just the raw rate. The GNN can compute ratios, differences,
    etc. through message passing.
    
    Args:
        R: (N, N) rate matrix
        edge_index: (2, E) LongTensor
    
    Returns:
        edge_feat: (E, 1) FloatTensor
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    feats = R[src, dst].astype(np.float32)
    return torch.tensor(feats[:, None], dtype=torch.float32)
```

## Updated Ex19 Dataset

```python
class StationaryDistDataset(torch.utils.data.Dataset):
    """
    Now includes edge features, no per-node context.
    
    Each graph instance provides: (name, R_asym, R_base, positions)
    - R_asym: asymmetric rate matrix (for computing pi and edge features)
    - R_base: symmetric base graph (for OT transport)
    
    Returns 7-tuple:
        (mu_tau, tau, node_context, R_target, edge_index, edge_feat, N)
    
    node_context is empty (context_dim=0) — a (N, 0) tensor.
    edge_feat is (E, 1) — the rate R_ab per directed edge.
    """
    def __init__(self, graph_instances, mode='dirichlet',
                 dirichlet_alpha=1.0, n_starts_per_graph=5,
                 transport_graph='undirected',
                 n_samples=20000, seed=42):
        
        rng = np.random.default_rng(seed)
        all_items = []
        self.all_pairs = []
        
        for name, R_asym, R_base, pos in graph_instances:
            N = R_asym.shape[0]
            pi = compute_stationary(R_asym)
            
            # OT transport on undirected or directed graph
            if transport_graph == 'undirected':
                R_transport = R_base
            else:
                R_transport = R_asym
            
            graph_struct = GraphStructure(R_transport)
            cost = compute_cost_matrix(graph_struct)
            geo_cache = GeodesicCache(graph_struct)
            
            # Edge index from base graph (has all edges in both directions)
            edge_index = rate_matrix_to_edge_index(R_base)
            
            # Edge features from asymmetric R
            edge_feat = build_edge_features(R_asym, edge_index)
            
            # Empty node context (context_dim=0)
            node_ctx = np.zeros((N, 0), dtype=np.float32)
            
            self.all_pairs.append({
                'name': name, 'N': N,
                'R_asym': R_asym, 'R_base': R_base,
                'positions': pos, 'edge_index': edge_index,
                'edge_feat': edge_feat, 'pi': pi,
            })
            
            # Dirichlet starts
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
                        edge_index,
                        edge_feat,
                        N,
                    ))
        
        idx = rng.permutation(len(all_items))
        self.samples = [all_items[i] for i in idx[:n_samples]]
```

## Updated DirectGNN Baseline

The DirectGNNPredictor also needs edge features for a fair comparison.
Same edge-aware message passing, same edge features, different training
objective.

```python
class EdgeAwareDirectGNNPredictor(nn.Module):
    """
    Direct prediction of target distribution using edge-aware GNN.
    No flow matching — single forward pass from edge features to output.
    
    Input per node: constant (no context needed)
    Edge features: R_ab per directed edge
    Output: softmax distribution (predicted pi)
    """
    def __init__(self, hidden_dim=64, n_layers=6, edge_dim=1):
        super().__init__()
        self.input_proj = nn.Linear(1, hidden_dim)  # just a learnable bias per node
        
        self.mp_layers = nn.ModuleList([
            EdgeAwareMessagePassing(hidden_dim, hidden_dim, edge_dim)
            for _ in range(n_layers)
        ])
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, edge_index, edge_feat, N):
        """
        edge_index: (2, E)
        edge_feat:  (E, edge_dim)
        N:          number of nodes
        
        Returns: (N,) softmax distribution
        """
        # Initialize all nodes with constant input
        h = torch.ones(N, 1, device=edge_feat.device)
        h = self.input_proj(h)
        
        for mp in self.mp_layers:
            h = mp(h, edge_index, edge_feat)
        
        logits = self.readout(h).squeeze(-1)  # (N,)
        return torch.softmax(logits, dim=0)
```

Training:
```python
def train_edge_aware_direct_gnn(model, graph_instances, n_epochs=1000,
                                 lr=5e-4, device='cpu', ema_decay=0.999):
    """Train EdgeAwareDirectGNNPredictor on (graph, pi) pairs."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ema = EMA(model, decay=ema_decay)
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        idx = np.random.permutation(len(graph_instances))
        
        for i in idx:
            name, R, R_base, pos = graph_instances[i]
            pi = compute_stationary(R)
            edge_index = rate_matrix_to_edge_index(R_base)
            edge_feat = build_edge_features(R, edge_index)
            N = R.shape[0]
            
            pi_t = torch.tensor(pi, dtype=torch.float32, device=device)
            ei = edge_index.to(device)
            ef = edge_feat.to(device)
            
            pi_pred = model(ei, ef, N)
            loss = (pi_t * (pi_t.clamp(min=1e-10).log()
                           - pi_pred.clamp(min=1e-10).log())).sum()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 50 == 0:
            print(f"  DirectGNN Epoch {epoch+1}/{n_epochs} | "
                  f"Loss: {epoch_loss/len(graph_instances):.6f}")
    
    ema.apply(model)
```

Both models see identical information:
- FM model: edge features R_ab, no node context, flow matching training
- DirectGNN: edge features R_ab, no node context, direct KL training

Any performance difference is purely due to the training formulation.

## Updated Ex19 FM Model

```python
model = FlexibleConditionalGNNRateMatrixPredictor(
    context_dim=0,        # no per-node context
    hidden_dim=64,
    n_layers=6,
    edge_dim=1,           # R_ab per edge
)
```

Per-node input is just [mu(a), tau]. All graph structure information
comes from edge features through message passing.

```python
parser.add_argument('--transport-graph', type=str, default='undirected',
                    choices=['directed', 'undirected'],
                    help='Use directed or undirected graph for OT transport')
parser.add_argument('--edge-dim', type=int, default=1,
                    help='Dimension of edge features (0 to disable)')
```

## Training Loop Update

The train_flexible_conditional function needs to handle the extra
edge_feat tensor in the dataset tuples. Two approaches:

### Option A: Extend the 6-tuple to 7-tuple

Dataset returns:
    (mu_tau, tau, node_context, R_target, edge_index, edge_feat, N)

Training loop unpacks accordingly. When edge_feat is None or edge_dim=0,
the 7th element is a dummy tensor.

### Option B: Pack edge_feat into the edge_index

Store edge features alongside edge_index as a tuple:
    edge_info = (edge_index, edge_feat)

The training loop checks if edge_info is a tuple and unpacks.

Option A is cleaner. Update train_flexible_conditional to accept
7-tuples when edge features are present.

## Backward Compatibility

All existing experiments use edge_dim=0 and 6-tuple datasets.
The updated code must handle both:

```python
for sample in batch:
    if len(sample) == 7:
        mu, tau, ctx, R_target, ei, ef, N = sample
    else:
        mu, tau, ctx, R_target, ei, N = sample
        ef = None
```

## Apply To

- meta_fm/model.py: Add EdgeAwareMessagePassing, update
  FlexibleConditionalGNNRateMatrixPredictor with edge_dim param
- meta_fm/train.py: Update train_flexible_conditional to handle
  7-tuple datasets with edge features
- experiments/ex19_stationary.py: Use edge features + undirected
  transport + minimal node context

## Expected Impact on Ex19

With edge features:
- The model sees the raw asymmetric rates per edge
- No information loss from node-level summarization
- The GNN can compute per-node stationary probability by
  aggregating directional flow information through message passing

With undirected transport:
- OT coupling uses symmetric costs (hop distance)
- Flow matching targets are clean symmetric transport
- The model learns to rearrange mass efficiently toward pi

Combined, these should substantially improve FM performance on Ex19,
closing the gap with DirectGNN.
