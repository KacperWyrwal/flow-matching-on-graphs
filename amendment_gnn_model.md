# Amendment: GNN-Based Rate Matrix Predictor

## Motivation

Experiment 4 revealed that the MLP-based RateMatrixPredictor fails to generalize
to out-of-distribution peak locations on the cycle graph. The root cause: the MLP
treats the distribution as a flat vector with no awareness of graph structure, so
it memorizes node identities rather than learning structural patterns.

The fix: replace the MLP with a graph neural network that processes the distribution
as a signal on the graph. This makes the model automatically equivariant to graph
symmetries and allows it to generalize across structurally equivalent nodes.

## New Model: `meta_fm/model.py`

Keep the existing RateMatrixPredictor class (rename to RateMatrixPredictorMLP for
reference), and add a new GNN-based model as the default.

Uses PyTorch Geometric for message passing and batching. This scales to large
graphs out of the box and avoids hand-rolled message passing.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected


class RateMessagePassing(MessagePassing):
    """
    Custom message-passing layer for rate matrix prediction.
    
    Uses the "source_to_target" flow convention (PyG default).
    
    message(x_i, x_j):
        x_i: target node features (auto-lifted by PyG from edge_index[1])
        x_j: source node features (auto-lifted by PyG from edge_index[0])
        return MLP_msg(cat(x_i, x_j))
    
    update(aggr_out, x):
        aggr_out: aggregated messages per node (sum aggregation)
        x: current node features
        return MLP_update(cat(x, aggr_out))
    
    Constructor args:
        in_dim: int — input feature dimension
        hidden_dim: int — output feature dimension
    """
    def __init__(self, in_dim, hidden_dim):
        super().__init__(aggr='add')  # sum aggregation
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
    
    def forward(self, x, edge_index):
        # x: (num_nodes, in_dim)
        # edge_index: (2, num_edges)
        aggr = self.propagate(edge_index, x=x)
        return self.update_mlp(torch.cat([x, aggr], dim=-1))
    
    def message(self, x_i, x_j):
        return self.msg_mlp(torch.cat([x_i, x_j], dim=-1))


class GNNRateMatrixPredictor(nn.Module):
    """
    Graph neural network that predicts rate matrices from (distribution, time).
    
    Architecture:
        Input: per-node features h_a = [mu(a), t]  (2 dims per node)
        Message passing: L layers of RateMessagePassing
        Edge readout: MLP on pairs of endpoint features -> softplus -> rate
        Assembly: fill rate matrix on edges, set diagonal = -row sum
    
    Constructor args:
        edge_index: torch.LongTensor (2, num_edges) — directed edge list
            from the graph. Include both directions for undirected graphs.
            Use torch_geometric.utils.to_undirected if needed.
        n_nodes: int — number of nodes in the graph
        hidden_dim: int = 64
        n_layers: int = 4
    
    The edge_index is stored as a buffer (not a parameter).
    
    forward(mu, t):
        mu: (batch_size, N) — distributions
        t: (batch_size, 1) — times
        returns: (batch_size, N, N) — valid rate matrices
    
    Internal batching strategy:
        Convert each (mu, t) pair into a PyG Data object:
            Data(x=node_features, edge_index=edge_index)
        Use torch_geometric.data.Batch.from_data_list() to batch them.
        Run message passing on the batched graph.
        Unbatch node features, compute edge readout, assemble rate matrices.
    """
    def __init__(self, edge_index, n_nodes, hidden_dim=64, n_layers=4):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.register_buffer('edge_index', edge_index)
        
        # Message passing layers
        self.mp_layers = nn.ModuleList()
        self.mp_layers.append(RateMessagePassing(in_dim=2, hidden_dim=hidden_dim))
        for _ in range(n_layers - 1):
            self.mp_layers.append(RateMessagePassing(in_dim=hidden_dim, hidden_dim=hidden_dim))
        
        # Edge readout MLP: takes concatenated endpoint features -> scalar rate
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, mu, t):
        B, N = mu.shape
        device = mu.device
        
        # Build per-node features: [mu(a), t] for each node
        t_expanded = t.expand(B, N)                      # (B, N)
        node_features = torch.stack([mu, t_expanded], dim=-1)  # (B, N, 2)
        
        # Create PyG batch
        data_list = []
        for b in range(B):
            data_list.append(Data(x=node_features[b], edge_index=self.edge_index))
        batch = Batch.from_data_list(data_list)
        
        # Message passing
        h = batch.x  # (B*N, 2)
        for mp_layer in self.mp_layers:
            h = mp_layer(h, batch.edge_index)  # (B*N, hidden_dim)
        
        # Reshape back: (B, N, hidden_dim)
        h = h.view(B, N, self.hidden_dim)
        
        # Edge readout: predict rate for each edge
        src, dst = self.edge_index  # original (unbatched) edge indices
        h_src = h[:, src, :]  # (B, num_edges, hidden_dim)
        h_dst = h[:, dst, :]  # (B, num_edges, hidden_dim)
        edge_features = torch.cat([h_src, h_dst], dim=-1)  # (B, num_edges, 2*hidden_dim)
        edge_rates = F.softplus(self.edge_mlp(edge_features).squeeze(-1))  # (B, num_edges)
        
        # Assemble rate matrix
        rate_matrix = torch.zeros(B, N, N, device=device)
        rate_matrix[:, src, dst] = edge_rates
        
        # Set diagonal = -row sum
        rate_matrix[:, range(N), range(N)] = -rate_matrix.sum(dim=-1)
        
        return rate_matrix
```

### Helper: Building edge_index from rate matrix

```python
def rate_matrix_to_edge_index(R: np.ndarray) -> torch.LongTensor:
    """
    Convert a rate matrix to a PyG edge_index tensor.
    
    Extracts all (i, j) pairs where R[i, j] > 0 and i != j.
    Returns torch.LongTensor of shape (2, num_edges).
    
    For undirected graphs, R is symmetric off-diagonal, so both
    directions are included automatically.
    """
    rows, cols = np.where((R > 0) & (np.eye(len(R)) == 0))
    return torch.tensor(np.stack([rows, cols]), dtype=torch.long)
```

## Changes to `meta_fm/train.py`

The training loop stays exactly the same. The only change is instantiating
GNNRateMatrixPredictor instead of RateMatrixPredictor:

```python
# Old:
# model = RateMatrixPredictor(n_nodes=graph.N)

# New:
edge_index = rate_matrix_to_edge_index(graph.R)
model = GNNRateMatrixPredictor(edge_index=edge_index, n_nodes=graph.N, hidden_dim=64, n_layers=4)
```

Loss function, optimizer, batch size, learning rate — all unchanged.

## Changes to `meta_fm/sample.py`

No changes needed. The sample_trajectory and backward_trajectory functions call
model(mu, t) which has the same interface for both MLP and GNN models.

## Updated Dependencies

Add to requirements.txt:
```
torch-geometric>=2.4
```

Install with: `pip install torch-geometric`
(PyG will pull in torch-scatter, torch-sparse, etc. as needed for the
installed torch version.)

## Re-run Experiments

### Experiment 3 (re-run with GNN)

Re-run ex3_meta_level.py with GNNRateMatrixPredictor. Save results as
ex3_meta_level_gnn.png. Save model checkpoint as checkpoints/meta_model_gnn.pt.

Expected: comparable or better performance to the MLP version (the in-distribution
task should be no harder for a GNN).

### Experiment 4 (re-run with GNN)

Re-run ex4_generalization.py with GNNRateMatrixPredictor. Save results as
ex4_generalization_gnn.png.

Expected: OOD performance should now be comparable to in-distribution performance.
The GNN's equivariance means that a distribution peaked at node 3 is processed
identically (up to the graph structure) to one peaked at node 0. The entropy
curves in Panel C should now track the exact curves, and the TV bars in Panel D
should show no systematic gap between in-distribution and OOD.

### Experiment 5 (run with GNN)

Run ex5_source_localization.py using the GNN model checkpoint. This is the first
run of Ex5 (it was waiting on a model that generalizes).

Expected: successful source recovery, especially at low noise, since the GNN model
can handle arbitrary peak locations.

## Hyperparameter Notes

- hidden_dim=64 is likely sufficient for N=6. For N=16 (grid), may need 128.
- n_layers=4 gives receptive field of 4 hops, covering the full diameter of the
  6-cycle (diameter=3) and most of the 4x4 grid (diameter=6, so may want 6 layers).
- Learning rate 1e-3 with Adam, same as before.
- If training is slower to converge (GNN has fewer parameters, more structured),
  increase to 1000 epochs. Monitor loss — if still decreasing at 500, keep going.

## Dependencies

Adds one new dependency: `torch-geometric>=2.4`. Install with `pip install torch-geometric`.
No other new dependencies.
