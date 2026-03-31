# Experiment 10: `ex10_topology_generalization.py`

## Motivation

All experiments so far use a single fixed graph. The GNN architecture naturally
processes any graph via message passing, so in principle it should generalize
across topologies. This experiment tests whether a model trained on a mix of
graph topologies can perform conditional source recovery on unseen graphs.

This is the key experiment for the "foundation model for distributions on graphs"
vision: if topology generalization works, the framework applies to any graph
without retraining.

## Setup

### Training graphs

Train on a diverse mix of small graph topologies:

- **Grid graphs**: 3x3, 4x4, 5x5 (9, 16, 25 nodes)
- **Cycle graphs**: N=8, N=10, N=12
- **Path graphs**: N=8, N=10 (linear chain, no cycles)
- **Star graphs**: N=7, N=9 (one central hub connected to all others)
- **Complete bipartite**: K_{3,3}, K_{4,4} (6, 8 nodes)

Total: ~13 different graph topologies, varying in size (6-25 nodes),
connectivity (sparse paths to dense complete bipartite), and structure
(regular grids, trees, cycles).

All graphs unweighted (R_ij = 1 for edges).

### Task

Same as Experiment 8.2: conditional source recovery of multi-peak distributions
from diffused observations. For each training sample:

1. Sample a graph from the training set.
2. Generate a source distribution with 1-3 peaks on that graph.
3. Sample tau_diff in [0.3, 1.5].
4. Diffuse to get observation.
5. Compute exact OT flow from observation to source.
6. Sample along the flow to get (mu_tau, tau, context, R_target).

### Test graphs (held out — never seen during training)

- **Grid graph**: 3x5 (15 nodes) — rectangular, not square like any training grid
- **Cycle graph**: N=15 — larger than any training cycle
- **Barbell graph**: two K_4 cliques connected by a path of length 3 (11 nodes) —
  qualitatively different topology from anything in training
- **Petersen graph**: classic 10-node graph with unusual structure

These test graphs differ from training graphs in size and/or topology.

### Handling variable graph sizes

The GNN naturally handles variable sizes via message passing. The key
implementation details:

**Edge readout**: The model predicts one rate per edge, not an N×N matrix.
This is already how GNNRateMatrixPredictor works — edges come from edge_index,
which varies per graph. The rate matrix is assembled from edge predictions.

**Batching**: Each training sample includes its own edge_index. Use PyG's
Batch.from_data_list() which handles variable-size graphs natively by
offsetting node indices.

**Node features**: [mu(a), t, mu_obs(a), tau_diff] — 4 features per node,
same regardless of graph size.

### Architecture change

The ConditionalGNNRateMatrixPredictor currently stores a fixed edge_index
as a buffer. For topology generalization, the edge_index must be passed
per-sample instead:

```python
class FlexibleConditionalGNNRateMatrixPredictor(nn.Module):
    """
    Like ConditionalGNNRateMatrixPredictor but accepts edge_index as input
    rather than storing it as a fixed buffer. Enables varying graph topology
    per forward pass.
    
    Constructor args:
        context_dim: int = 2
        hidden_dim: int = 64
        n_layers: int = 4
        (no edge_index or n_nodes — these vary per input)
    
    forward(mu, t, context, edge_index, n_nodes):
        mu: (batch_total_nodes,) — concatenated node features across batch
        t: (batch_total_nodes, 1) — time per node (same within each graph)
        context: (batch_total_nodes, context_dim) — per-node conditioning
        edge_index: (2, batch_total_edges) — batched edge indices (PyG format)
        n_nodes: list[int] — number of nodes per graph in the batch
        
        returns: list of (N_i, N_i) rate matrices, one per graph in batch
    
    The forward pass:
        1. Concatenate [mu, t, context] as node features.
        2. Run message passing on the batched graph.
        3. Compute edge rates via edge readout MLP.
        4. Unbatch and assemble per-graph rate matrices.
    """
```

Alternative simpler approach: since graphs are small, process each sample
individually (no batching across different topologies). Loop over the batch:

```python
def forward_single(self, mu, t, context, edge_index, n_nodes):
    """Process one graph at a time. Simpler, fine for small graphs."""
    # mu: (N,), t: (1,), context: (N, context_dim), edge_index: (2, E)
    # Returns: (N, N) rate matrix
```

The loop approach avoids the complexity of mixed-size batching. For small
graphs (N <= 25) the overhead is negligible.

## Dataset

```python
class TopologyGeneralizationDataset(torch.utils.data.Dataset):
    """
    Generates training data across multiple graph topologies.
    
    Each sample includes the graph structure (edge_index, n_nodes) alongside
    the usual (mu_tau, tau, context, R_target).
    
    Constructor args:
        graphs: list of (GraphStructure, edge_index) pairs
        n_samples_per_graph: int — samples to generate per graph topology
        tau_diff_range: tuple = (0.3, 1.5)
        seed: int = 42
    
    For each sample:
        1. Pick a random graph from the list.
        2. Generate multi-peak source (1-3 peaks) on that graph.
        3. Sample tau_diff, diffuse.
        4. Compute exact OT flow.
        5. Sample (mu_tau, tau, R_target) along the flow.
        6. Build context = [mu_obs, tau_diff broadcast].
        7. Store (mu_tau, tau, context, R_target, edge_index, n_nodes).
    
    __getitem__ returns:
        mu: (N,) — padded to max_N or variable length
        tau: (1,)
        context: (N, 2)
        R_target: (N, N)
        edge_index: (2, E)
        n_nodes: int
    """
```

### Padding vs variable length

Two options for handling variable graph sizes in a dataset:

**Option A: No padding, custom collate.** Each sample has its true size.
Use a custom collate function that returns lists instead of stacked tensors.
Process each graph individually in the training loop.

**Option B: Pad to max size.** Pad all distributions, rate matrices, and
contexts to max_N = 25. Mask out padded nodes. Simpler batching but wastes
compute on padding.

Recommend Option A for cleanliness. The graphs are small enough that
per-sample processing is fast.

## Training

- FlexibleConditionalGNNRateMatrixPredictor with context_dim=2
- hidden_dim=64, n_layers=4 (same as previous experiments)
- 1000 epochs (longer since more diverse training data)
- Adam lr=1e-3, batch_size=1 (process graphs individually) or use custom
  collate with batch_size=32
- 1000 samples per graph topology × 13 topologies = 13000 total samples
- 1/(1-t) factorization applied

## Evaluation

For each test graph:
1. Generate 20 test cases (multi-peak sources, tau_diff=0.8).
2. Run the trained model (never seen this graph topology).
3. Compute TV distance to true source.
4. Compute peak recovery (top-k).
5. Compare to exact inverse baseline.

Also evaluate on a held-out sample from each TRAINING graph topology (to
measure in-distribution performance for comparison).

## Plots (single figure, 2x3 grid)

- **Panel A**: Training loss curve.

- **Panel B**: TV distance by graph topology. Grouped bar chart with two
  groups: training topologies (in-distribution) and test topologies
  (out-of-distribution). Each bar is the mean TV for that topology.
  Color-code training vs test. Shows whether OOD topologies have
  comparable TV to training topologies.

- **Panel C**: Example recoveries on test graphs. For each of the 4 test
  topologies, show the observation and recovered distribution side by side.
  Use networkx to draw the graph with node color/size proportional to mass.
  This is the visual highlight — seeing the model work on graphs it's
  never seen before.

- **Panel D**: Peak recovery (top-k) by topology, training vs test.
  Same grouping as Panel B.

- **Panel E**: TV vs graph size (number of nodes). Scatter plot with one
  point per topology, colored by train/test. Shows whether larger graphs
  are harder (expected) and whether the model extrapolates to larger sizes.

- **Panel F**: Example recovery trajectory on the barbell graph (the most
  exotic test topology). Show the graph at t=0, 0.5, 1.0 with node colors
  representing the distribution. Mass should flow from a diffused state
  to peaks, respecting the bottleneck between the two cliques.

## Validation checks (print to console)

```
=== Experiment 10: Topology Generalization Results ===

Training topologies (in-distribution):
  grid_3x3:    TV=X.XX, peak_recovery=XX%
  grid_4x4:    TV=X.XX, peak_recovery=XX%
  grid_5x5:    TV=X.XX, peak_recovery=XX%
  cycle_8:     TV=X.XX, peak_recovery=XX%
  cycle_10:    TV=X.XX, peak_recovery=XX%
  cycle_12:    TV=X.XX, peak_recovery=XX%
  ...

Test topologies (out-of-distribution):
  grid_3x5:    TV=X.XX, peak_recovery=XX%
  cycle_15:    TV=X.XX, peak_recovery=XX%
  barbell:     TV=X.XX, peak_recovery=XX%
  petersen:    TV=X.XX, peak_recovery=XX%

Mean TV (training): X.XX
Mean TV (test):     X.XX
Ratio:              X.XX (want close to 1.0)
```

## Expected Outcome

**Optimistic**: The GNN learns general principles (mass flows along edges,
concentrates at specified locations, respects bottlenecks) that transfer to
unseen topologies. Test TV is within 2x of training TV.

**Realistic**: Good transfer to similar topologies (grid_3x5 from training
on grids, cycle_15 from training on cycles) but degraded performance on
qualitatively different structures (barbell, Petersen). The bottleneck in
the barbell graph is particularly challenging since no training graph has
this feature.

**Pessimistic**: The model overfits to training topologies and fails on all
test graphs. This would indicate that the GNN doesn't extract transferable
principles and each graph needs its own model.

Any of these outcomes is informative. Even the pessimistic result tells us
something about the limits of GNN generalization for this task.

## Graph Construction Helpers

Add to `graph_ot_fm/utils.py`:

```python
def make_path_graph(n: int) -> np.ndarray:
    """Linear chain: 0-1-2-...(n-1)."""

def make_star_graph(n: int) -> np.ndarray:
    """Central node 0 connected to nodes 1..n-1."""

def make_complete_bipartite_graph(n1: int, n2: int) -> np.ndarray:
    """K_{n1,n2}: every node in set A connected to every node in set B."""

def make_barbell_graph(clique_size: int, path_length: int) -> np.ndarray:
    """Two complete graphs of clique_size connected by a path of path_length."""

def make_petersen_graph() -> np.ndarray:
    """The Petersen graph (10 nodes, 15 edges)."""
```

## Dependencies

No new dependencies. NetworkX can be used for graph construction convenience
but all we need are the adjacency/rate matrices.
