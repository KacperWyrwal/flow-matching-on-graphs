# Spec: Fix Dynamic Edge Features and k=4 Indexing

## Problem 1: Edge-labeled configurations need dynamic edge features

The ConfigurationSpace API has:
- `node_features(config)` — dynamic, depends on current configuration
- `position_edge_features()` — static, depends only on the graph structure

For node-labeled problems (Johnson, Kawasaki), the configuration enters
through `node_features` — each node's label is a feature. The position
graph edges and their features are static.

For edge-labeled problems (degree-preserving graph generation), the
configuration IS the edge labels — which edges exist. The model needs
to see $A_{ij}$ for every pair $(i,j)$ as an edge feature. This changes
with the configuration and cannot be static.

### Fix: Add `edge_features(config)` to ConfigurationSpace

```python
class ConfigurationSpace(ABC):
    
    @abstractmethod
    def position_edge_features(self) -> np.ndarray | None:
        """STATIC edge features of the position graph.
        Shape: (n_edges, d_static_edge). Does not depend on config.
        E.g., J[i,j] coupling strengths, lattice bond type."""
        pass
    
    def dynamic_edge_features(self, config) -> np.ndarray | None:
        """DYNAMIC edge features depending on current configuration.
        Shape: (n_edges, d_dynamic_edge). Default: None.
        Override for edge-labeled problems."""
        return None
```

The model receives the concatenation of static and dynamic edge features.
The training loop and inference code call both methods:

```python
static_ef = space.position_edge_features()       # computed once
dynamic_ef = space.dynamic_edge_features(config)  # computed per sample

if static_ef is not None and dynamic_ef is not None:
    edge_features = np.concatenate([static_ef, dynamic_ef], axis=-1)
elif static_ef is not None:
    edge_features = static_ef
elif dynamic_ef is not None:
    edge_features = dynamic_ef
else:
    edge_features = None
```

### Implementation per space

**JohnsonSpace:** `dynamic_edge_features` returns None (default).
Configuration is in node features. No change needed.

**KawasakiSpace:** `dynamic_edge_features` returns None (default).
Configuration is in node features. No change needed.

**DFMSpace:** `dynamic_edge_features` returns None (default).
Configuration is in node features. No change needed.

**DegreeSequenceSpace:**

```python
def dynamic_edge_features(self, config):
    """Current edge existence as dynamic edge feature.
    
    The position graph is K_n, so edges are all pairs (i,j).
    For each pair, return [A_ij] indicating current edge existence.
    """
    A = self._config_to_adj(config)
    edges = self.position_graph_edges()  # (2, n*(n-1))
    n_edges = edges.shape[1]
    feats = np.zeros((n_edges, 1), dtype=np.float32)
    for e in range(n_edges):
        i, j = edges[0, e], edges[1, e]
        feats[e, 0] = A[i, j]
    return feats
```

This gives the model the current adjacency as attention biases. Combined
with node features (current degree), the model has full information about
the current graph.

### Model change

Update `ConfigurationRatePredictor.forward` and the training/inference
loops to handle dynamic edge features:

```python
# In training loop:
static_ef = space.position_edge_features()  # once
for each sample:
    dynamic_ef = space.dynamic_edge_features(config_t)
    edge_features = concatenate(static_ef, dynamic_ef)  # or just dynamic if no static
    
# In model forward:
# edge_feature_dim = d_static + d_dynamic
# The backbone receives the concatenated features
```

For DegreeSequenceSpace: `edge_feature_dim = 1` (just A_ij, no static
features). The model constructor should be:

```python
fm_model = ConfigurationRatePredictor(
    node_feature_dim=1,
    edge_feature_dim=1,   # dynamic A_ij
    global_dim=2,
    hidden_dim=128,
    n_layers=4,
    transition_order=4,
)
```

## Problem 2: k=4 transition indexing

For k=2 (Johnson, Kawasaki), the model outputs an (n, n) rate matrix.
Transition index is a flat index into this matrix: `idx = i * n + j`
decodes to swap positions (i, j).

For k=4, a transition is a double edge swap involving 4 nodes. The model
needs to score all valid double swaps and the training loop needs to
provide target rates in a matching format. We need a consistent indexing
scheme.

### Design: index double swaps by pairs of edges

A double edge swap removes two existing edges and adds two new edges.
From the model's perspective, the "action" is selecting:
- One existing edge to remove: $(a, b)$ where $A_{ab} = 1$
- One non-edge to add: $(c, d)$ where $A_{cd} = 0$

Wait — a double swap removes TWO edges and adds TWO edges. But the two
removals and two additions are coupled (they must preserve degree). The
constraint is: the four nodes $\{a, b, c, d\}$ must be distinct, and the
swap replaces $\{(a,b), (c,d)\}$ with either $\{(a,c), (b,d)\}$ or
$\{(a,d), (b,c)\}$.

### Enumeration approach

Rather than trying to output a full (n_edges × n_edges) rate matrix,
enumerate valid double swaps explicitly and output a rate for each:

```python
class DegreeSequenceSpace(ConfigurationSpace):
    
    def enumerate_valid_swaps(self, config):
        """Enumerate all valid double edge swaps from current config.
        
        Returns:
            swaps: list of tuples (a, b, c, d, rewiring)
                where (a,b) and (c,d) are removed,
                and rewiring indicates which new edges are added:
                'ac_bd' -> add (a,c) and (b,d)
                'ad_bc' -> add (a,d) and (b,c)
            n_swaps: number of valid swaps
        """
        A = self._config_to_adj(config)
        edges = [(i, j) for i in range(self.n)
                 for j in range(i+1, self.n) if A[i,j] == 1]
        
        swaps = []
        for idx1 in range(len(edges)):
            for idx2 in range(idx1 + 1, len(edges)):
                a, b = edges[idx1]
                c, d = edges[idx2]
                # All 4 nodes must be distinct
                if len({a, b, c, d}) < 4:
                    continue
                # Try rewiring 1: (a,b),(c,d) -> (a,c),(b,d)
                if A[a, c] == 0 and A[b, d] == 0:
                    swaps.append((a, b, c, d, 'ac_bd'))
                # Try rewiring 2: (a,b),(c,d) -> (a,d),(b,c)
                if A[a, d] == 0 and A[b, c] == 0:
                    swaps.append((a, b, c, d, 'ad_bc'))
        
        return swaps
```

### Model output: score each valid swap

The model needs to output one rate per valid swap. Two approaches:

**Approach A: Explicit enumeration + MLP scoring.** Enumerate valid
swaps, compute a score for each using node embeddings:

```python
def score_swaps(self, node_embeddings, valid_swaps):
    """Score each valid double swap.
    
    For swap (a, b, c, d, rewiring):
    - Remove pair: embeddings of a, b
    - Add pair: embeddings of the new edge endpoints
    
    Score = f(h_a, h_b, h_c, h_d, rewiring)
    """
    scores = []
    for (a, b, c, d, rewiring) in valid_swaps:
        if rewiring == 'ac_bd':
            # Remove (a,b), (c,d); add (a,c), (b,d)
            score = self.swap_mlp(torch.cat([
                node_embeddings[a], node_embeddings[b],
                node_embeddings[c], node_embeddings[d]
            ]))
        else:
            # Remove (a,b), (c,d); add (a,d), (b,c)
            score = self.swap_mlp(torch.cat([
                node_embeddings[a], node_embeddings[b],
                node_embeddings[d], node_embeddings[c]  # reorder
            ]))
        scores.append(score)
    return torch.stack(scores)  # (n_swaps,)
```

**Approach B: Pair-of-pairs attention.** Compute edge embeddings for
all existing edges and all non-edges, then use attention to score
(existing edge, non-edge) pairs:

```python
def score_swaps_attention(self, node_embeddings, A):
    """Score swaps via attention over edge pairs.
    
    1. Compute edge embeddings for existing edges (removal candidates)
    2. Compute edge embeddings for non-edges (addition candidates)  
    3. Attention: each existing-edge queries each non-edge
    4. Score = sum of two (removal, addition) attention scores
    """
    # Edge embeddings
    exist_edges = [(i,j) for i,j in ... if A[i,j] == 1]
    non_edges = [(i,j) for i,j in ... if A[i,j] == 0]
    
    exist_emb = [MLP([h_i, h_j]) for (i,j) in exist_edges]  # (m, d)
    non_emb = [MLP([h_i, h_j]) for (i,j) in non_edges]      # (m', d)
    
    # Pairwise scores: which removal pairs with which addition
    Q = W_Q(exist_emb)  # (m, d)
    K = W_K(non_emb)     # (m', d)
    pair_scores = Q @ K.T  # (m, m')
    
    # A double swap needs TWO removals and TWO additions
    # Score = pair_score[r1, a1] + pair_score[r2, a2]
    # This is expensive to enumerate...
```

**Recommendation: Use Approach A for correctness, optimize later.**

Approach A is simpler, guaranteed correct, and the number of valid
swaps is manageable for n=20 (at most a few thousand). The MLP sees
all 4 node embeddings and can learn arbitrary scoring functions.

Approach B is more elegant but harder to implement correctly for the
coupled 4-body constraint. Save it for optimization.

### Integration with training loop

The training loop needs to handle variable-length swap lists. The
target rates are a vector over valid swaps, not a fixed-size matrix.

```python
# In ConfigurationSpace:
def compute_target_rates(self, config_0, config_T, config_t, t):
    """Returns (valid_swaps, target_rates) instead of a fixed matrix."""
    valid_swaps = self.enumerate_valid_swaps(config_t)
    # ... compute which swaps are geodesic-progressing
    # ... assign rate 1/d_rem to those, 0 to others
    rates = np.zeros(len(valid_swaps), dtype=np.float32)
    for idx, swap in enumerate(valid_swaps):
        if is_geodesic_progressing(swap, config_t, config_T):
            rates[idx] = 1.0 / d_rem
    return valid_swaps, rates

# In model forward:
def forward(self, node_features, edge_index, edge_features,
            global_features, valid_swaps):
    """
    valid_swaps: list of (a, b, c, d, rewiring) tuples
    Returns: (n_swaps,) rate predictions
    """
    h = self.backbone(node_features, edge_index, edge_features,
                      global_features)
    rates = self.head.score_swaps(h, valid_swaps)
    return rates
```

### apply_transition for k=4

```python
def apply_transition(self, config, swap_idx, valid_swaps):
    """Apply the swap at index swap_idx in valid_swaps list."""
    a, b, c, d, rewiring = valid_swaps[swap_idx]
    A = self._config_to_adj(config)
    A[a, b] = A[b, a] = 0
    A[c, d] = A[d, c] = 0
    if rewiring == 'ac_bd':
        A[a, c] = A[c, a] = 1
        A[b, d] = A[d, b] = 1
    else:
        A[a, d] = A[d, a] = 1
        A[b, c] = A[c, b] = 1
    return self._adj_to_config(A)
```

### Inference loop adaptation

```python
# In generate_samples, for k=4:
valid_swaps = space.enumerate_valid_swaps(config)
rates = model(node_feat, edge_idx, edge_feat, global_feat, valid_swaps)
rates = rates / (1 - t + 1e-10)

total_rate = rates.sum()
if total_rate > 0:
    n_events = rng.poisson(total_rate * dt)
    if n_events > 0:
        probs = rates / total_rate
        swap_idx = rng.choice(len(valid_swaps), p=probs)
        config = space.apply_transition(config, swap_idx, valid_swaps)
```

## Problem 3: API changes to ConfigurationSpace

The k=4 changes break the current API where `transition_mask` returns a
fixed-shape array and `apply_transition` takes a flat index. We need a
more flexible interface.

### Updated API

```python
class ConfigurationSpace(ABC):
    
    # EXISTING (unchanged for k=1, k=2):
    def transition_mask(self, config) -> np.ndarray:
        """Fixed-shape mask for k=1 and k=2 transitions."""
        ...
    
    def apply_transition(self, config, transition_idx) -> np.ndarray:
        """Apply transition by flat index for k=1 and k=2."""
        ...
    
    # NEW (for k=4 and general k):
    def enumerate_transitions(self, config) -> list:
        """Enumerate valid transitions as structured objects.
        
        For k=2: list of (i, j) pairs (equivalent to mask nonzeros)
        For k=4: list of (a, b, c, d, rewiring) tuples
        For general k: list of transition descriptors
        """
        ...
    
    def apply_transition_by_descriptor(self, config, descriptor):
        """Apply a transition given its structured descriptor."""
        ...
    
    def compute_target_rates_enumerated(self, config_0, config_T,
                                         config_t, t):
        """Returns (transitions, rates) where transitions is from
        enumerate_transitions and rates is a parallel array."""
        ...
```

For k=1 and k=2, the existing mask-based API still works (it's more
efficient for dense outputs). For k=4+, use the enumeration-based API.
The training loop checks `transition_order` and uses the appropriate
path:

```python
if space.transition_order <= 2:
    # Use mask-based API (existing code)
    mask = space.transition_mask(config_t)
    target_rates = space.compute_target_rates(...)
    pred_rates = model(node_feat, edge_idx, edge_feat, global_feat, mask)
    loss = rate_kl_loss(pred_rates, target_rates, mask)
else:
    # Use enumeration-based API
    transitions = space.enumerate_transitions(config_t)
    target_transitions, target_rates = space.compute_target_rates_enumerated(...)
    pred_rates = model.score_transitions(h, transitions)
    loss = rate_kl_loss_enumerated(pred_rates, target_rates)
```

### Model head for k=4

```python
class SwapScoringHead(nn.Module):
    """Score double edge swaps from node embeddings.
    
    Input: node embeddings (n, d), list of valid swaps
    Output: (n_swaps,) rates
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.swap_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, h, valid_swaps):
        """
        h: (batch, n, d) node embeddings
        valid_swaps: list of lists of (a, b, c, d, rewiring)
            (one list per batch element)
        Returns: list of (n_swaps_b,) rate tensors
        """
        all_rates = []
        for b in range(h.shape[0]):
            if not valid_swaps[b]:
                all_rates.append(torch.zeros(0, device=h.device))
                continue
            
            swap_feats = []
            for (a, b_node, c, d, rewiring) in valid_swaps[b]:
                if rewiring == 'ac_bd':
                    feat = torch.cat([h[b, a], h[b, b_node],
                                      h[b, c], h[b, d]])
                else:
                    feat = torch.cat([h[b, a], h[b, b_node],
                                      h[b, d], h[b, c]])
                swap_feats.append(feat)
            
            swap_feats = torch.stack(swap_feats)  # (n_swaps, 4*d)
            rates = self.swap_mlp(swap_feats).squeeze(-1)  # (n_swaps,)
            rates = torch.nn.functional.softplus(rates)
            all_rates.append(rates)
        
        return all_rates
```

## Implementation order

1. Add `dynamic_edge_features(config)` to ConfigurationSpace base class
   (default returns None)

2. Implement `dynamic_edge_features` in DegreeSequenceSpace (returns A_ij)

3. Update training loop and inference to call `dynamic_edge_features`
   and concatenate with static features

4. Add `enumerate_transitions` and `apply_transition_by_descriptor`
   to ConfigurationSpace base class

5. Implement both for DegreeSequenceSpace

6. Add `SwapScoringHead` to models/heads.py

7. Update ConfigurationRatePredictor to use SwapScoringHead when
   transition_order >= 4

8. Update training loop to branch on transition_order for mask-based
   vs enumeration-based API

9. Update inference loop similarly

10. Verify on Ex20 (k=2, should be unchanged) before running Ex22

## Backward compatibility

The k=1 and k=2 code paths are completely unchanged. The new methods
have defaults in the base class:
- `dynamic_edge_features` returns None
- `enumerate_transitions` can default to converting from transition_mask
- `apply_transition_by_descriptor` can default to apply_transition

Existing experiments (Ex20, Ex21) should produce identical results
after this change.
