# Optimization Spec: Efficient Graph-Level Solver

## Overview

The current implementation recomputes geodesic random walks, conditional marginals,
and rate matrices from scratch for every (i, j, t) query. Most of this computation
is redundant — the spatial structure depends only on the graph and the pair (i, j),
not on t. This spec describes a precomputation strategy that eliminates redundant
work and significantly speeds up meta-level dataset generation.

---

## 1. Precomputed Geodesic Cache

### New class: `graph_ot_fm/geodesic_cache.py`

```python
class GeodesicCache:
    """
    Precomputes and stores all time-independent geodesic structure for a graph.
    Built once per graph, reused across all distribution pairs and time queries.
    
    Constructor args:
        graph: GraphStructure
    
    Precomputes:
        spatial_weights: dict[(i, j)] -> np.ndarray of shape (d+1, N)
            spatial_weights[(i,j)][k, x] = w_k(x | i, j)
            The probability of being at node x after k steps of the
            geodesic random walk from i toward j.
            
            Only computed for pairs (i, j) where d(i,j) > 0.
            For d(i,j) == 0: trivially delta_i for all k (just k=0).
            
        branch_structure: dict[(i, j)] -> list of (a, b, weight) tuples
            The time-independent part of the conditional rate matrix.
            For each geodesic node a on a shortest path from i to j,
            for each b in N^-(a):
                weight = d(a,j) * R[a,b] * N_b / N_a
            
            So that R_t^{i->j}(a, b) = weight / (1 - t)
            
            Stored as sparse list of (a, b, weight) tuples, not full NxN matrix.
    
    Methods:
        get_spatial_weights(i, j) -> np.ndarray (d+1, N):
            Returns precomputed spatial weights. Raises KeyError if not cached.
        
        conditional_marginal(i, j, t) -> np.ndarray (N,):
            p_t(x | i, j) = sum_k Binom(k; d, t) * spatial_weights[k, x]
            Just a dot product of binomial coefficients with cached weights.
        
        conditional_rate_matrix(i, j, t) -> np.ndarray (N, N):
            Multiplies cached branch_structure by 1/(1-t).
            Returns sparse-ish NxN matrix.
        
        conditional_rate_sparse(i, j, t) -> list of (a, b, rate):
            Same as above but returns sparse list instead of dense matrix.
            More efficient when N is large and only a few geodesic edges exist.
    """
```

### Precomputation algorithm

```python
def __init__(self, graph):
    self.graph = graph
    self.spatial_weights = {}
    self.branch_structure = {}
    
    N = graph.N
    
    # Precompute for all reachable pairs (i, j)
    # Optimization: only compute for pairs where d(i,j) < inf
    # Further optimization: many pairs share the same geodesic structure
    # toward the same target j. Group by target j.
    
    for j in range(N):
        # Build geodesic transition matrix toward j (target-specific)
        # P_geo[a, b] = R[a,b] * N_b / N_a for b in N^-(a), else 0
        P_geo_j = np.zeros((N, N))
        for a in range(N):
            if graph.dist[a, j] == 0 or np.isinf(graph.dist[a, j]):
                continue
            for b in graph.closer_neighbors.get((a, j), []):
                P_geo_j[a, b] = graph.R[a, b] * graph.geodesic_count[b, j] / graph.geodesic_count[a, j]
        
        # For each source i, compute spatial weights by applying P_geo_j
        # repeatedly starting from delta_i.
        # 
        # Key optimization: if multiple sources i have the same distance d
        # to j, we can batch the matrix-vector products.
        
        for i in range(N):
            d = int(graph.dist[i, j])
            if d == 0 or np.isinf(graph.dist[i, j]):
                continue
            
            weights = np.zeros((d + 1, N))
            weights[0, i] = 1.0
            for k in range(d):
                weights[k + 1] = weights[k] @ P_geo_j
            
            self.spatial_weights[(i, j)] = weights
            
            # Branch structure: collect (a, b, weight) for all geodesic edges
            branches = []
            for a in range(N):
                da = graph.dist[a, j]
                if np.isinf(da) or da == 0:
                    continue
                # Check if a is on a geodesic from i to j
                if graph.dist[i, a] + graph.dist[a, j] != graph.dist[i, j]:
                    continue
                for b in graph.closer_neighbors.get((a, j), []):
                    w = da * graph.R[a, b] * graph.geodesic_count[b, j] / graph.geodesic_count[a, j]
                    branches.append((a, b, w))
            
            self.branch_structure[(i, j)] = branches
```

### Memory optimization

For large graphs, not all N^2 pairs need to be cached. Options:

1. **Lazy computation**: only compute and cache when first requested.
   ```python
   def get_spatial_weights(self, i, j):
       if (i, j) not in self.spatial_weights:
           self._compute_pair(i, j)
       return self.spatial_weights[(i, j)]
   ```

2. **Coupling-aware caching**: given an OT coupling pi, only precompute
   pairs (i, j) where pi(i,j) > 0. Since the coupling has at most 2N-1
   nonzero entries, this limits cache size to O(N) pairs.
   ```python
   def precompute_for_coupling(self, coupling):
       """Only cache pairs that appear in the coupling."""
       nonzero = np.argwhere(coupling > 1e-12)
       for i, j in nonzero:
           self._compute_pair(i, j)
   ```

---

## 2. Fast Marginal Rate Matrix Assembly

### Updated `graph_ot_fm/flow.py`

```python
def marginal_rate_matrix_fast(
    cache: GeodesicCache,
    coupling: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Compute u_t(a, b) using precomputed cache.
    
    Much faster than the current implementation because:
    1. Spatial weights are precomputed (no geodesic walk per query)
    2. Branch structure is precomputed (no neighbor lookups per query)
    3. Only iterates over nonzero coupling entries (sparse)
    
    Algorithm:
        1. Compute p_t(a) = sum_{i,j} pi(i,j) * cache.conditional_marginal(i,j,t)
        2. For each (i,j) with pi(i,j) > 0:
            For each (a, b, weight) in cache.branch_structure[(i,j)]:
                u_t[a,b] += pi(i,j) * p_t(a|i,j) / p_t(a) * weight / (1-t)
        3. Set diagonal.
    
    Returns: np.ndarray (N, N)
    """
    N = cache.graph.N
    inv_1mt = 1.0 / (1.0 - t)
    
    # Step 1: marginal distribution and per-pair conditionals
    p_t = np.zeros(N)
    conditionals = {}
    nonzero_pairs = np.argwhere(coupling > 1e-12)
    
    for idx in range(len(nonzero_pairs)):
        i, j = nonzero_pairs[idx]
        if i == j:
            cond = np.zeros(N)
            cond[i] = 1.0
        else:
            cond = cache.conditional_marginal(i, j, t)
        conditionals[(i, j)] = cond
        p_t += coupling[i, j] * cond
    
    # Step 2: assemble rate matrix
    u_t = np.zeros((N, N))
    
    for idx in range(len(nonzero_pairs)):
        i, j = nonzero_pairs[idx]
        if i == j:
            continue
        
        pi_ij = coupling[i, j]
        cond = conditionals[(i, j)]
        
        for a, b, weight in cache.branch_structure[(i, j)]:
            if p_t[a] > 1e-12:
                u_t[a, b] += pi_ij * (cond[a] / p_t[a]) * weight * inv_1mt
    
    # Step 3: diagonal
    np.fill_diagonal(u_t, 0)
    u_t[np.arange(N), np.arange(N)] = -u_t.sum(axis=1)
    
    return u_t
```

### Vectorized conditional marginal

```python
def conditional_marginal(self, i, j, t):
    """Compute p_t(x | i, j) from cached spatial weights."""
    if i == j:
        out = np.zeros(self.graph.N)
        out[i] = 1.0
        return out
    
    d = int(self.graph.dist[i, j])
    weights = self.spatial_weights[(i, j)]  # (d+1, N)
    
    # Binomial coefficients: shape (d+1,)
    from scipy.stats import binom as binom_dist
    binom_pmf = binom_dist.pmf(np.arange(d + 1), d, t)
    
    # Dot product: sum_k binom_pmf[k] * weights[k, :]
    return binom_pmf @ weights  # (N,)
```

The `binom_pmf` computation can also be cached if we query the same t repeatedly
(e.g., when assembling the marginal rate matrix, all pairs use the same t).

---

## 3. Sinkhorn for Meta-OT Coupling

### Updated `graph_ot_fm/ot_solver.py`

```python
def compute_ot_coupling_sinkhorn(
    mu0: np.ndarray,
    mu1: np.ndarray,
    cost: np.ndarray,
    reg: float = 0.05,
) -> np.ndarray:
    """
    Entropic OT coupling via Sinkhorn algorithm.
    
    Much faster than exact LP for the meta-level coupling where:
    - We compute W(mu_s, nu_t) for all M x M pairs of distributions
    - Slight suboptimality in the coupling is fine
    - We may want gradients later (for inverse problems)
    
    Use: ot.sinkhorn(mu0, mu1, cost, reg)
    
    Args:
        reg: entropic regularization. Smaller = closer to exact OT but slower.
             0.05 is a good default for normalized cost matrices.
    
    Returns: np.ndarray coupling matrix.
    """
    import ot
    return ot.sinkhorn(mu0, mu1, cost, reg)


def compute_meta_cost_matrix_batch(
    source_dists: list[np.ndarray],
    target_dists: list[np.ndarray],
    cost: np.ndarray,
    use_sinkhorn: bool = True,
    reg: float = 0.05,
) -> np.ndarray:
    """
    Compute W(mu_s, nu_t) for all pairs of source and target distributions.
    
    This is the main bottleneck for meta-level dataset generation.
    
    If use_sinkhorn: uses entropic OT for each pair (faster, batchable).
    Else: uses exact LP (slower but exact).
    
    Further optimization: the Sinkhorn algorithm can be batched over
    pairs using GPU tensors (via geomloss or ot.gpu if available).
    For now, simple loop is fine.
    
    Returns: np.ndarray (M_source, M_target) meta-cost matrix.
    """
    import ot
    M_s = len(source_dists)
    M_t = len(target_dists)
    W = np.zeros((M_s, M_t))
    
    solver = ot.sinkhorn2 if use_sinkhorn else ot.emd2
    kwargs = {'reg': reg} if use_sinkhorn else {}
    
    for s in range(M_s):
        for t in range(M_t):
            W[s, t] = solver(source_dists[s], target_dists[t], cost, **kwargs)
    
    return W
```

---

## 4. Optimized Dataset Generation

### Updated `meta_fm/dataset.py`

The MetaFlowMatchingDataset should accept an optional GeodesicCache and use
the fast paths:

```python
class MetaFlowMatchingDataset(torch.utils.data.Dataset):
    def __init__(self, graph, source_distributions, target_distributions,
                 n_samples, seed=42, cache=None):
        """
        If cache is provided, uses precomputed geodesic structure for
        fast conditional marginal and rate matrix computation.
        If cache is None, falls back to current (slower) implementation.
        """
        self.cache = cache or GeodesicCache(graph)
        
        # ... rest of init uses self.cache for all computations
```

The sample generation loop becomes:

```python
for sample_idx in range(n_samples):
    # Draw (s, t) from meta-OT coupling (same as before)
    s, t = draw_from_coupling(meta_coupling)
    mu_s = source_distributions[s]
    mu_t = target_distributions[t]
    
    # Get graph-level coupling (precomputed and stored)
    coupling = graph_couplings[(s, t)]
    
    # Precompute cache for this coupling (if not already done)
    self.cache.precompute_for_coupling(coupling)
    
    # Draw time
    tau = rng.uniform(0, 1)
    
    # Fast conditional marginal (uses cache)
    mu_tau = marginal_distribution_fast(self.cache, coupling, tau)
    
    # Fast rate matrix (uses cache)
    R_target = marginal_rate_matrix_fast(self.cache, coupling, tau)
    
    self.samples.append((mu_tau, tau, R_target))
```

### Precompute graph-level couplings

The graph-level OT couplings between all paired source/target distributions
should be computed once and stored, not recomputed per sample:

```python
# In dataset __init__:
# Precompute all graph-level couplings for pairs in the meta-coupling support
self.graph_couplings = {}
meta_nonzero = np.argwhere(meta_coupling > 1e-12)
for s_idx, t_idx in meta_nonzero:
    coupling = compute_ot_coupling(source_distributions[s_idx],
                                    target_distributions[t_idx], cost)
    self.graph_couplings[(s_idx, t_idx)] = coupling
    self.cache.precompute_for_coupling(coupling)
```

---

## 5. Summary of Speedups

| Operation | Current | Optimized | Speedup |
|-----------|---------|-----------|---------|
| Spatial weights w_k | Recomputed per (i,j,t) query | Precomputed once per (i,j) | ~n_samples x |
| Conditional marginal | Geodesic walk + binomial per query | Cached weights @ binom vector | ~d_max x |
| Branch structure | Neighbor lookup per query | Precomputed sparse list | ~constant |
| Rate matrix assembly | Dense NxN per (i,j) pair | Sparse accumulation | ~N/E_geo x |
| Meta-cost matrix | Exact LP per pair | Sinkhorn (optional) | ~5-10x |
| Graph-level couplings | Recomputed per sample | Precomputed once per pair | ~n_samples x |

The dominant speedup comes from precomputing spatial weights and graph-level
couplings. For the current experiments (N=6-25, 5000-10000 samples), this
should reduce dataset generation from minutes to seconds.

---

## 6. Implementation Priority

1. **GeodesicCache** with spatial_weights and branch_structure — biggest win.
2. **Coupling-aware caching** — limits memory for larger graphs.
3. **Precomputed graph-level couplings** in dataset — eliminates redundant LP solves.
4. **Sinkhorn for meta-cost matrix** — optional, matters when M is large.
5. **Vectorized conditional marginal** with cached binom_pmf — minor but clean.

All changes are backward-compatible. The existing API should still work;
the cache is an optional acceleration passed to functions that support it.
