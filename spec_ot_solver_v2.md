# Implementation Spec: Two-Stage Graph Optimal Transport Solver

## Overview

Implement the canonical optimal transport coupling on a finite weighted graph, derived from the Schrödinger bridge zero-noise limit. Integrated into the existing `graph_ot_fm` package as a replacement/upgrade to the current `compute_ot_coupling` function.

Two-stage approach:

1. **Stage 1 (LP):** Solve the Wasserstein-1 optimal transport problem via network simplex to obtain the W₁ value and the optimal face (support set of all W₁-optimal couplings).
2. **Stage 2 (Constrained Entropic OT):** Solve an entropic OT problem restricted to the optimal face, using a refined cost derived from path combinatorics, to select the unique canonical coupling.

---

## 1. Mathematical Background

### 1.1 Problem Definition

**Input:**
- A connected directed or undirected graph $G$ with $N$ nodes, specified by a rate matrix $R$.
- Two probability distributions $\mu_0, \mu_1$ on the node set $\{0, \ldots, N-1\}$.

**Output:**
- The optimal coupling $\pi^* \in \mathbb{R}^{N \times N}_{\geq 0}$ with marginals $\mu_0, \mu_1$ that:
  - Minimises expected shortest-path distance (W₁-optimal), and
  - Among all W₁-optimal couplings, minimises $\sum_{ij} \pi_{ij} \tilde{c}_{ij} - H(\pi)$ where $\tilde{c}_{ij} = \log(d(i,j)!) - \log \mathcal{N}_{ij}$.

### 1.2 Key Quantities

| Symbol | Definition | How to compute |
|--------|-----------|----------------|
| $d(i,j)$ | Shortest-path distance from $i$ to $j$ on $G$ | All-pairs Dijkstra or Floyd-Warshall |
| $\mathcal{N}_{ij}$ | Weighted geodesic count: $(R^{d(i,j)})_{ij}$ = sum of products of edge rates over all shortest paths from $i$ to $j$ | DP on shortest-path DAG (see §3.2) |
| $\tilde{c}_{ij}$ | Tiebreaker cost: $\log(d(i,j)!) - \log \mathcal{N}_{ij}$ | Direct computation from $d$ and $\mathcal{N}$ |
| $W_1$ | Wasserstein-1 value: $\min_\pi \sum_{ij} \pi_{ij} d(i,j)$ | Network simplex (Stage 1) |
| $\Pi_1$ | Set of W₁-optimal couplings | Characterised by dual variables from Stage 1 |
| $S$ | Support of optimal face: $\{(i,j) : d(i,j) = \alpha_i + \beta_j\}$ | Complementary slackness from Stage 1 duals |

### 1.3 Distance Convention

Two natural conventions for edge weights, controlled by a parameter:

- **`metric="hop"`**: Unweighted graph distance. Each edge has unit length. $d(i,j)$ is the hop count. This is the natural choice when $R$ is the combinatorial Laplacian (all nonzero $R_{ij}$ equal).
- **`metric="rate"`**: Edge length is $-\log R_{ij}$ (the "rate distance"). This arises naturally from the Schrödinger bridge when edge rates are heterogeneous: the dominant path maximises the product of rates, i.e., minimises the sum of $-\log R_{ij}$. In this case, all geodesics between a given pair have equal rate-product (by definition of the rate distance), so $\mathcal{N}_{ij}$ reduces to the unweighted count of shortest paths.
- **`metric="custom"`**: User provides an explicit edge-weight matrix for distances.

**Default:** `metric="hop"` (the most common case).

### 1.4 Backward Compatibility

The current `compute_ot_coupling` function uses $c(i,j) = d(i,j)^2$ (W₂ cost). This is retained as a legacy option via `cost="w2"`:

- **`cost="sbp"` (new default):** Two-stage W₁ + tiebreaker, as derived from the Schrödinger bridge.
- **`cost="w1"`:** Pure W₁ without tiebreaker (Stage 1 only, returns the LP solution directly).
- **`cost="w2"`:** Legacy behaviour, $c(i,j) = d(i,j)^2$, single LP solve.

---

## 2. Integration into `graph_ot_fm`

### 2.1 Module Structure

Add new modules to the existing package:

```
graph_ot_fm/
├── __init__.py              # Update exports
├── graph.py                 # Existing: GraphStructure (update)
├── ot_solver.py             # Existing: update compute_ot_coupling
├── ot_stage1.py             # NEW: W₁ LP via network simplex
├── ot_stage2.py             # NEW: Constrained entropic OT via Sinkhorn
├── shortest_paths.py        # NEW: All-pairs shortest paths + geodesic counts
│                            #   (replaces/supplements distance computation
│                            #    currently in GraphStructure)
├── flow.py                  # Existing: unchanged
├── tests/
│   ├── test_shortest_paths.py   # NEW
│   ├── test_ot_stage1.py        # NEW
│   ├── test_ot_stage2.py        # NEW
│   └── test_ot_solver.py        # NEW (integration tests)
└── ...
```

### 2.2 Updated Public API

#### Top-Level Function (updated `compute_ot_coupling`)

```python
def compute_ot_coupling(
    mu0: np.ndarray,
    mu1: np.ndarray,
    cost_matrix: np.ndarray | None = None,
    graph_struct: GraphStructure | None = None,
    cost: str = "sbp",
    metric: str = "hop",
    sinkhorn_max_iter: int = 5000,
    sinkhorn_tol: float = 1e-9,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Compute the optimal transport coupling on a graph.

    Parameters
    ----------
    mu0, mu1 : arrays, shape (N,)
        Source and target probability distributions.

    cost_matrix : array or None
        Explicit cost matrix. If provided, used directly (legacy mode).
        If None, computed from graph_struct using the specified cost type.

    graph_struct : GraphStructure or None
        Graph structure with distances and geodesic counts.
        Required when cost_matrix is None.

    cost : {"sbp", "w1", "w2"}
        Cost type:
        - "sbp": Two-stage SBP-derived coupling (W₁ + canonical tiebreaker).
                 This is the theoretically correct cost from the Schrödinger
                 bridge zero-noise limit. NEW DEFAULT.
        - "w1": Pure Wasserstein-1 (Stage 1 only, LP solution).
        - "w2": Legacy Wasserstein-2, c(i,j) = d(i,j)^2.

    metric : {"hop", "rate", "custom"}
        Distance convention. Only used when cost_matrix is None.

    sinkhorn_max_iter : int
        Maximum Sinkhorn iterations for Stage 2 (only for cost="sbp").

    sinkhorn_tol : float
        Convergence tolerance for Stage 2.

    return_info : bool
        If True, return (coupling, info_dict).

    Returns
    -------
    coupling : np.ndarray, shape (N, N)

    info : dict (only if return_info=True)
        Keys: "w1_value", "support_size", "sinkhorn_iterations",
        "sinkhorn_error", "dual_alpha", "dual_beta".
    """
```

This signature is backward compatible: existing code that passes
`compute_ot_coupling(mu0, mu1, cost_matrix)` still works, using the
explicit cost matrix in legacy mode.

#### Lower-Level Functions (new, also public)

```python
# In ot_stage1.py
def solve_w1(
    dist_matrix: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (w1_value, w1_coupling, dual_alpha, dual_beta).
    Uses POT's network simplex solver.
    """

def extract_optimal_face(
    dist_matrix: np.ndarray,
    dual_alpha: np.ndarray,
    dual_beta: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Returns boolean mask (N, N) indicating which (i,j) pairs
    belong to the optimal face S.
    """

# In ot_stage2.py
def solve_tiebreaker(
    support_mask: np.ndarray,
    log_geodesic_counts: np.ndarray,
    distances: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    max_iter: int = 5000,
    tol: float = 1e-9,
) -> np.ndarray:
    """
    Log-domain Sinkhorn on the restricted support.
    Returns the canonical coupling.
    """

# In shortest_paths.py
def compute_shortest_paths_and_geodesics(
    R: np.ndarray,
    metric: str = "hop",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (dist_matrix, log_geodesic_count_matrix).

    dist_matrix[i,j] = shortest path distance.
    log_geodesic_count_matrix[i,j] = log N_{ij}.

    Geodesic counts are stored in log-space to avoid overflow/underflow.
    """
```

### 2.3 Update to GraphStructure

The existing `GraphStructure` class computes distances and geodesic
counts. Update it to:

1. Store `log_geodesic_counts` instead of (or in addition to) raw counts.
2. Use the new `compute_shortest_paths_and_geodesics` function internally.
3. Support the `metric` parameter.

```python
class GraphStructure:
    def __init__(self, R, metric="hop"):
        self.R = R
        self.N = R.shape[0]
        self.metric = metric

        # Compute distances and geodesic counts
        self.distances, self.log_geodesic_counts = \
            compute_shortest_paths_and_geodesics(R, metric)

        # Backward compat: raw geodesic counts (may overflow for large graphs)
        self.geodesic_counts = np.exp(self.log_geodesic_counts)

        # Precompute graph neighbors, etc. (existing code)
        ...
```

### 2.4 Update to `compute_cost_matrix`

```python
def compute_cost_matrix(graph_struct, cost="sbp"):
    """
    Compute the OT cost matrix from graph structure.

    cost="sbp": returns d(i,j) for W₁ primary cost.
                Tiebreaker cost is computed separately in Stage 2.
    cost="w1":  same as "sbp" (W₁ cost = d(i,j)).
    cost="w2":  returns d(i,j)^2 (legacy).
    """
    if cost in ("sbp", "w1"):
        return graph_struct.distances.copy()
    elif cost == "w2":
        return graph_struct.distances ** 2
    else:
        raise ValueError(f"Unknown cost type: {cost}")
```

---

## 3. Algorithm Details

### 3.1 All-Pairs Shortest Paths + Geodesic Counts (`shortest_paths.py`)

For each source node $i = 0, \ldots, N-1$:

1. **Dijkstra pass:** Run Dijkstra's algorithm from node $i$ with edge weights:
   - `"hop"`: all edge weights = 1
   - `"rate"`: edge weight for $a \to b$ is $-\log R_{ab}$
   
   Yields $d(i, j)$ for all $j$.

2. **Geodesic count DP:** Process nodes in order of increasing distance from $i$. For each node $j$:

   $$\mathcal{N}_{ij} = \sum_{\substack{k : R_{kj} > 0 \\ d(i,k) + w_{kj} = d(i,j)}} \mathcal{N}_{ik} \cdot R_{kj}$$

   Base case: $\mathcal{N}_{ii} = 1$.

   Here $w_{kj}$ is the Dijkstra edge weight, and $R_{kj}$ is the actual rate. This correctly computes $(R^{d(i,j)})_{ij}$.

   **Log-space version** (for numerical stability):
   
   $$\log \mathcal{N}_{ij} = \mathrm{LSE}_{k : \text{tight}} \left(\log \mathcal{N}_{ik} + \log R_{kj}\right)$$

#### Implementation Notes

- Use `scipy.sparse.csgraph.shortest_path` with `method='D'` for Dijkstra, which handles sparse graphs efficiently.
- Implement the geodesic count DP separately (scipy doesn't provide it).
- For the combined Dijkstra + DP pass, consider a custom implementation using `heapq` where the DP is interleaved: when node $j$ is settled, compute $\log\mathcal{N}_{ij}$ immediately.
- Store `log_geodesic_counts` as a dense array (same shape as distances).

### 3.2 Stage 1: W₁ via Network Simplex (`ot_stage1.py`)

Use POT library:

```python
import ot

def solve_w1(dist_matrix, mu0, mu1):
    coupling, log_info = ot.emd(mu0, mu1, dist_matrix, log=True)
    w1_value = np.sum(coupling * dist_matrix)
    
    # Extract dual variables from POT
    # POT returns 'u' and 'v' in log_info
    alpha = log_info['u']
    beta = log_info['v']
    
    # Verify dual feasibility: alpha_i + beta_j <= d(i,j)
    # and equality on support of coupling
    
    return w1_value, coupling, alpha, beta
```

#### Extracting the Optimal Face

$$S = \{(i,j) : |d(i,j) - \alpha_i - \beta_j| < \texttt{tol} \cdot (1 + |d(i,j)|)\}$$

Use relative tolerance for floating-point robustness.

**Verification:**
- Every $(i,j)$ with $\pi^*_{ij} > 0$ must satisfy $(i,j) \in S$.
- For every $i$ with $\mu_0(i) > 0$, there must exist $j$ with $(i,j) \in S$.
- For every $j$ with $\mu_1(j) > 0$, there must exist $i$ with $(i,j) \in S$.

If verification fails, increase tolerance and retry.

### 3.3 Stage 2: Constrained Entropic OT (`ot_stage2.py`)

#### Problem

$$\min_{\pi \in \Pi(\mu_0, \mu_1),\, \mathrm{supp}(\pi) \subseteq S} \sum_{(i,j) \in S} \pi_{ij} \, \tilde{c}_{ij} - H(\pi)$$

where $\tilde{c}_{ij} = \log(d(i,j)!) - \log \mathcal{N}_{ij}$ and $H(\pi) = -\sum_{ij} \pi_{ij} \log \pi_{ij}$.

Equivalently, minimise $\mathrm{KL}(\pi \| \tilde{K})$ over $\Pi(\mu_0, \mu_1)$ restricted to $S$, where $\log \tilde{K}_{ij} = \log \mathcal{N}_{ij} - \mathrm{gammaln}(d(i,j) + 1)$.

#### Log-Domain Sinkhorn on Restricted Support

Maintain dual variables $f_i, g_j$ in log-space. The coupling is $\pi_{ij} = \exp(f_i + \log \tilde{K}_{ij} + g_j)$ for $(i,j) \in S$, zero otherwise.

Update rules:

$$f_i \leftarrow \log \mu_0(i) - \mathrm{LSE}_{j: (i,j) \in S}\bigl(\log \tilde{K}_{ij} + g_j\bigr)$$

$$g_j \leftarrow \log \mu_1(j) - \mathrm{LSE}_{i: (i,j) \in S}\bigl(\log \tilde{K}_{ij} + f_i\bigr)$$

**Convergence:** Check marginal violation every 10 iterations. Stop when $\max(\varepsilon_0, \varepsilon_1) < \texttt{tol}$.

#### Edge Cases

1. **$d(i,j) = 0$ ($i = j$):** $\tilde{c}_{ii} = \log(0!) - \log(1) = 0$, so $\tilde{K}_{ii} = 1$.
2. **$\mathcal{N}_{ij} = 0$ for $(i,j) \in S$:** Should never occur on connected graph. Raise error.
3. **Large $d(i,j)$:** Use `scipy.special.gammaln(d + 1)` for $\log(d!)$.
4. **Disconnected support:** Detect bipartite components in $S$, solve each independently.

#### Initialisation

$f_i = \log \mu_0(i)$, $g_j = 0$ for all $j$.

---

## 4. Testing Strategy

### 4.1 Unit Tests

#### `test_shortest_paths.py`
- **Path graph $P_4$:** $d(0,3) = 3$, $\mathcal{N}_{0,3} = R_{01} R_{12} R_{23}$ (one geodesic).
- **Cycle graph $C_4$:** $d(0,2) = 2$, $\mathcal{N}_{0,2} = R_{01}R_{12} + R_{03}R_{32}$ (two geodesics).
- **Complete graph $K_4$, uniform rates:** $d(i,j) = 1$, $\mathcal{N}_{ij} = 1$.
- **Grid graph:** verify against known combinatorial formulas.
- **Asymmetric graph:** verify $d(i,j)$ may differ from $d(j,i)$ and $\mathcal{N}_{ij} \neq \mathcal{N}_{ji}$.
- **Consistency:** For unweighted graphs with unit rates, $\mathcal{N}_{ij}$ = number of shortest paths.

#### `test_ot_stage1.py`
- **Dirac masses:** $\mu_0 = \delta_i$, $\mu_1 = \delta_j$. Then $W_1 = d(i,j)$.
- **Dual feasibility:** $\alpha_i + \beta_j \leq d(i,j)$ for all pairs, equality on support.
- **Strong duality:** $W_1 = \sum_i \alpha_i \mu_0(i) + \sum_j \beta_j \mu_1(j)$.
- **Optimal face coverage:** support of $\mu_0$ and $\mu_1$ covered by $S$.

#### `test_ot_stage2.py`
- **Unique Stage 1 solution:** Stage 2 returns the same coupling.
- **Symmetric problem:** $\mu_0 = \mu_1$ yields identity coupling.
- **Marginal verification:** both marginals satisfied.

#### `test_ot_solver.py` (Integration)
- **W₁-optimality:** canonical coupling achieves $W_1$.
- **Determinism:** same input, same output.
- **2-node graph:** coupling fully determined by marginals. Verify.
- **Cycle $C_6$, balanced transport:** multiple W₁-optimal plans exist; verify unique selection.
- **Backward compat:** `cost="w2"` reproduces old behaviour.
- **Comparison:** for complete graph $K_N$, the SBP coupling should match
  the W₂ coupling (since on $K_N$, $d(i,j) = 1$ for all pairs, so W₁ and
  W₂ are equivalent up to scaling).

### 4.2 Property-Based Tests

Random connected graphs + random distributions. Verify:
- Marginals correct
- Non-negativity
- W₁-optimality (expected distance = W₁)
- Determinism

---

## 5. Dependencies

**Required (already in project):**
- `numpy >= 1.24`
- `scipy >= 1.10`
- `pot >= 0.9`

**No new dependencies.**

---

## 6. Error Handling

Use existing exception patterns in `graph_ot_fm`. Add:

```python
class SinkhornConvergenceError(Exception):
    """Sinkhorn did not converge. Includes iteration count and error."""
    def __init__(self, message, iterations, error):
        super().__init__(message)
        self.iterations = iterations
        self.error = error
```

---

## 7. Numerical Considerations

1. **Log-space throughout Stage 2.** Use `scipy.special.logsumexp`.
2. **$\log(d!)$ via `gammaln`.** `scipy.special.gammaln(d + 1)`.
3. **Floating-point tolerance for face extraction.** Relative tolerance: $(i,j) \in S$ iff $|d(i,j) - \alpha_i - \beta_j| < \texttt{tol} \cdot (1 + |d(i,j)|)$.
4. **Prune zero-mass nodes** from Stage 2 to reduce Sinkhorn problem size.
5. **Log-space geodesic counts.** Store and compute $\log \mathcal{N}_{ij}$ throughout, using LSE for the DP accumulation.
6. **Sparse output for large $N$.** Return `scipy.sparse.csr_matrix` when $N > 1000$.

---

## 8. Migration Plan

### Phase 1: Implement new modules
- `shortest_paths.py` with `compute_shortest_paths_and_geodesics`
- `ot_stage1.py` with `solve_w1` and `extract_optimal_face`
- `ot_stage2.py` with `solve_tiebreaker`
- Tests for each module

### Phase 2: Update existing code
- Update `GraphStructure` to use new shortest paths module and store log geodesic counts
- Update `compute_ot_coupling` to support `cost="sbp"` as the new default
- Update `compute_cost_matrix` to return $d(i,j)$ for SBP/W₁ cost
- Add `cost="w2"` backward compatibility path

### Phase 3: Update experiments
- All experiments switch to `cost="sbp"` (the new default)
- Verify results are comparable or improved
- Document any differences in a migration note

### Phase 4: Validate
- Run full test suite
- Re-run a subset of experiments (Ex17, Ex18) to verify the new solver produces correct couplings
- Compare SBP coupling vs old W₂ coupling on a few diagnostic cases

---

## 9. Stretch Goals

- **GPU acceleration:** Port Stage 2 Sinkhorn to PyTorch for GPU execution.
- **Approximate mode:** $\varepsilon$-scaling Sinkhorn as alternative to two-stage for very large graphs.
- **Lazy geodesic counts:** For $N > 5000$, compute $\mathcal{N}_{ij}$ only for $(i,j) \in S$ after Stage 1 identifies the face.
- **Visualisation:** Plot coupling on graph (edge thickness $\propto \pi_{ij}$).
