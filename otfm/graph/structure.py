"""
Graph structure with precomputed geodesic information for flow matching.

Combines graph_ot_fm/graph.py, geodesic_cache.py, conditional.py,
and meta_fm/model.py::rate_matrix_to_edge_index.
"""

import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from scipy.stats import binom


class GraphStructure:
    """
    Precomputes and caches all geodesic information needed for flow matching.

    Constructor args:
        rate_matrix: np.ndarray of shape (N, N)
            Reference rate matrix R. R[i,j] > 0 iff edge i->j exists.
            Diagonal entries are ignored (recomputed as -sum of off-diag row).

    Attributes:
        N: int — number of nodes
        R: np.ndarray (N, N) — cleaned rate matrix (diagonal = -sum of off-diag)
        dist: np.ndarray (N, N) — shortest path distances d(i,j) (hop count)
        geodesic_count: np.ndarray (N, N) — N_a values
            geodesic_count[a, j] = (R_offdiag^{d(a,j)})_{aj}
        closer_neighbors: dict
            closer_neighbors[(a, j)] = list of nodes b where:
              R[a,b] > 0 AND dist[b,j] == dist[a,j] - 1
    """

    def __init__(self, rate_matrix: np.ndarray):
        rate_matrix = np.array(rate_matrix, dtype=float)
        N = rate_matrix.shape[0]
        assert rate_matrix.shape == (N, N), "Rate matrix must be square"

        self.N = N

        # Build cleaned rate matrix: off-diagonal from input, diagonal = -sum(off-diag row)
        R = rate_matrix.copy()
        np.fill_diagonal(R, 0.0)
        np.fill_diagonal(R, -R.sum(axis=1))
        self.R = R

        # Off-diagonal part only (used for matrix powers)
        R_offdiag = R.copy()
        np.fill_diagonal(R_offdiag, 0.0)

        # Adjacency matrix for hop-distance computation (binary, unweighted)
        adjacency = (R_offdiag > 0).astype(float)

        # Shortest path distances (hop count)
        dist_raw = shortest_path(csr_matrix(adjacency), method='D', directed=True)
        self.dist = dist_raw  # inf for unreachable pairs

        dmax = int(np.nanmax(dist_raw[np.isfinite(dist_raw)]))

        # Compute geodesic counts via matrix powers of R_offdiag
        # geodesic_count[a, j] = (R_offdiag^{d(a,j)})_{a,j}
        geodesic_count = np.zeros((N, N))

        # R_offdiag^0 = identity (d=0 case)
        R_pow = np.eye(N)
        # d=0: a==j, geodesic_count[a,a] = 1 (by convention for the formula)
        # Actually for d=0, R^0 = I, so (R^0)[a,j] = delta(a,j)
        for a in range(N):
            geodesic_count[a, a] = 1.0  # d=0

        # Compute R_offdiag^1, R_offdiag^2, ..., R_offdiag^dmax
        R_pow = R_offdiag.copy()
        for d in range(1, dmax + 1):
            for a in range(N):
                for j in range(N):
                    if int(round(dist_raw[a, j])) == d:
                        geodesic_count[a, j] = R_pow[a, j]
            if d < dmax:
                R_pow = R_pow @ R_offdiag

        self.geodesic_count = geodesic_count

        # Log geodesic counts for numerical stability
        self.log_geodesic_counts = np.full((N, N), -np.inf)
        positive = geodesic_count > 0
        self.log_geodesic_counts[positive] = np.log(geodesic_count[positive])

        # Alias for consistency with shortest_paths module
        self.distances = self.dist

        # Precompute closer_neighbors
        # closer_neighbors[(a, j)] = list of b with R[a,b] > 0 and dist[b,j] == dist[a,j] - 1
        closer_neighbors = {}
        for a in range(N):
            for j in range(N):
                if a == j or not np.isfinite(dist_raw[a, j]):
                    closer_neighbors[(a, j)] = []
                    continue
                d_aj = dist_raw[a, j]
                neighbors = []
                for b in range(N):
                    if R_offdiag[a, b] > 0 and np.isfinite(dist_raw[b, j]):
                        if abs(dist_raw[b, j] - (d_aj - 1)) < 1e-9:
                            neighbors.append(b)
                closer_neighbors[(a, j)] = neighbors
        self.closer_neighbors = closer_neighbors

    def branching_probs(self, a: int, j: int) -> dict:
        """
        Returns {b: R[a,b] * N_b / N_a} for b in closer_neighbors[(a,j)].
        These are the geodesic random walk transition probabilities (sum to 1).
        """
        neighbors = self.closer_neighbors[(a, j)]
        if not neighbors:
            return {}

        N_a = self.geodesic_count[a, j]
        if N_a == 0:
            return {}

        probs = {}
        for b in neighbors:
            N_b = self.geodesic_count[b, j]
            probs[b] = self.R[a, b] * N_b / N_a

        # Verify they sum to 1
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6, (
            f"Branching probs don't sum to 1 for a={a}, j={j}: sum={total}, probs={probs}"
        )
        return probs


# ── Conditional probability paths and rate matrices ───────────────────────────

def conditional_marginal(
    graph: GraphStructure,
    i: int,
    j: int,
    t: float,
) -> np.ndarray:
    """
    Compute p_t(x | i, j) for all nodes x.

    p_t(x | i,j) = sum_{k=0}^{d} Binom(k; d, t) * w_k(x | i, j)

    where d = dist[i,j] and w_k(x) is the probability of being at x
    after k steps of the geodesic random walk from i toward j.
    """
    N = graph.N
    t = float(np.clip(t, 0.0, 1.0))

    # Edge cases
    if i == j:
        result = np.zeros(N)
        result[i] = 1.0
        return result

    d = int(round(graph.dist[i, j]))

    if t == 0.0:
        result = np.zeros(N)
        result[i] = 1.0
        return result

    if t >= 1.0:
        result = np.zeros(N)
        result[j] = 1.0
        return result

    # Compute w_k(x) for k = 0, 1, ..., d
    # w_0 = delta(x, i)
    # w_{k+1}(x) = sum_{a: x in N^-(a)} w_k(a) * P_geo(a->x|j)
    # i.e., w_{k+1}(x) = sum_a w_k(a) * P_geo(a->x|j)

    # Build the geodesic transition matrix T where T[a, b] = P_geo(a->b|j)
    # T[a, b] = R[a,b] * N_b / N_a if b in closer_neighbors[(a,j)], else 0
    T = np.zeros((N, N))
    for a in range(N):
        probs = graph.branching_probs(a, j)
        for b, p in probs.items():
            T[a, b] = p

    # Compute w_k for k=0,...,d using repeated application of T
    w_list = []
    w = np.zeros(N)
    w[i] = 1.0
    w_list.append(w.copy())

    for _ in range(d):
        w = w @ T  # row vector @ transition matrix
        w_list.append(w.copy())

    # Combine: p_t = sum_k Binom(k; d, t) * w_k
    result = np.zeros(N)
    for k in range(d + 1):
        binom_weight = binom.pmf(k, d, t)
        result += binom_weight * w_list[k]

    # Clip for numerical stability
    result = np.clip(result, 0.0, None)
    s = result.sum()
    if s > 0:
        result /= s
    return result


def conditional_rate_matrix(
    graph: GraphStructure,
    i: int,
    j: int,
    t: float,
) -> np.ndarray:
    """
    Compute the conditional rate matrix R_t^{i->j}(a, b) for all a, b.

    For node a on a geodesic from i to j (d(i,a) + d(a,j) == d(i,j)):
        For b in closer_neighbors[(a,j)]:
            R_t[a, b] = d(a,j) / (1-t) * R[a,b] * N_b / N_a
        R_t[a, a] = -d(a,j) / (1-t)
    """
    N = graph.N
    t = float(np.clip(t, 0.0, 0.999))

    # Edge case
    if i == j:
        return np.zeros((N, N))

    d_ij = graph.dist[i, j]
    R_t = np.zeros((N, N))

    for a in range(N):
        d_ia = graph.dist[i, a]
        d_aj = graph.dist[a, j]
        # Check if a is on some geodesic from i to j
        if not (np.isfinite(d_ia) and np.isfinite(d_aj)):
            continue
        if abs(d_ia + d_aj - d_ij) > 1e-9:
            continue
        # a is on a geodesic
        if d_aj < 1e-9:
            # a == j, no outgoing rate
            continue

        scale = d_aj / (1.0 - t)
        neighbors = graph.closer_neighbors[(a, j)]
        N_a = graph.geodesic_count[a, j]
        if N_a == 0:
            continue

        off_diag_sum = 0.0
        for b in neighbors:
            N_b = graph.geodesic_count[b, j]
            rate = scale * graph.R[a, b] * N_b / N_a
            R_t[a, b] = rate
            off_diag_sum += rate

        R_t[a, a] = -off_diag_sum

    return R_t


def sample_conditional_state(
    graph: GraphStructure,
    i: int,
    j: int,
    t: float,
    rng: np.random.Generator,
) -> int:
    """
    Sample x ~ p_t(.|i,j).

    Algorithm:
        1. d = dist[i,j]
        2. k ~ Binom(d, t)
        3. Run k steps of geodesic random walk from i toward j
        4. Return final node
    """
    t = float(np.clip(t, 0.0, 1.0))

    if i == j:
        return i

    d = int(round(graph.dist[i, j]))

    if t == 0.0:
        return i
    if t >= 1.0:
        return j

    # Sample k ~ Binom(d, t)
    k = int(rng.binomial(d, t))

    # Run k steps of geodesic random walk from i toward j
    current = i
    for _ in range(k):
        probs = graph.branching_probs(current, j)
        if not probs:
            break
        nodes = list(probs.keys())
        weights = np.array([probs[b] for b in nodes])
        current = int(rng.choice(nodes, p=weights))

    return current


# ── GeodesicCache ─────────────────────────────────────────────────────────────

class GeodesicCache:
    """
    Precomputes and stores all time-independent geodesic structure for a graph.

    Attributes:
        spatial_weights: dict[(i,j)] -> np.ndarray (d+1, N)
            spatial_weights[(i,j)][k, x] = probability of being at node x
            after exactly k steps of the geodesic random walk from i toward j.

        branch_structure: dict[(i,j)] -> list of (a, b, weight) tuples
            Time-independent part of the conditional rate matrix.
            R_t^{i->j}(a,b) = weight / (1 - t)

    Computation is lazy: pairs are only computed when first requested
    (or when precompute_for_coupling is called).
    """

    def __init__(self, graph: GraphStructure):
        self.graph = graph
        self.spatial_weights: dict = {}
        self.branch_structure: dict = {}

        # Build per-target geodesic transition matrices (target-indexed, shared)
        self._P_geo: dict = {}  # j -> (N, N) transition matrix toward j

    # -- internal -----------------------------------------------------------

    def _build_P_geo(self, j: int) -> np.ndarray:
        """Build the geodesic transition matrix toward target j (cached)."""
        if j in self._P_geo:
            return self._P_geo[j]
        graph = self.graph
        N = graph.N
        P = np.zeros((N, N))
        for a in range(N):
            da = graph.dist[a, j]
            if da == 0 or np.isinf(da):
                continue
            Na = graph.geodesic_count[a, j]
            if Na <= 0:
                continue
            for b in graph.closer_neighbors.get((a, j), []):
                Nb = graph.geodesic_count[b, j]
                P[a, b] = graph.R[a, b] * Nb / Na
        self._P_geo[j] = P
        return P

    def _compute_pair(self, i: int, j: int) -> None:
        """Compute and cache spatial_weights and branch_structure for (i, j)."""
        if (i, j) in self.spatial_weights:
            return
        graph = self.graph
        d = int(graph.dist[i, j])
        if d == 0 or np.isinf(graph.dist[i, j]):
            return

        N = graph.N
        P = self._build_P_geo(j)

        # Spatial weights: w_k via repeated matrix-vector products
        weights = np.zeros((d + 1, N))
        weights[0, i] = 1.0
        for k in range(d):
            weights[k + 1] = weights[k] @ P
        self.spatial_weights[(i, j)] = weights

        # Branch structure: sparse list of (a, b, weight)
        d_ij = graph.dist[i, j]
        branches = []
        for a in range(N):
            da = graph.dist[a, j]
            if np.isinf(da) or da == 0:
                continue
            if graph.dist[i, a] + da != d_ij:
                continue  # not on any geodesic from i to j
            Na = graph.geodesic_count[a, j]
            if Na <= 0:
                continue
            for b in graph.closer_neighbors.get((a, j), []):
                Nb = graph.geodesic_count[b, j]
                w = da * graph.R[a, b] * Nb / Na
                branches.append((a, b, w))
        self.branch_structure[(i, j)] = branches

    # -- public API ---------------------------------------------------------

    def precompute_for_coupling(self, coupling: np.ndarray) -> None:
        """Cache all (i, j) pairs with pi(i,j) > 0 (at most 2N-1 entries)."""
        for i, j in np.argwhere(coupling > 1e-12):
            i, j = int(i), int(j)
            if i != j:
                self._compute_pair(i, j)

    def get_spatial_weights(self, i: int, j: int) -> np.ndarray:
        """Return cached (d+1, N) spatial weights, computing on demand."""
        if (i, j) not in self.spatial_weights:
            self._compute_pair(i, j)
        return self.spatial_weights[(i, j)]

    def conditional_marginal(self, i: int, j: int, t: float,
                              binom_pmf: np.ndarray | None = None) -> np.ndarray:
        """
        p_t(x | i, j) = sum_k Binom(k; d, t) * spatial_weights[k, x]

        If binom_pmf is pre-computed and passed in, it is used directly
        (avoids recomputing for the same (d, t) across multiple i values).
        """
        from scipy.stats import binom as binom_dist

        if i == j:
            out = np.zeros(self.graph.N)
            out[i] = 1.0
            return out

        d = int(self.graph.dist[i, j])
        weights = self.get_spatial_weights(i, j)  # (d+1, N)

        if binom_pmf is None:
            binom_pmf = binom_dist.pmf(np.arange(d + 1), d, t)
        return binom_pmf @ weights  # (N,)

    def conditional_rate_sparse(self, i: int, j: int, t: float) -> list:
        """
        Returns list of (a, b, rate) where rate = weight / (1 - t).
        More memory-efficient than the full NxN matrix for sparse graphs.
        """
        if i == j:
            return []
        if (i, j) not in self.branch_structure:
            self._compute_pair(i, j)
        inv_1mt = 1.0 / (1.0 - t)
        return [(a, b, w * inv_1mt) for a, b, w in self.branch_structure[(i, j)]]


# ── Utility: rate_matrix_to_edge_index (from meta_fm/model.py) ────────────────

def rate_matrix_to_edge_index(R: np.ndarray) -> torch.LongTensor:
    """
    Convert a rate matrix to a PyG edge_index tensor.

    Extracts all (i, j) pairs where R[i, j] > 0 and i != j.
    Returns torch.LongTensor of shape (2, num_edges).
    """
    rows, cols = np.where((R > 0) & (np.eye(len(R)) == 0))
    return torch.tensor(np.stack([rows, cols]), dtype=torch.long)
