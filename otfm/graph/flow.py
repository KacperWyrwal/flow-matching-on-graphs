"""
Marginal flow computation for optimal transport flow matching on graphs.

From graph_ot_fm/flow.py.
"""

import numpy as np

from otfm.graph.structure import (
    GraphStructure,
    conditional_marginal,
    conditional_rate_matrix,
)


def marginal_distribution(
    graph: GraphStructure,
    coupling: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Compute p_t(a) = sum_{i,j} pi(i,j) * p_t(a|i,j).

    Returns: np.ndarray (N,) probability distribution.
    """
    N = graph.N
    t = float(np.clip(t, 0.0, 1.0))
    p_t = np.zeros(N)

    # Iterate over nonzero entries of coupling
    nonzero_idx = np.argwhere(coupling > 1e-12)
    for (src, tgt) in nonzero_idx:
        pi_ij = coupling[src, tgt]
        p_cond = conditional_marginal(graph, int(src), int(tgt), t)
        p_t += pi_ij * p_cond

    # Normalize
    s = p_t.sum()
    if s > 0:
        p_t /= s
    return p_t


def marginal_rate_matrix(
    graph: GraphStructure,
    coupling: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Compute the exact marginal rate matrix u_t(a, b).

    u_t(a, b) = sum_{i,j} pi(i,j) * p_t(a|i,j) / p_t(a) * R_t^{i->j}(a,b)

    Returns: np.ndarray (N, N) rate matrix.
    """
    N = graph.N
    t = float(np.clip(t, 0.0, 0.999))

    # Step 1: compute marginal p_t(a)
    p_t = marginal_distribution(graph, coupling, t)

    # Step 2: accumulate weighted rate matrices
    u_t = np.zeros((N, N))

    nonzero_idx = np.argwhere(coupling > 1e-12)
    for (src, tgt) in nonzero_idx:
        pi_ij = coupling[src, tgt]
        p_cond = conditional_marginal(graph, int(src), int(tgt), t)
        R_cond = conditional_rate_matrix(graph, int(src), int(tgt), t)

        # Weight: p_t(a|i,j) / p_t(a), handle division by zero
        weight = np.where(p_t > 1e-15, p_cond / p_t, 0.0)

        # Accumulate: u_t[a, b] += pi_ij * weight[a] * R_cond[a, b]
        # Broadcasting: weight[:, None] * R_cond gives (N, N)
        u_t += pi_ij * weight[:, None] * R_cond

    # Step 3: enforce valid rate matrix (zero row sums)
    # Recompute diagonal to ensure row sums = 0
    np.fill_diagonal(u_t, 0.0)
    np.fill_diagonal(u_t, -u_t.sum(axis=1))

    return u_t


def marginal_distribution_fast(
    cache,
    coupling: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Compute p_t(a) using precomputed GeodesicCache.

    Equivalent to marginal_distribution but avoids redundant geodesic walk
    computation by using cached spatial weights. Binom PMF is cached by
    distance so it is computed at most once per unique distance value.

    Returns: np.ndarray (N,) probability distribution.
    """
    from scipy.stats import binom as binom_dist

    N = cache.graph.N
    t = float(np.clip(t, 0.0, 1.0))
    p_t = np.zeros(N)
    binom_cache: dict = {}  # d -> binom_pmf vector

    nonzero_pairs = np.argwhere(coupling > 1e-12)
    for i, j in nonzero_pairs:
        i, j = int(i), int(j)
        pi_ij = coupling[i, j]
        if i == j:
            p_t[i] += pi_ij
        else:
            d = int(cache.graph.dist[i, j])
            if d not in binom_cache:
                binom_cache[d] = binom_dist.pmf(np.arange(d + 1), d, t)
            cond = cache.conditional_marginal(i, j, t, binom_pmf=binom_cache[d])
            p_t += pi_ij * cond

    s = p_t.sum()
    if s > 0:
        p_t /= s
    return p_t


def marginal_rate_matrix_fast(
    cache,
    coupling: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Compute u_t(a, b) using precomputed GeodesicCache.

    Speedups over marginal_rate_matrix:
      - Spatial weights precomputed (no geodesic walk per query)
      - Branch structure precomputed (no neighbor lookups per query)
      - Sparse accumulation over geodesic edges only
      - Binomial PMF computed once per (d, t) value, shared across sources

    Returns: np.ndarray (N, N) rate matrix.
    """
    from scipy.stats import binom as binom_dist

    N = cache.graph.N
    t = float(np.clip(t, 0.0, 0.999))
    inv_1mt = 1.0 / (1.0 - t)

    # Step 1: marginal p_t and per-pair conditionals (cache binom by distance)
    nonzero_pairs = np.argwhere(coupling > 1e-12)
    p_t = np.zeros(N)
    conditionals = {}
    binom_cache: dict = {}  # d -> binom_pmf vector

    for i, j in nonzero_pairs:
        i, j = int(i), int(j)
        pi_ij = coupling[i, j]
        if i == j:
            cond = np.zeros(N)
            cond[i] = 1.0
        else:
            d = int(cache.graph.dist[i, j])
            if d not in binom_cache:
                binom_cache[d] = binom_dist.pmf(np.arange(d + 1), d, t)
            cond = cache.conditional_marginal(i, j, t, binom_pmf=binom_cache[d])
        conditionals[(i, j)] = cond
        p_t += pi_ij * cond

    # Step 2: sparse rate matrix accumulation
    u_t = np.zeros((N, N))

    for i, j in nonzero_pairs:
        i, j = int(i), int(j)
        if i == j:
            continue
        if (i, j) not in cache.branch_structure:
            cache._compute_pair(i, j)
        pi_ij = coupling[i, j]
        cond = conditionals[(i, j)]

        for a, b, weight in cache.branch_structure[(i, j)]:
            if p_t[a] > 1e-12:
                u_t[a, b] += pi_ij * (cond[a] / p_t[a]) * weight * inv_1mt

    # Step 3: fix diagonal
    np.fill_diagonal(u_t, 0.0)
    np.fill_diagonal(u_t, -u_t.sum(axis=1))

    return u_t


def evolve_distribution(
    p0: np.ndarray,
    rate_matrix_fn,
    t_span: tuple,
    n_steps: int = 200,
) -> tuple:
    """
    Numerically evolve dp/dt = p @ R(t) from t_span[0] to t_span[1].

    Uses simple Euler method.

    Returns:
        times: np.ndarray (n_steps,)
        distributions: np.ndarray (n_steps, N)
    """
    t0, t1 = t_span
    times = np.linspace(t0, t1, n_steps)
    dt = times[1] - times[0] if n_steps > 1 else 0.0

    N = len(p0)
    distributions = np.zeros((n_steps, N))
    p = p0.copy().astype(float)

    for k, t in enumerate(times):
        distributions[k] = p
        if k < n_steps - 1:
            t_clamped = float(np.clip(t, 0.0, 0.999))
            R_t = rate_matrix_fn(t_clamped)
            dp = p @ R_t
            p = p + dt * dp
            # Clip and renormalize
            p = np.clip(p, 0.0, None)
            s = p.sum()
            if s > 1e-15:
                p /= s

    return times, distributions
