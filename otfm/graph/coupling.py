"""
Optimal transport coupling computation for graph-based flow matching.

From graph_ot_fm/ot_solver.py.
"""

import numpy as np
import ot

from otfm.graph.structure import GraphStructure
from otfm.core.ot import solve_w1, extract_optimal_face, solve_tiebreaker


def compute_cost_matrix(graph: GraphStructure, cost="w2") -> np.ndarray:
    """
    Compute the pairwise OT cost c(i,j) for all node pairs.

    Parameters
    ----------
    graph : GraphStructure
    cost : str, one of "w2", "w1", "sbp"
        "w2" — original behavior (DP or d^2)
        "w1" or "sbp" — returns hop distance matrix

    Returns: np.ndarray (N, N) cost matrix.
    """
    if cost in ("sbp", "w1"):
        return graph.dist.copy()

    # cost == "w2": original behavior
    N = graph.N
    R = graph.R
    dist = graph.dist

    # Check if graph is unweighted (all off-diagonal R in {0,1})
    R_offdiag = R.copy()
    np.fill_diagonal(R_offdiag, 0.0)
    off_diag_vals = R_offdiag[R_offdiag > 0]
    is_unweighted = np.allclose(off_diag_vals, 1.0) if len(off_diag_vals) > 0 else True

    if is_unweighted:
        # Fall back to squared hop distance
        cost_mat = dist.copy()
        cost_mat[np.isfinite(cost_mat)] = cost_mat[np.isfinite(cost_mat)] ** 2
        # i == j case: cost = 0
        np.fill_diagonal(cost_mat, 0.0)
        return cost_mat

    # Weighted case: dynamic programming
    # V[j][a] = expected cost-to-go from a to j
    cost_mat = np.full((N, N), np.inf)
    np.fill_diagonal(cost_mat, 0.0)

    for j in range(N):
        # V[a] = expected neg-log cost from a to j
        V = np.full(N, np.inf)
        V[j] = 0.0

        # Process nodes in order of increasing distance to j
        # We process from closest to j outward
        finite_mask = np.isfinite(dist[:, j])
        nodes_by_dist = sorted(
            [a for a in range(N) if finite_mask[a]],
            key=lambda a: dist[a, j]
        )

        for a in nodes_by_dist:
            if a == j:
                V[a] = 0.0
                continue
            neighbors = graph.closer_neighbors[(a, j)]
            if not neighbors:
                continue
            probs = graph.branching_probs(a, j)
            val = 0.0
            for b, p in probs.items():
                r_ab = R[a, b]
                if r_ab <= 0:
                    continue
                val += p * (-np.log(r_ab) + V[b])
            V[a] = val

        for a in range(N):
            if np.isfinite(V[a]):
                cost_mat[a, j] = V[a]

    return cost_mat


def compute_ot_coupling_sinkhorn(
    mu0: np.ndarray,
    mu1: np.ndarray,
    cost: np.ndarray,
    reg: float = 0.05,
) -> np.ndarray:
    """
    Entropic OT coupling via Sinkhorn algorithm.

    Faster than exact LP for the meta-level coupling where slight
    suboptimality in the coupling is acceptable.

    Args:
        reg: entropic regularization. Smaller = closer to exact OT but slower.
             0.05 is a good default for normalized cost matrices.

    Returns: np.ndarray (N, N) coupling matrix.
    """
    mu0 = np.array(mu0, dtype=float)
    mu1 = np.array(mu1, dtype=float)
    cost_finite = np.array(cost, dtype=float)
    max_finite = np.max(cost_finite[np.isfinite(cost_finite)])
    cost_finite[~np.isfinite(cost_finite)] = max_finite * 1e6
    return ot.sinkhorn(mu0, mu1, cost_finite, reg)


def compute_meta_cost_matrix_batch(
    source_dists: list,
    target_dists: list,
    cost: np.ndarray,
    use_sinkhorn: bool = True,
    reg: float = 0.05,
) -> np.ndarray:
    """
    Compute W(mu_s, nu_t) for all (source, target) pairs.

    Returns: np.ndarray (M_source, M_target) meta-cost matrix.

    If use_sinkhorn=True, uses entropic OT (~5-10x faster than exact LP).
    If use_sinkhorn=False, uses exact LP (ot.emd2).
    """
    cost_finite = np.array(cost, dtype=float)
    max_finite = np.max(cost_finite[np.isfinite(cost_finite)])
    cost_finite[~np.isfinite(cost_finite)] = max_finite * 1e6

    M_s = len(source_dists)
    M_t = len(target_dists)
    W = np.zeros((M_s, M_t))

    if use_sinkhorn:
        for s in range(M_s):
            for t in range(M_t):
                W[s, t] = ot.sinkhorn2(
                    np.array(source_dists[s], dtype=float),
                    np.array(target_dists[t], dtype=float),
                    cost_finite, reg,
                )
    else:
        for s in range(M_s):
            for t in range(M_t):
                W[s, t] = ot.emd2(
                    np.array(source_dists[s], dtype=float),
                    np.array(target_dists[t], dtype=float),
                    cost_finite,
                )

    return W


def compute_ot_coupling(
    mu0: np.ndarray,
    mu1: np.ndarray,
    cost_matrix=None,
    *,
    graph_struct=None,
    cost="sbp",
    metric="hop",
    sinkhorn_max_iter=5000,
    sinkhorn_tol=1e-9,
    return_info=False,
):
    """
    Solve the discrete OT problem for coupling pi(i,j).

    Parameters
    ----------
    mu0 : (N,) source distribution
    mu1 : (N,) target distribution
    cost_matrix : (N, N) cost matrix. If provided, forces legacy W2 mode
        regardless of the cost parameter.
    graph_struct : GraphStructure (required for cost="sbp" or cost="w1")
    cost : str, one of "sbp" (default), "w1", "w2"
    metric : str, "hop" or "rate" (used for distance computation)
    sinkhorn_max_iter : int, max iterations for Stage 2 Sinkhorn
    sinkhorn_tol : float, convergence tolerance for Stage 2
    return_info : bool, if True return dict with extra info

    Returns
    -------
    coupling : (N, N) optimal coupling matrix
    info : dict (only if return_info=True)
    """
    mu0 = np.array(mu0, dtype=np.float64)
    mu1 = np.array(mu1, dtype=np.float64)

    # Backward compat: if cost_matrix is passed, use W2 mode
    if cost_matrix is not None:
        cost = "w2"

    if cost == "w2":
        # Legacy mode: use cost_matrix with ot.emd
        if cost_matrix is None:
            raise ValueError("cost_matrix is required for cost='w2'")
        cost_mat = np.array(cost_matrix, dtype=np.float64)

        # Replace inf with large finite value for OT solver
        cost_finite = cost_mat.copy()
        max_finite = np.max(cost_finite[np.isfinite(cost_finite)])
        cost_finite[~np.isfinite(cost_finite)] = max_finite * 1e6

        coupling = ot.emd(mu0, mu1, cost_finite)

        if return_info:
            return coupling, {"cost": "w2", "transport_cost": np.sum(coupling * cost_finite)}
        return coupling

    elif cost == "w1":
        # Stage 1 only: W1 optimal transport
        if graph_struct is None:
            raise ValueError("graph_struct is required for cost='w1'")

        distances = graph_struct.dist
        w1_value, coupling, alpha, beta = solve_w1(distances, mu0, mu1)

        if return_info:
            return coupling, {
                "cost": "w1",
                "w1_value": w1_value,
                "dual_alpha": alpha,
                "dual_beta": beta,
            }
        return coupling

    elif cost == "sbp":
        # Two-stage: W1 + geodesic tiebreaker
        if graph_struct is None:
            raise ValueError("graph_struct is required for cost='sbp'")

        distances = graph_struct.dist
        log_geo = graph_struct.log_geodesic_counts

        # Stage 1: W1 optimal transport
        w1_value, w1_coupling, alpha, beta = solve_w1(distances, mu0, mu1)

        # Extract optimal face
        face = extract_optimal_face(distances, alpha, beta)

        # Stage 2: Sinkhorn tiebreaker on optimal face
        coupling = solve_tiebreaker(
            face, log_geo, distances, mu0, mu1,
            max_iter=sinkhorn_max_iter, tol=sinkhorn_tol,
        )

        if return_info:
            return coupling, {
                "cost": "sbp",
                "w1_value": w1_value,
                "w1_coupling": w1_coupling,
                "dual_alpha": alpha,
                "dual_beta": beta,
                "optimal_face": face,
            }
        return coupling

    else:
        raise ValueError(f"Unknown cost type: {cost}")
