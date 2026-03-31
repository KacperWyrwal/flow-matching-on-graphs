"""
Optimal transport solvers for graph-based flow matching.

Combines graph_ot_fm/ot_stage1.py, ot_stage2.py, and shortest_paths.py.
"""

import numpy as np
import ot
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from scipy.special import gammaln, logsumexp


# ── Stage 1: W1 optimal transport via network simplex ─────────────────────────

def solve_w1(dist_matrix, mu0, mu1):
    """Solve W1 OT via POT's network simplex.

    Returns (w1_value, coupling, dual_alpha, dual_beta).
    """
    mu0 = np.asarray(mu0, dtype=np.float64)
    mu1 = np.asarray(mu1, dtype=np.float64)
    dist = np.asarray(dist_matrix, dtype=np.float64)

    # Replace inf with large value
    dist_safe = dist.copy()
    max_finite = np.max(dist_safe[np.isfinite(dist_safe)]) if np.any(np.isfinite(dist_safe)) else 1.0
    dist_safe[~np.isfinite(dist_safe)] = max_finite * 1e6

    coupling, log_info = ot.emd(mu0, mu1, dist_safe, log=True)
    w1_value = np.sum(coupling * dist_safe)

    alpha = log_info['u']
    beta = log_info['v']

    return w1_value, coupling, alpha, beta


def extract_optimal_face(dist_matrix, dual_alpha, dual_beta, tol=1e-8):
    """Extract the optimal face S from dual variables.

    S = {(i,j) : |d(i,j) - alpha_i - beta_j| < tol * (1 + |d(i,j)|)}

    Returns boolean mask (N, N).
    """
    dist = np.asarray(dist_matrix, dtype=np.float64)
    alpha = np.asarray(dual_alpha, dtype=np.float64)
    beta = np.asarray(dual_beta, dtype=np.float64)

    slack = dist - alpha[:, None] - beta[None, :]
    threshold = tol * (1.0 + np.abs(dist))

    # Only consider finite distances
    mask = np.isfinite(dist) & (np.abs(slack) < threshold)
    return mask


# ── Stage 2: Constrained entropic OT via log-domain Sinkhorn ─────────────────

class SinkhornConvergenceError(Exception):
    def __init__(self, message, iterations, error):
        super().__init__(message)
        self.iterations = iterations
        self.error = error


def solve_tiebreaker(support_mask, log_geodesic_counts, distances,
                     mu0, mu1, max_iter=5000, tol=1e-9):
    """Log-domain Sinkhorn on restricted support to find canonical coupling.

    Parameters
    ----------
    support_mask : (N, N) boolean, the optimal face S
    log_geodesic_counts : (N, N) log N_ij
    distances : (N, N) shortest path distances
    mu0, mu1 : (N,) probability distributions
    max_iter : int, maximum Sinkhorn iterations
    tol : float, convergence tolerance on marginal error

    Returns
    -------
    coupling : (N, N)
    """
    N = len(mu0)
    mu0 = np.asarray(mu0, dtype=np.float64)
    mu1 = np.asarray(mu1, dtype=np.float64)

    # Build log kernel: log K_ij = log N_ij - gammaln(d_ij + 1)
    # For (i,j) in S only; -inf otherwise
    log_K = np.full((N, N), -np.inf)

    # Vectorized construction
    si, sj = np.where(support_mask)
    d_vals = distances[si, sj]
    log_n_vals = log_geodesic_counts[si, sj]

    # Filter: need finite distance and finite log_n (not -inf)
    valid = np.isfinite(d_vals) & np.isfinite(log_n_vals) & (log_n_vals > -np.inf)
    si_v, sj_v = si[valid], sj[valid]
    d_v = d_vals[valid]
    log_n_v = log_n_vals[valid]

    log_K[si_v, sj_v] = log_n_v - gammaln(d_v + 1)

    # Log-domain Sinkhorn (vectorized)
    log_mu0 = np.log(mu0 + 1e-300)
    log_mu1 = np.log(mu1 + 1e-300)

    f = log_mu0.copy()  # (N,)
    g = np.zeros(N)     # (N,)

    # Precompute mask for rows/cols that have any support
    row_has_support = support_mask.any(axis=1)
    col_has_support = support_mask.any(axis=0)

    # Create masked log_K where unsupported entries are -inf (already done)
    for iteration in range(max_iter):
        # Vectorized f update: f_i = log mu0_i - LSE_j(log_K_ij + g_j)
        log_K_plus_g = log_K + g[None, :]  # (N, N)
        log_K_plus_g[~support_mask] = -np.inf
        lse_rows = logsumexp(log_K_plus_g, axis=1)
        f_new = log_mu0 - lse_rows
        # Only update rows with support
        f[row_has_support] = f_new[row_has_support]

        # Vectorized g update: g_j = log mu1_j - LSE_i(log_K_ij + f_i)
        log_K_plus_f = log_K + f[:, None]  # (N, N)
        log_K_plus_f[~support_mask] = -np.inf
        lse_cols = logsumexp(log_K_plus_f, axis=0)
        g_new = log_mu1 - lse_cols
        g[col_has_support] = g_new[col_has_support]

        # Check convergence every 10 iterations
        if (iteration + 1) % 10 == 0:
            log_pi = f[:, None] + log_K + g[None, :]
            log_pi[~support_mask] = -np.inf

            row_sums = logsumexp(log_pi, axis=1)
            col_sums = logsumexp(log_pi, axis=0)

            err0 = np.max(np.abs(np.exp(row_sums) - mu0))
            err1 = np.max(np.abs(np.exp(col_sums) - mu1))

            if max(err0, err1) < tol:
                break

    # Build coupling
    log_pi = f[:, None] + log_K + g[None, :]
    log_pi[~support_mask] = -np.inf
    coupling = np.exp(log_pi)
    coupling[~support_mask] = 0.0
    coupling = np.clip(coupling, 0.0, None)

    return coupling


# ── All-pairs shortest paths and log geodesic counts ──────────────────────────

def compute_shortest_paths_and_geodesics(R, metric="hop"):
    """Compute all-pairs shortest paths and log geodesic counts.

    Parameters
    ----------
    R : (N, N) rate matrix (off-diagonal >= 0, diagonal ignored)
    metric : "hop" (unit edge weights) or "rate" (-log R_ab edge weights)

    Returns
    -------
    dist_matrix : (N, N) shortest path distances (inf for unreachable)
    log_geodesic_counts : (N, N) log N_ij in log-space
    """
    N = R.shape[0]
    R_off = R.copy()
    np.fill_diagonal(R_off, 0.0)

    # Build edge weight matrix for Dijkstra
    if metric == "hop":
        # Unit edge weights
        weights = (R_off > 0).astype(float)
    elif metric == "rate":
        # Edge weight = -log(R_ab), only for edges with R_ab > 0
        weights = np.zeros((N, N))
        mask = R_off > 0
        weights[mask] = -np.log(R_off[mask])
        # Non-edges stay at 0 (no edge in sparse graph)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # All-pairs Dijkstra
    dist_matrix = shortest_path(csr_matrix(weights), method='D', directed=True)

    # Geodesic count DP in log-space
    log_geo = np.full((N, N), -np.inf)  # log(0) = -inf
    np.fill_diagonal(log_geo, 0.0)  # log(1) = 0

    # Precompute edge weights for predecessor check
    if metric == "hop":
        edge_weights = (R_off > 0).astype(float)
    else:
        edge_weights = np.zeros((N, N))
        m = R_off > 0
        edge_weights[m] = -np.log(R_off[m])

    log_R_off = np.full((N, N), -np.inf)
    pos = R_off > 0
    log_R_off[pos] = np.log(R_off[pos])

    for i in range(N):
        # For source i, process nodes in order of increasing distance
        dists_from_i = dist_matrix[i]
        finite_mask = np.isfinite(dists_from_i)
        nodes = np.where(finite_mask)[0]
        order = nodes[np.argsort(dists_from_i[nodes])]

        for j in order:
            if j == i:
                continue

            d_ij = dists_from_i[j]

            # Find predecessors: k where R_off[k,j] > 0 and d(i,k) + w(k,j) = d(i,j)
            # Vectorized: check all k at once
            has_edge = R_off[:, j] > 0  # (N,)
            k_finite = np.isfinite(dists_from_i) & has_edge
            if not k_finite.any():
                continue

            k_indices = np.where(k_finite)[0]
            w_kj = edge_weights[k_indices, j]
            d_ik = dists_from_i[k_indices]

            is_pred = np.abs(d_ik + w_kj - d_ij) < 1e-9
            pred_indices = k_indices[is_pred]

            if len(pred_indices) == 0:
                continue

            # log_terms[m] = log_geo[i, k] + log(R_off[k, j])
            log_terms = log_geo[i, pred_indices] + log_R_off[pred_indices, j]
            valid = np.isfinite(log_terms)
            if valid.any():
                log_geo[i, j] = logsumexp(log_terms[valid])

    return dist_matrix, log_geo
