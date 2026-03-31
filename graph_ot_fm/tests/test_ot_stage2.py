"""Tests for Stage 2: Constrained entropic OT tiebreaker."""
import numpy as np
import pytest
from scipy.special import gammaln

from graph_ot_fm.ot_stage2 import solve_tiebreaker
from graph_ot_fm.ot_stage1 import solve_w1, extract_optimal_face


def _make_path_dist(n):
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = abs(i - j)
    return dist


def _make_cycle_dist(n):
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = min(abs(i - j), n - abs(i - j))
    return dist.astype(float)


def _log_geo_from_count(count):
    """Convert count matrix to log counts."""
    log_geo = np.full_like(count, -np.inf)
    pos = count > 0
    log_geo[pos] = np.log(count[pos])
    return log_geo


class TestIdentityProblem:
    def test_identity_coupling(self):
        """mu0 = mu1 => coupling should be diag(mu0)."""
        N = 4
        dist = _make_path_dist(N)
        mu = np.array([0.1, 0.2, 0.3, 0.4])

        # For identity problem, the optimal face includes diagonal
        _, _, alpha, beta = solve_w1(dist, mu, mu)
        face = extract_optimal_face(dist, alpha, beta)

        # Geodesic counts for path graph
        geo_count = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                # On path graph, exactly one geodesic between any pair
                geo_count[i, j] = 1.0
        log_geo = _log_geo_from_count(geo_count)

        coupling = solve_tiebreaker(face, log_geo, dist, mu, mu)

        # Should be close to diagonal
        expected = np.diag(mu)
        np.testing.assert_allclose(coupling, expected, atol=1e-6)


class TestMarginalVerification:
    def test_marginals_path(self):
        """Row and column sums should match mu0 and mu1."""
        N = 5
        dist = _make_path_dist(N)
        mu0 = np.array([0.3, 0.2, 0.1, 0.2, 0.2])
        mu1 = np.array([0.1, 0.1, 0.3, 0.3, 0.2])

        _, _, alpha, beta = solve_w1(dist, mu0, mu1)
        face = extract_optimal_face(dist, alpha, beta)

        geo_count = np.ones((N, N))  # path graph: 1 geodesic per pair
        log_geo = _log_geo_from_count(geo_count)

        coupling = solve_tiebreaker(face, log_geo, dist, mu0, mu1)

        np.testing.assert_allclose(coupling.sum(axis=1), mu0, atol=1e-4)
        np.testing.assert_allclose(coupling.sum(axis=0), mu1, atol=1e-4)

    def test_marginals_cycle(self):
        N = 6
        dist = _make_cycle_dist(N)
        mu0 = np.ones(N) / N
        mu1 = np.array([0.4, 0.1, 0.1, 0.1, 0.1, 0.2])

        _, _, alpha, beta = solve_w1(dist, mu0, mu1)
        face = extract_optimal_face(dist, alpha, beta)

        # Cycle C6: for distance d, there are 2 geodesics (except d=0 and d=3)
        geo_count = np.ones((N, N))
        for i in range(N):
            for j in range(N):
                d = dist[i, j]
                if d == 0:
                    geo_count[i, j] = 1
                elif 0 < d < N / 2:
                    geo_count[i, j] = 2  # two paths (CW and CCW have same hop length only at d=N/2)
                elif d == N / 2:
                    geo_count[i, j] = 2
                # Actually on unit-weight C6: at distance d with unit edges,
                # matrix power gives geo count. For d<N/2 there's 1 path each way = 2^0 * path_count
                # Let's just use 1 for simplicity - the test checks marginals not specific values
                geo_count[i, j] = 1
        log_geo = _log_geo_from_count(geo_count)

        coupling = solve_tiebreaker(face, log_geo, dist, mu0, mu1)

        np.testing.assert_allclose(coupling.sum(axis=1), mu0, atol=1e-4)
        np.testing.assert_allclose(coupling.sum(axis=0), mu1, atol=1e-4)


class TestUniqueStage1:
    def test_unique_solution_preserved(self):
        """When Stage 1 has unique solution, Stage 2 should return same."""
        N = 2
        dist = np.array([[0, 1], [1, 0]], dtype=float)
        mu0 = np.array([0.7, 0.3])
        mu1 = np.array([0.4, 0.6])

        _, w1_coupling, alpha, beta = solve_w1(dist, mu0, mu1)
        face = extract_optimal_face(dist, alpha, beta)

        geo_count = np.ones((N, N))
        log_geo = _log_geo_from_count(geo_count)

        coupling = solve_tiebreaker(face, log_geo, dist, mu0, mu1)

        # For 2 nodes, the LP solution is unique
        np.testing.assert_allclose(coupling, w1_coupling, atol=1e-6)


class TestNonNegativity:
    def test_coupling_nonneg(self):
        N = 4
        dist = _make_path_dist(N)
        mu0 = np.array([0.5, 0.2, 0.2, 0.1])
        mu1 = np.array([0.1, 0.3, 0.3, 0.3])

        _, _, alpha, beta = solve_w1(dist, mu0, mu1)
        face = extract_optimal_face(dist, alpha, beta)

        geo_count = np.ones((N, N))
        log_geo = _log_geo_from_count(geo_count)

        coupling = solve_tiebreaker(face, log_geo, dist, mu0, mu1)

        assert np.all(coupling >= -1e-12)
