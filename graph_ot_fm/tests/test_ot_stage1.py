"""Tests for Stage 1: W1 optimal transport."""
import numpy as np
import pytest
from graph_ot_fm.ot_stage1 import solve_w1, extract_optimal_face


def _make_path_dist(n):
    """Distance matrix for path graph P_n."""
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = abs(i - j)
    return dist


def _make_cycle_dist(n):
    """Distance matrix for cycle graph C_n."""
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = min(abs(i - j), n - abs(i - j))
    return dist.astype(float)


class TestDiracMasses:
    def test_w1_dirac_path(self):
        dist = _make_path_dist(5)
        mu0 = np.array([1, 0, 0, 0, 0], dtype=float)
        mu1 = np.array([0, 0, 0, 1, 0], dtype=float)
        w1, coupling, alpha, beta = solve_w1(dist, mu0, mu1)
        assert np.isclose(w1, 3.0)

    def test_w1_dirac_same(self):
        dist = _make_path_dist(4)
        mu0 = np.array([0, 1, 0, 0], dtype=float)
        mu1 = np.array([0, 1, 0, 0], dtype=float)
        w1, coupling, alpha, beta = solve_w1(dist, mu0, mu1)
        assert np.isclose(w1, 0.0)

    def test_coupling_dirac(self):
        dist = _make_path_dist(4)
        mu0 = np.array([1, 0, 0, 0], dtype=float)
        mu1 = np.array([0, 0, 0, 1], dtype=float)
        w1, coupling, alpha, beta = solve_w1(dist, mu0, mu1)
        assert np.isclose(coupling[0, 3], 1.0)


class TestStrongDuality:
    def test_duality_path(self):
        dist = _make_path_dist(5)
        mu0 = np.array([0.3, 0.2, 0.1, 0.2, 0.2])
        mu1 = np.array([0.1, 0.1, 0.3, 0.3, 0.2])
        w1, coupling, alpha, beta = solve_w1(dist, mu0, mu1)
        # Strong duality: W1 = alpha @ mu0 + beta @ mu1
        dual_obj = alpha @ mu0 + beta @ mu1
        assert np.isclose(w1, dual_obj, rtol=1e-6)

    def test_duality_cycle(self):
        dist = _make_cycle_dist(6)
        mu0 = np.ones(6) / 6
        mu1 = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        w1, coupling, alpha, beta = solve_w1(dist, mu0, mu1)
        dual_obj = alpha @ mu0 + beta @ mu1
        assert np.isclose(w1, dual_obj, rtol=1e-6)


class TestDualFeasibility:
    def test_dual_constraint(self):
        """alpha_i + beta_j <= d(i,j) for all (i,j)."""
        dist = _make_path_dist(5)
        mu0 = np.array([0.3, 0.2, 0.1, 0.2, 0.2])
        mu1 = np.array([0.1, 0.1, 0.3, 0.3, 0.2])
        w1, coupling, alpha, beta = solve_w1(dist, mu0, mu1)

        slack = dist - alpha[:, None] - beta[None, :]
        # All slack values should be >= -tol
        assert np.all(slack > -1e-6)


class TestOptimalFace:
    def test_face_covers_coupling(self):
        """The optimal face must contain all (i,j) with coupling > 0."""
        dist = _make_path_dist(5)
        mu0 = np.array([0.3, 0.2, 0.1, 0.2, 0.2])
        mu1 = np.array([0.1, 0.1, 0.3, 0.3, 0.2])
        w1, coupling, alpha, beta = solve_w1(dist, mu0, mu1)
        face = extract_optimal_face(dist, alpha, beta)

        # Every nonzero coupling entry should be in the face
        nonzero = coupling > 1e-12
        assert np.all(face[nonzero])

    def test_face_diagonal_included(self):
        """Diagonal (i,i) should always be in face (d=0, alpha+beta=0)."""
        dist = _make_path_dist(4)
        mu0 = np.array([0.25, 0.25, 0.25, 0.25])
        mu1 = np.array([0.25, 0.25, 0.25, 0.25])
        w1, coupling, alpha, beta = solve_w1(dist, mu0, mu1)
        face = extract_optimal_face(dist, alpha, beta)
        for i in range(4):
            assert face[i, i]
