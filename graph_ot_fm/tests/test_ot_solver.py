"""Tests for the updated ot_solver with W1/SBP support."""
import numpy as np
import pytest
from graph_ot_fm.graph import GraphStructure
from graph_ot_fm.ot_solver import compute_cost_matrix, compute_ot_coupling


def _make_path_rate_matrix(n):
    R = np.zeros((n, n))
    for i in range(n - 1):
        R[i, i + 1] = 1.0
        R[i + 1, i] = 1.0
    return R


def _make_cycle_rate_matrix(n):
    R = np.zeros((n, n))
    for i in range(n):
        R[i, (i + 1) % n] = 1.0
        R[(i + 1) % n, i] = 1.0
    return R


class TestBackwardCompat:
    def test_w2_positional(self):
        """Legacy call: compute_ot_coupling(mu0, mu1, cost_matrix) works."""
        R = _make_path_rate_matrix(4)
        g = GraphStructure(R)
        cost = compute_cost_matrix(g)
        mu0 = np.array([0.4, 0.3, 0.2, 0.1])
        mu1 = np.array([0.1, 0.2, 0.3, 0.4])
        coupling = compute_ot_coupling(mu0, mu1, cost)
        # Check it's a valid coupling
        np.testing.assert_allclose(coupling.sum(axis=1), mu0, atol=1e-8)
        np.testing.assert_allclose(coupling.sum(axis=0), mu1, atol=1e-8)

    def test_w2_keyword(self):
        """Explicit cost='w2' matches legacy behavior."""
        R = _make_path_rate_matrix(4)
        g = GraphStructure(R)
        cost = compute_cost_matrix(g)
        mu0 = np.array([0.4, 0.3, 0.2, 0.1])
        mu1 = np.array([0.1, 0.2, 0.3, 0.4])
        c1 = compute_ot_coupling(mu0, mu1, cost)
        c2 = compute_ot_coupling(mu0, mu1, cost, cost="w2")
        np.testing.assert_allclose(c1, c2, atol=1e-12)

    def test_compute_cost_matrix_default(self):
        """compute_cost_matrix(graph) still works (defaults to w2)."""
        R = _make_path_rate_matrix(4)
        g = GraphStructure(R)
        cost = compute_cost_matrix(g)
        # For unweighted path graph, cost = d^2
        assert np.isclose(cost[0, 3], 9.0)  # 3^2


class TestW1Optimality:
    def test_sbp_w1_optimal(self):
        """SBP coupling should achieve same W1 cost as pure W1."""
        R = _make_cycle_rate_matrix(6)
        g = GraphStructure(R)
        mu0 = np.ones(6) / 6
        mu1 = np.array([0.4, 0.1, 0.1, 0.1, 0.1, 0.2])

        c_w1 = compute_ot_coupling(mu0, mu1, graph_struct=g, cost="w1")
        c_sbp = compute_ot_coupling(mu0, mu1, graph_struct=g, cost="sbp")

        w1_cost = np.sum(c_w1 * g.dist)
        sbp_cost = np.sum(c_sbp * g.dist)

        # SBP should achieve same W1 cost (it's on the optimal face)
        np.testing.assert_allclose(sbp_cost, w1_cost, rtol=1e-6)


class TestDeterminism:
    def test_sbp_deterministic(self):
        """Running SBP twice gives identical results."""
        R = _make_path_rate_matrix(5)
        g = GraphStructure(R)
        mu0 = np.array([0.3, 0.2, 0.1, 0.2, 0.2])
        mu1 = np.array([0.1, 0.1, 0.3, 0.3, 0.2])

        c1 = compute_ot_coupling(mu0, mu1, graph_struct=g, cost="sbp")
        c2 = compute_ot_coupling(mu0, mu1, graph_struct=g, cost="sbp")

        np.testing.assert_allclose(c1, c2, atol=1e-12)


class TestTwoNode:
    def test_two_node_fully_determined(self):
        """2-node graph: coupling is fully determined by marginals."""
        R = np.array([[0, 1], [1, 0]], dtype=float)
        g = GraphStructure(R)
        mu0 = np.array([0.7, 0.3])
        mu1 = np.array([0.4, 0.6])

        coupling = compute_ot_coupling(mu0, mu1, graph_struct=g, cost="sbp")

        # For 2 nodes: pi[0,0]=0.4, pi[0,1]=0.3, pi[1,0]=0, pi[1,1]=0.3
        # Actually: pi[0,0] = min(0.7, 0.4) = 0.4, pi[1,1] = min(0.3, 0.6) = 0.3
        # pi[0,1] = 0.7 - 0.4 = 0.3, pi[1,0] = 0.3 - 0.3 = 0
        expected = np.array([[0.4, 0.3], [0.0, 0.3]])
        np.testing.assert_allclose(coupling, expected, atol=1e-6)

    def test_two_node_marginals(self):
        R = np.array([[0, 1], [1, 0]], dtype=float)
        g = GraphStructure(R)
        mu0 = np.array([0.7, 0.3])
        mu1 = np.array([0.4, 0.6])

        coupling = compute_ot_coupling(mu0, mu1, graph_struct=g, cost="sbp")
        np.testing.assert_allclose(coupling.sum(axis=1), mu0, atol=1e-8)
        np.testing.assert_allclose(coupling.sum(axis=0), mu1, atol=1e-8)


class TestCycleC6:
    def test_balanced_transport(self):
        """C6 balanced: uniform->uniform should give identity-like coupling."""
        R = _make_cycle_rate_matrix(6)
        g = GraphStructure(R)
        mu = np.ones(6) / 6

        coupling = compute_ot_coupling(mu, mu, graph_struct=g, cost="sbp")

        # mu0 = mu1 = uniform, so coupling should be diag(1/6)
        expected = np.diag(np.ones(6) / 6)
        np.testing.assert_allclose(coupling, expected, atol=1e-6)


class TestReturnInfo:
    def test_return_info_w1(self):
        R = _make_path_rate_matrix(4)
        g = GraphStructure(R)
        mu0 = np.array([0.5, 0.2, 0.2, 0.1])
        mu1 = np.array([0.1, 0.3, 0.3, 0.3])

        coupling, info = compute_ot_coupling(
            mu0, mu1, graph_struct=g, cost="w1", return_info=True
        )
        assert "w1_value" in info
        assert "dual_alpha" in info
        assert "dual_beta" in info

    def test_return_info_sbp(self):
        R = _make_path_rate_matrix(4)
        g = GraphStructure(R)
        mu0 = np.array([0.5, 0.2, 0.2, 0.1])
        mu1 = np.array([0.1, 0.3, 0.3, 0.3])

        coupling, info = compute_ot_coupling(
            mu0, mu1, graph_struct=g, cost="sbp", return_info=True
        )
        assert "w1_value" in info
        assert "optimal_face" in info
        assert "w1_coupling" in info


class TestCostMatrixModes:
    def test_w1_cost_is_distance(self):
        R = _make_path_rate_matrix(4)
        g = GraphStructure(R)
        cost = compute_cost_matrix(g, cost="w1")
        np.testing.assert_allclose(cost, g.dist)

    def test_sbp_cost_is_distance(self):
        R = _make_path_rate_matrix(4)
        g = GraphStructure(R)
        cost = compute_cost_matrix(g, cost="sbp")
        np.testing.assert_allclose(cost, g.dist)

    def test_w2_cost_is_squared(self):
        R = _make_path_rate_matrix(4)
        g = GraphStructure(R)
        cost = compute_cost_matrix(g, cost="w2")
        expected = g.dist.copy()
        expected[np.isfinite(expected)] = expected[np.isfinite(expected)] ** 2
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_allclose(cost, expected)
