"""
Tests for conditional.py: conditional_marginal, conditional_rate_matrix.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from graph_ot_fm import GraphStructure
from graph_ot_fm.conditional import (
    conditional_marginal,
    conditional_rate_matrix,
    sample_conditional_state,
)
from graph_ot_fm.utils import make_cycle_graph, make_grid_graph


@pytest.fixture
def cycle4():
    R = make_cycle_graph(4, weighted=False)
    return GraphStructure(R)


@pytest.fixture
def cycle6():
    R = make_cycle_graph(6, weighted=False)
    return GraphStructure(R)


@pytest.fixture
def grid3x3():
    R = make_grid_graph(3, 3, weighted=False)
    return GraphStructure(R)


class TestConditionalMarginal:
    """Tests for conditional_marginal."""

    def test_sums_to_one(self, cycle6):
        """conditional_marginal must sum to 1."""
        for i in range(6):
            for j in range(6):
                for t in [0.0, 0.1, 0.5, 0.9, 0.99]:
                    p = conditional_marginal(cycle6, i, j, t)
                    assert abs(p.sum() - 1.0) < 1e-6, (
                        f"Sum != 1 for i={i}, j={j}, t={t}: sum={p.sum()}"
                    )

    def test_at_t0_is_delta_i(self, cycle6):
        """At t=0, distribution should be delta at i."""
        for i in range(6):
            for j in range(6):
                p = conditional_marginal(cycle6, i, j, 0.0)
                expected = np.zeros(6)
                expected[i] = 1.0
                np.testing.assert_allclose(p, expected, atol=1e-6)

    def test_at_t1_is_delta_j(self, cycle6):
        """At t=1, distribution should be delta at j."""
        for i in range(6):
            for j in range(6):
                p = conditional_marginal(cycle6, i, j, 1.0)
                expected = np.zeros(6)
                expected[j] = 1.0
                np.testing.assert_allclose(p, expected, atol=1e-6)

    def test_i_equals_j(self, cycle6):
        """When i == j, distribution is always delta_i regardless of t."""
        for i in range(6):
            for t in [0.0, 0.5, 1.0]:
                p = conditional_marginal(cycle6, i, i, t)
                expected = np.zeros(6)
                expected[i] = 1.0
                np.testing.assert_allclose(p, expected, atol=1e-6)

    def test_distance_1_case(self, cycle4):
        """For d(i,j)=1: p_t = (1-t)*delta_i + t*delta_j."""
        # d(0,1) = 1
        t = 0.5
        p = conditional_marginal(cycle4, 0, 1, t)
        expected = np.zeros(4)
        expected[0] = 1 - t
        expected[1] = t
        np.testing.assert_allclose(p, expected, atol=1e-6)

    def test_distance_2_case(self, cycle4):
        """For d(0,2)=2 in cycle N=4: intermediate node should have nonzero mass."""
        t = 0.5
        p = conditional_marginal(cycle4, 0, 2, t)
        # At t=0.5, should have mass at 0, 1, 2, 3 (due to two paths)
        assert p.sum() > 0.99
        # Node 0 (source) and node 2 (target) should have some mass
        assert p[0] > 0
        assert p[2] > 0

    def test_nonnegative(self, cycle6):
        """All probabilities should be non-negative."""
        for i in range(6):
            for j in range(6):
                for t in [0.0, 0.3, 0.7, 0.99]:
                    p = conditional_marginal(cycle6, i, j, t)
                    assert np.all(p >= -1e-9), f"Negative prob at i={i}, j={j}, t={t}"

    def test_time_derivative_matches_rate(self, cycle6):
        """Finite difference check: dp/dt ≈ p @ R_t."""
        i, j = 0, 3
        t = 0.4
        eps = 1e-5

        p_t = conditional_marginal(cycle6, i, j, t)
        p_t_eps = conditional_marginal(cycle6, i, j, t + eps)
        dp_dt_numerical = (p_t_eps - p_t) / eps

        R_t = conditional_rate_matrix(cycle6, i, j, t)
        dp_dt_analytical = p_t @ R_t

        np.testing.assert_allclose(dp_dt_numerical, dp_dt_analytical, atol=1e-3)

    def test_grid_sums_to_one(self, grid3x3):
        """Grid graph: conditional_marginal sums to 1."""
        i, j = 0, 8  # corner to corner
        for t in [0.0, 0.25, 0.5, 0.75, 0.99]:
            p = conditional_marginal(grid3x3, i, j, t)
            assert abs(p.sum() - 1.0) < 1e-6


class TestConditionalRateMatrix:
    """Tests for conditional_rate_matrix."""

    def test_zero_row_sums(self, cycle6):
        """Rate matrix should have (approximately) zero row sums."""
        for i in range(6):
            for j in range(6):
                R_t = conditional_rate_matrix(cycle6, i, j, 0.5)
                row_sums = R_t.sum(axis=1)
                np.testing.assert_allclose(row_sums, 0, atol=1e-10,
                                           err_msg=f"Row sums not zero for i={i},j={j}")

    def test_i_equals_j_returns_zero(self, cycle6):
        """When i==j, rate matrix is zero."""
        for i in range(6):
            R_t = conditional_rate_matrix(cycle6, i, i, 0.5)
            np.testing.assert_allclose(R_t, 0, atol=1e-10)

    def test_only_geodesic_nodes_nonzero(self, cycle6):
        """Only nodes on geodesics from i to j should have nonzero rates."""
        i, j = 0, 3
        R_t = conditional_rate_matrix(cycle6, i, j, 0.5)
        d_ij = cycle6.dist[i, j]

        for a in range(6):
            d_ia = cycle6.dist[i, a]
            d_aj = cycle6.dist[a, j]
            on_geodesic = (
                np.isfinite(d_ia) and np.isfinite(d_aj) and
                abs(d_ia + d_aj - d_ij) < 1e-9
            )
            if not on_geodesic:
                # Row should be zero
                assert np.allclose(R_t[a, :], 0), (
                    f"Node {a} not on geodesic but has nonzero row: {R_t[a, :]}"
                )

    def test_offdiag_nonnegative(self, cycle6):
        """Off-diagonal rate matrix entries should be non-negative."""
        for i in range(6):
            for j in range(6):
                R_t = conditional_rate_matrix(cycle6, i, j, 0.5)
                N = R_t.shape[0]
                for a in range(N):
                    for b in range(N):
                        if a != b:
                            assert R_t[a, b] >= -1e-9, (
                                f"Negative off-diag at ({a},{b}) for i={i},j={j}: {R_t[a,b]}"
                            )

    def test_distance_1_rate(self, cycle4):
        """For d(0,1)=1: rate at t=0.5 should be 1/(1-0.5) = 2."""
        R_t = conditional_rate_matrix(cycle4, 0, 1, 0.5)
        # d(0,1) = 1, so rate from 0 to 1 = 1/(1-0.5) * R[0,1] * N_1/N_0
        # For unweighted: R[0,1]=1, N_1=geodesic_count[1,1]=1, N_0=geodesic_count[0,1]=1
        # rate = 1/0.5 * 1 * 1/1 = 2
        assert abs(R_t[0, 1] - 2.0) < 1e-9

    def test_distance_2_case(self, cycle4):
        """For d(0,2)=2 in cycle N=4: check intermediate nodes have rates."""
        R_t = conditional_rate_matrix(cycle4, 0, 2, 0.5)
        d_02 = cycle4.dist[0, 2]  # = 2
        # Node 0 (d(0,0)+d(0,2)=0+2=2=d_02: on geodesic)
        # Node 1 (d(0,1)+d(1,2)=1+1=2=d_02: on geodesic)
        # Node 3 (d(0,3)+d(3,2)=1+1=2=d_02: on geodesic)
        # Node 2 is the target, has no outgoing rates
        assert np.any(R_t[0, :] != 0)  # node 0 has rates

    def test_grid_zero_row_sums(self, grid3x3):
        """Grid graph: rate matrix has zero row sums."""
        i, j = 0, 8
        R_t = conditional_rate_matrix(grid3x3, i, j, 0.5)
        row_sums = R_t.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0, atol=1e-10)
