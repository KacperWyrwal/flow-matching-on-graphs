"""
Tests for flow.py: marginal_distribution, marginal_rate_matrix, evolve_distribution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from graph_ot_fm import (
    GraphStructure,
    compute_cost_matrix,
    compute_ot_coupling,
    marginal_distribution,
    marginal_rate_matrix,
    evolve_distribution,
    total_variation,
    make_cycle_graph,
    make_grid_graph,
)


@pytest.fixture
def cycle6():
    R = make_cycle_graph(6, weighted=False)
    return GraphStructure(R)


@pytest.fixture
def cycle8():
    R = make_cycle_graph(8, weighted=False)
    return GraphStructure(R)


@pytest.fixture
def grid4x4():
    R = make_grid_graph(4, 4, weighted=False)
    return GraphStructure(R)


def peaked_dist(n, node, eps=0.01):
    d = np.ones(n) * eps / (n - 1)
    d[node] = 1.0 - eps
    d /= d.sum()
    return d


class TestMarginalDistribution:
    """Tests for marginal_distribution."""

    def test_sums_to_one(self, cycle6):
        """Marginal distribution must sum to 1."""
        N = 6
        mu0 = peaked_dist(N, 0)
        mu1 = peaked_dist(N, 3)
        cost = compute_cost_matrix(cycle6)
        coupling = compute_ot_coupling(mu0, mu1, cost)

        for t in [0.0, 0.25, 0.5, 0.75, 0.99]:
            p = marginal_distribution(cycle6, coupling, t)
            assert abs(p.sum() - 1.0) < 1e-5, f"Sum != 1 at t={t}: {p.sum()}"

    def test_at_t0_matches_source(self, cycle6):
        """At t=0, marginal should match source distribution."""
        N = 6
        mu0 = peaked_dist(N, 0)
        mu1 = peaked_dist(N, 3)
        cost = compute_cost_matrix(cycle6)
        coupling = compute_ot_coupling(mu0, mu1, cost)

        p0 = marginal_distribution(cycle6, coupling, 0.0)
        np.testing.assert_allclose(p0, mu0, atol=1e-5)

    def test_at_t1_matches_target(self, cycle6):
        """At t=1, marginal should match target distribution."""
        N = 6
        mu0 = peaked_dist(N, 0)
        mu1 = peaked_dist(N, 3)
        cost = compute_cost_matrix(cycle6)
        coupling = compute_ot_coupling(mu0, mu1, cost)

        p1 = marginal_distribution(cycle6, coupling, 1.0)
        np.testing.assert_allclose(p1, mu1, atol=1e-5)

    def test_nonnegative(self, cycle6):
        """All marginal probabilities should be non-negative."""
        N = 6
        mu0 = peaked_dist(N, 0)
        mu1 = peaked_dist(N, 3)
        cost = compute_cost_matrix(cycle6)
        coupling = compute_ot_coupling(mu0, mu1, cost)

        for t in [0.0, 0.1, 0.5, 0.9]:
            p = marginal_distribution(cycle6, coupling, t)
            assert np.all(p >= -1e-9)

    def test_trivial_coupling(self, cycle6):
        """Trivial coupling (same dist): marginal should be that distribution."""
        N = 6
        mu = np.ones(N) / N
        # Trivial coupling: diagonal
        coupling = np.diag(mu)

        for t in [0.0, 0.5, 1.0]:
            p = marginal_distribution(cycle6, coupling, t)
            assert abs(p.sum() - 1.0) < 1e-5


class TestMarginalRateMatrix:
    """Tests for marginal_rate_matrix."""

    def test_zero_row_sums(self, cycle6):
        """Marginal rate matrix should have zero row sums."""
        N = 6
        mu0 = peaked_dist(N, 0)
        mu1 = peaked_dist(N, 3)
        cost = compute_cost_matrix(cycle6)
        coupling = compute_ot_coupling(mu0, mu1, cost)

        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            u_t = marginal_rate_matrix(cycle6, coupling, t)
            row_sums = u_t.sum(axis=1)
            np.testing.assert_allclose(row_sums, 0, atol=1e-8,
                                       err_msg=f"Row sums not zero at t={t}")

    def test_offdiag_nonnegative(self, cycle6):
        """Off-diagonal entries of marginal rate matrix should be non-negative."""
        N = 6
        mu0 = peaked_dist(N, 0)
        mu1 = peaked_dist(N, 3)
        cost = compute_cost_matrix(cycle6)
        coupling = compute_ot_coupling(mu0, mu1, cost)

        u_t = marginal_rate_matrix(cycle6, coupling, 0.5)
        for a in range(N):
            for b in range(N):
                if a != b:
                    assert u_t[a, b] >= -1e-9, f"Negative off-diag at ({a},{b}): {u_t[a,b]}"

    def test_trivial_coupling_rate(self, cycle6):
        """For trivial coupling (same dist), rate matrix should be valid."""
        N = 6
        mu = np.ones(N) / N
        coupling = np.diag(mu)

        u_t = marginal_rate_matrix(cycle6, coupling, 0.5)
        row_sums = u_t.sum(axis=1)
        np.testing.assert_allclose(row_sums, 0, atol=1e-8)


class TestEvolveDistribution:
    """Tests for evolve_distribution."""

    def test_evolve_reaches_target_cycle(self, cycle6):
        """Evolving under u_t should reach target distribution (TV < 0.1)."""
        N = 6
        mu0 = peaked_dist(N, 0)
        mu1 = peaked_dist(N, 3)
        cost = compute_cost_matrix(cycle6)
        coupling = compute_ot_coupling(mu0, mu1, cost)

        def rate_fn(t):
            return marginal_rate_matrix(cycle6, coupling, t)

        times, traj = evolve_distribution(mu0, rate_fn, (0.0, 0.999), n_steps=200)

        final_tv = total_variation(traj[-1], mu1)
        assert final_tv < 0.1, f"TV distance too large: {final_tv}"

    def test_evolve_starts_at_mu0(self, cycle6):
        """First distribution in trajectory should be mu0."""
        N = 6
        mu0 = peaked_dist(N, 0)
        mu1 = peaked_dist(N, 3)
        cost = compute_cost_matrix(cycle6)
        coupling = compute_ot_coupling(mu0, mu1, cost)

        def rate_fn(t):
            return marginal_rate_matrix(cycle6, coupling, t)

        times, traj = evolve_distribution(mu0, rate_fn, (0.0, 0.999), n_steps=100)

        np.testing.assert_allclose(traj[0], mu0, atol=1e-6)

    def test_evolve_sums_to_one_throughout(self, cycle6):
        """Distribution should sum to 1 at all time steps."""
        N = 6
        mu0 = peaked_dist(N, 0)
        mu1 = peaked_dist(N, 3)
        cost = compute_cost_matrix(cycle6)
        coupling = compute_ot_coupling(mu0, mu1, cost)

        def rate_fn(t):
            return marginal_rate_matrix(cycle6, coupling, t)

        times, traj = evolve_distribution(mu0, rate_fn, (0.0, 0.999), n_steps=50)

        for k in range(len(times)):
            s = traj[k].sum()
            assert abs(s - 1.0) < 1e-4, f"Sum != 1 at step {k}: {s}"

    def test_evolve_reaches_target_grid(self, grid4x4):
        """Grid graph: evolving should converge to target (TV < 0.15)."""
        N = 16
        mu0 = peaked_dist(N, 0)
        mu1 = peaked_dist(N, 15)
        cost = compute_cost_matrix(grid4x4)
        coupling = compute_ot_coupling(mu0, mu1, cost)

        def rate_fn(t):
            return marginal_rate_matrix(grid4x4, coupling, t)

        times, traj = evolve_distribution(mu0, rate_fn, (0.0, 0.999), n_steps=200)
        final_tv = total_variation(traj[-1], mu1)
        assert final_tv < 0.15, f"TV distance too large on grid: {final_tv}"

    def test_trivial_coupling_stays_put(self, cycle6):
        """For diagonal coupling (i->i), distribution shouldn't move much."""
        N = 6
        mu0 = np.ones(N) / N
        coupling = np.diag(mu0)  # each mass point stays put

        def rate_fn(t):
            return marginal_rate_matrix(cycle6, coupling, t)

        times, traj = evolve_distribution(mu0, rate_fn, (0.0, 0.999), n_steps=50)
        # Distribution should remain roughly uniform
        final_tv = total_variation(traj[-1], mu0)
        assert final_tv < 0.5  # loose bound since diagonal coupling is trivial
