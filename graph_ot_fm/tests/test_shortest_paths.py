"""Tests for shortest_paths module."""
import numpy as np
import pytest
from graph_ot_fm.shortest_paths import compute_shortest_paths_and_geodesics


def _make_path_graph(n):
    """Path graph P_n: 0-1-2-..-(n-1), unit rates."""
    R = np.zeros((n, n))
    for i in range(n - 1):
        R[i, i + 1] = 1.0
        R[i + 1, i] = 1.0
    return R


def _make_cycle_graph(n):
    """Cycle graph C_n: 0-1-..-(n-1)-0, unit rates."""
    R = np.zeros((n, n))
    for i in range(n):
        R[i, (i + 1) % n] = 1.0
        R[(i + 1) % n, i] = 1.0
    return R


def _make_complete_graph(n, rate=1.0):
    """Complete graph K_n with uniform rate."""
    R = np.full((n, n), rate)
    np.fill_diagonal(R, 0.0)
    return R


class TestPathGraph:
    def test_distances_p4(self):
        R = _make_path_graph(4)
        dist, log_geo = compute_shortest_paths_and_geodesics(R, metric="hop")
        assert dist[0, 3] == 3.0
        assert dist[0, 1] == 1.0
        assert dist[1, 3] == 2.0

    def test_single_geodesic_p4(self):
        R = _make_path_graph(4)
        dist, log_geo = compute_shortest_paths_and_geodesics(R, metric="hop")
        # Only one geodesic path 0->1->2->3
        # N_{0,3} = R[0,1]*R[1,2]*R[2,3] = 1*1*1 = 1
        assert np.isclose(np.exp(log_geo[0, 3]), 1.0)

    def test_diagonal_is_one(self):
        R = _make_path_graph(4)
        dist, log_geo = compute_shortest_paths_and_geodesics(R, metric="hop")
        for i in range(4):
            assert np.isclose(log_geo[i, i], 0.0)  # log(1) = 0


class TestCycleGraph:
    def test_distances_c4(self):
        R = _make_cycle_graph(4)
        dist, log_geo = compute_shortest_paths_and_geodesics(R, metric="hop")
        assert dist[0, 2] == 2.0  # two hops either way

    def test_two_geodesics_c4(self):
        R = _make_cycle_graph(4)
        dist, log_geo = compute_shortest_paths_and_geodesics(R, metric="hop")
        # Two geodesic paths 0->1->2 and 0->3->2
        # N_{0,2} = R[0,1]*R[1,2] + R[0,3]*R[3,2] = 1+1 = 2
        assert np.isclose(np.exp(log_geo[0, 2]), 2.0)

    def test_adjacent_one_geodesic_c4(self):
        R = _make_cycle_graph(4)
        dist, log_geo = compute_shortest_paths_and_geodesics(R, metric="hop")
        # Adjacent nodes: one geodesic
        assert np.isclose(np.exp(log_geo[0, 1]), 1.0)


class TestCompleteGraph:
    def test_distances_k4(self):
        R = _make_complete_graph(4)
        dist, log_geo = compute_shortest_paths_and_geodesics(R, metric="hop")
        # All non-diagonal distances = 1
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert dist[i, j] == 1.0

    def test_geodesic_count_k4(self):
        R = _make_complete_graph(4)
        dist, log_geo = compute_shortest_paths_and_geodesics(R, metric="hop")
        # Distance 1, single direct edge, N_ij = R[i,j] = 1
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert np.isclose(np.exp(log_geo[i, j]), 1.0)


class TestAsymmetricGraph:
    def test_rate_metric_asymmetric(self):
        # Directed graph with rates < 1 (so -log gives positive weights)
        R = np.array([
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.3],
            [0.8, 0.0, 0.0],
        ])
        dist, log_geo = compute_shortest_paths_and_geodesics(R, metric="rate")
        # Edge weights: -log(0.5), -log(0.3), -log(0.8) (all positive)
        assert np.isclose(dist[0, 1], -np.log(0.5))
        assert np.isclose(dist[1, 2], -np.log(0.3))
        assert np.isclose(dist[2, 0], -np.log(0.8))

    def test_rate_metric_longer_path(self):
        R = np.array([
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.3],
            [0.8, 0.0, 0.0],
        ])
        dist, log_geo = compute_shortest_paths_and_geodesics(R, metric="rate")
        # d(0,2) should go via 0->1->2: -log(0.5) + -log(0.3)
        assert np.isclose(dist[0, 2], -np.log(0.5) + (-np.log(0.3)))


class TestConsistencyWithMatrixPower:
    def test_matches_matrix_power_path(self):
        """log_geodesic_counts should match np.log of matrix power."""
        R = _make_path_graph(5)
        R_off = R.copy()
        np.fill_diagonal(R_off, 0.0)

        dist, log_geo = compute_shortest_paths_and_geodesics(R, metric="hop")

        # Check via matrix power
        R_pow = np.eye(5)
        for d in range(5):
            for i in range(5):
                for j in range(5):
                    if np.isclose(dist[i, j], d) and R_pow[i, j] > 0:
                        assert np.isclose(np.exp(log_geo[i, j]), R_pow[i, j]), \
                            f"Mismatch at ({i},{j}), d={d}: got {np.exp(log_geo[i,j])}, expected {R_pow[i,j]}"
            R_pow = R_pow @ R_off

    def test_matches_matrix_power_cycle(self):
        R = _make_cycle_graph(6)
        R_off = R.copy()
        np.fill_diagonal(R_off, 0.0)

        dist, log_geo = compute_shortest_paths_and_geodesics(R, metric="hop")

        R_pow = np.eye(6)
        for d in range(4):  # max hop distance on C6 is 3
            for i in range(6):
                for j in range(6):
                    if np.isclose(dist[i, j], d) and R_pow[i, j] > 0:
                        assert np.isclose(np.exp(log_geo[i, j]), R_pow[i, j], rtol=1e-10), \
                            f"Mismatch at ({i},{j}), d={d}"
            R_pow = R_pow @ R_off
