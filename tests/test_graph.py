"""
Tests for graph.py: GraphStructure.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from graph_ot_fm import GraphStructure
from graph_ot_fm.utils import make_cycle_graph, make_grid_graph


class TestDistanceMatrix:
    """Tests for shortest path distance computation."""

    def test_cycle_graph_distances(self):
        """Cycle graph N=4: distances should be 0,1,2,1 from node 0."""
        R = make_cycle_graph(4, weighted=False)
        graph = GraphStructure(R)
        expected_from_0 = [0, 1, 2, 1]
        for j, expected in enumerate(expected_from_0):
            assert abs(graph.dist[0, j] - expected) < 1e-9, (
                f"dist[0,{j}] = {graph.dist[0,j]}, expected {expected}"
            )

    def test_cycle_graph_n8_distances(self):
        """Cycle N=8: max distance should be 4 (diameter)."""
        R = make_cycle_graph(8, weighted=False)
        graph = GraphStructure(R)
        finite_dists = graph.dist[np.isfinite(graph.dist)]
        assert int(np.max(finite_dists)) == 4

    def test_grid_graph_distances(self):
        """Grid 3x3: distance from corner (0) to corner (8) should be 4."""
        R = make_grid_graph(3, 3, weighted=False)
        graph = GraphStructure(R)
        # Node 0 (top-left) to node 8 (bottom-right) = 2+2 = 4 hops
        assert abs(graph.dist[0, 8] - 4.0) < 1e-9

    def test_grid_graph_adjacent_nodes(self):
        """Adjacent nodes in grid should have distance 1."""
        R = make_grid_graph(3, 3, weighted=False)
        graph = GraphStructure(R)
        # Node 0 and 1 are adjacent (same row)
        assert abs(graph.dist[0, 1] - 1.0) < 1e-9
        # Node 0 and 3 are adjacent (same column)
        assert abs(graph.dist[0, 3] - 1.0) < 1e-9

    def test_symmetry(self):
        """dist[i,j] == dist[j,i] for undirected graphs."""
        R = make_cycle_graph(6, weighted=False)
        graph = GraphStructure(R)
        for i in range(6):
            for j in range(6):
                assert abs(graph.dist[i, j] - graph.dist[j, i]) < 1e-9, (
                    f"Symmetry violated: dist[{i},{j}]={graph.dist[i,j]}, dist[{j},{i}]={graph.dist[j,i]}"
                )

    def test_symmetry_grid(self):
        """dist[i,j] == dist[j,i] for grid graph."""
        R = make_grid_graph(4, 4, weighted=False)
        graph = GraphStructure(R)
        for i in range(16):
            for j in range(16):
                assert abs(graph.dist[i, j] - graph.dist[j, i]) < 1e-9

    def test_self_distance_zero(self):
        """dist[i,i] == 0."""
        R = make_cycle_graph(5, weighted=False)
        graph = GraphStructure(R)
        for i in range(5):
            assert graph.dist[i, i] == 0.0

    def test_all_nodes_reachable_cycle(self):
        """All nodes reachable in cycle graph."""
        R = make_cycle_graph(6, weighted=False)
        graph = GraphStructure(R)
        assert np.all(np.isfinite(graph.dist))


class TestGeodesicCount:
    """Tests for geodesic_count computation."""

    def test_geodesic_count_self(self):
        """geodesic_count[a,a] should be 1."""
        R = make_cycle_graph(4, weighted=False)
        graph = GraphStructure(R)
        for a in range(4):
            assert graph.geodesic_count[a, a] == 1.0

    def test_geodesic_count_adjacent(self):
        """For adjacent nodes: geodesic_count[a,j] = R[a,j]."""
        R = make_cycle_graph(4, weighted=False)
        graph = GraphStructure(R)
        # Node 0 -> node 1 (distance 1)
        expected = R[0, 1]  # = 1.0 for unweighted
        assert abs(graph.geodesic_count[0, 1] - expected) < 1e-9

    def test_geodesic_count_recursion_cycle4(self):
        """
        Cycle N=4: d(0,2)=2. The geodesic paths are 0->1->2 and 0->3->2.
        geodesic_count[0,2] = (R_offdiag^2)[0,2].
        For unweighted cycle: R^2[0,2] = R[0,1]*R[1,2] + R[0,3]*R[3,2] = 1+1 = 2.
        """
        R = make_cycle_graph(4, weighted=False)
        graph = GraphStructure(R)
        # dist[0,2] = 2
        assert abs(graph.dist[0, 2] - 2.0) < 1e-9
        # Two paths: 0->1->2 and 0->3->2
        # R_offdiag^2[0,2] = R[0,1]*R[1,2] + R[0,3]*R[3,2] = 1*1 + 1*1 = 2
        R_off = R.copy()
        np.fill_diagonal(R_off, 0.0)
        R2 = R_off @ R_off
        expected = R2[0, 2]
        assert abs(graph.geodesic_count[0, 2] - expected) < 1e-9

    def test_geodesic_count_cycle6(self):
        """Cycle N=6, node 0 to node 3 (distance 3): multiple paths."""
        R = make_cycle_graph(6, weighted=False)
        graph = GraphStructure(R)
        assert abs(graph.dist[0, 3] - 3.0) < 1e-9
        R_off = R.copy()
        np.fill_diagonal(R_off, 0.0)
        R3 = R_off @ R_off @ R_off
        expected = R3[0, 3]
        assert abs(graph.geodesic_count[0, 3] - expected) < 1e-9


class TestBranchingProbs:
    """Tests for branching_probs."""

    def test_branching_probs_sum_to_one(self):
        """Branching probs must sum to 1 for all (a,j) pairs with finite distance."""
        R = make_cycle_graph(6, weighted=False)
        graph = GraphStructure(R)
        for a in range(6):
            for j in range(6):
                if a != j and np.isfinite(graph.dist[a, j]):
                    probs = graph.branching_probs(a, j)
                    if probs:
                        total = sum(probs.values())
                        assert abs(total - 1.0) < 1e-6, (
                            f"Branching probs don't sum to 1 for a={a}, j={j}: {total}"
                        )

    def test_branching_probs_sum_to_one_grid(self):
        """Branching probs must sum to 1 for grid graph."""
        R = make_grid_graph(3, 3, weighted=False)
        graph = GraphStructure(R)
        for a in range(9):
            for j in range(9):
                if a != j and np.isfinite(graph.dist[a, j]):
                    probs = graph.branching_probs(a, j)
                    if probs:
                        total = sum(probs.values())
                        assert abs(total - 1.0) < 1e-6

    def test_branching_probs_move_closer(self):
        """Branching probs should only assign mass to nodes closer to j."""
        R = make_cycle_graph(8, weighted=False)
        graph = GraphStructure(R)
        # From node 0 toward node 4
        probs = graph.branching_probs(0, 4)
        for b in probs:
            d_bj = graph.dist[b, 4]
            d_aj = graph.dist[0, 4]
            assert d_bj < d_aj, f"Node {b} is not closer to 4 than node 0"

    def test_branching_probs_nonnegative(self):
        """All branching probs should be non-negative."""
        R = make_cycle_graph(6, weighted=False)
        graph = GraphStructure(R)
        for a in range(6):
            for j in range(6):
                if a != j:
                    probs = graph.branching_probs(a, j)
                    for b, p in probs.items():
                        assert p >= 0, f"Negative prob at a={a}, j={j}, b={b}: {p}"

    def test_adjacent_nodes_deterministic(self):
        """For adjacent nodes (d=1), branching prob is deterministic."""
        R = make_cycle_graph(4, weighted=False)
        graph = GraphStructure(R)
        # Node 0 -> node 1 (distance 1): only one path
        probs = graph.branching_probs(0, 1)
        assert len(probs) == 1
        assert 1 in probs
        assert abs(probs[1] - 1.0) < 1e-9
