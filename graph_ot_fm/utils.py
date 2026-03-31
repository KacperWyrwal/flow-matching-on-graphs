"""Backward-compatible shim: re-exports from otfm.core.utils."""
from otfm.core.utils import (
    make_cycle_graph,
    make_grid_graph,
    make_path_graph,
    make_star_graph,
    make_complete_bipartite_graph,
    make_barbell_graph,
    make_petersen_graph,
    make_cube_graph,
    cube_boundary_mask,
    cube_node_depth,
    kl_divergence,
    total_variation,
)
