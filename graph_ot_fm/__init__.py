"""Backward-compatible shim: re-exports from otfm package."""

from otfm.graph.structure import (
    GraphStructure,
    GeodesicCache,
    conditional_marginal,
    conditional_rate_matrix,
    sample_conditional_state,
)
from otfm.graph.coupling import (
    compute_cost_matrix,
    compute_ot_coupling,
    compute_ot_coupling_sinkhorn,
    compute_meta_cost_matrix_batch,
)
from otfm.graph.flow import (
    marginal_rate_matrix,
    marginal_distribution,
    marginal_rate_matrix_fast,
    marginal_distribution_fast,
    evolve_distribution,
)
from otfm.core.ot import (
    compute_shortest_paths_and_geodesics,
    solve_w1,
    extract_optimal_face,
    solve_tiebreaker,
    SinkhornConvergenceError,
)
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
