"""Graph-level flow matching: structures, flows, couplings, sampling."""

from otfm.graph.structure import (
    GraphStructure,
    GeodesicCache,
    rate_matrix_to_edge_index,
)
from otfm.graph.flow import (
    marginal_distribution,
    marginal_distribution_fast,
    marginal_rate_matrix,
    marginal_rate_matrix_fast,
    evolve_distribution,
)
from otfm.graph.coupling import (
    compute_cost_matrix,
    compute_ot_coupling,
    compute_ot_coupling_sinkhorn,
    compute_meta_cost_matrix_batch,
)
from otfm.graph.sample import (
    sample_trajectory_flexible,
    sample_trajectory_film,
)
