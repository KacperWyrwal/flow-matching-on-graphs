"""
Unified Optimal Transport Flow Matching (OTFM) package.

Reorganizes graph_ot_fm, meta_fm, and config_fm into a single cohesive package.
"""

# Core utilities
from otfm.core.utils import EMA, total_variation, kl_divergence, get_device
from otfm.core.loss import rate_kl_divergence, mse_loss, rate_kl_loss

# Graph structures
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

# Models
from otfm.models.predictor import (
    FlexibleConditionalGNNRateMatrixPredictor,
    FiLMConditionalGNNRateMatrixPredictor,
    DirectGNNPredictor,
    EdgeAwareDirectGNNPredictor,
    ConfigurationRatePredictor,
)

# Training
from otfm.train.graph_marginal import train_flexible_conditional
from otfm.train.distribution import train_film_conditional
from otfm.train.direct import train_direct_gnn
from otfm.train.configuration import train_configuration_fm
