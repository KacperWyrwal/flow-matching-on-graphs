"""Backward-compatible shim: re-exports from otfm package."""

from otfm.core.utils import get_device, EMA
from otfm.distribution.dataset import (
    MetaFlowMatchingDataset,
    ConditionalMetaFlowMatchingDataset,
    InpaintingDataset,
    TopologyGeneralizationDataset,
    CubeBoundaryDataset,
    PosteriorSamplingDataset,
    CubePosteriorDataset,
)
from otfm.models.predictor import (
    RateMatrixPredictor,
    RateMatrixPredictorMLP,
    GNNRateMatrixPredictor,
    ConditionalGNNRateMatrixPredictor,
    FlexibleConditionalGNNRateMatrixPredictor,
    FiLMConditionalGNNRateMatrixPredictor,
    DirectGNNPredictor,
    EdgeAwareDirectGNNPredictor,
)
from otfm.models.conditioning import EdgeAwareMessagePassing
from otfm.graph.structure import rate_matrix_to_edge_index
from otfm.core.loss import rate_kl_divergence, mse_loss
from otfm.train.graph_marginal import (
    train,
    train_conditional,
    train_flexible_conditional,
)
from otfm.train.distribution import train_film_conditional
from otfm.train.direct import train_direct_gnn
from otfm.graph.sample import (
    sample_trajectory,
    sample_trajectory_conditional,
    sample_trajectory_guided,
    sample_trajectory_flexible,
    sample_trajectory_film,
    backward_trajectory,
)
from otfm.distribution.sample import sample_posterior_film
