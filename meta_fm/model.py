"""Backward-compatible shim: re-exports from otfm.models."""
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
from otfm.models.conditioning import (
    RateMessagePassing,
    FiLMRateMessagePassing,
    EdgeAwareMessagePassing,
)
from otfm.graph.structure import rate_matrix_to_edge_index
