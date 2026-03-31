"""Neural network models for flow matching on graphs."""

from otfm.models.backbone import (
    FiLMGNNBackbone,
    MessagePassingLayer,
    FiLMLayer,
)
from otfm.models.heads import (
    SingleNodeHead,
    PairwiseAttentionHead,
    EdgeRateHead,
)
from otfm.models.conditioning import (
    FiLMRateMessagePassing,
    RateMessagePassing,
    EdgeAwareMessagePassing,
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
    ConfigurationRatePredictor,
)
