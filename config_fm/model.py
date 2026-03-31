"""Backward-compatible shim: re-exports from otfm.models."""
from otfm.models.backbone import FiLMGNNBackbone, MessagePassingLayer, FiLMLayer
from otfm.models.heads import SingleNodeHead, PairwiseAttentionHead
from otfm.models.predictor import ConfigurationRatePredictor
