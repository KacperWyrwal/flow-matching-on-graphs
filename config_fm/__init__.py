"""Backward-compatible shim: re-exports from otfm package."""

from otfm.configuration.spaces.base import ConfigurationSpace
from otfm.models.predictor import ConfigurationRatePredictor
from otfm.train.configuration import train_configuration_fm
from otfm.configuration.sample import generate_samples
from otfm.core.loss import rate_kl_loss
