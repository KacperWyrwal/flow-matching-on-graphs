"""Backward-compatible shim: re-exports from otfm.train and otfm.core."""
from otfm.core.utils import EMA
from otfm.core.loss import rate_kl_divergence, mse_loss
from otfm.train.graph_marginal import (
    train,
    train_conditional,
    train_flexible_conditional,
)
from otfm.train.distribution import train_film_conditional
from otfm.train.direct import train_direct_gnn
