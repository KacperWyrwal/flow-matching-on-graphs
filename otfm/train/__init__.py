"""Training loops for flow matching on graphs."""

from otfm.train.graph_marginal import (
    train,
    train_conditional,
    train_flexible_conditional,
)
from otfm.train.distribution import train_film_conditional
from otfm.train.direct import train_direct_gnn
from otfm.train.configuration import train_configuration_fm
