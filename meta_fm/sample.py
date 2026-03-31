"""Backward-compatible shim: re-exports from otfm.graph.sample and otfm.distribution.sample."""
from otfm.graph.sample import (
    sample_trajectory,
    sample_trajectory_conditional,
    sample_trajectory_guided,
    sample_trajectory_flexible,
    sample_trajectory_film,
    backward_trajectory,
)
from otfm.distribution.sample import sample_posterior_film
