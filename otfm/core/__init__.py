"""Core utilities: loss functions, OT solvers, EMA, device detection."""

from otfm.core.utils import EMA, total_variation, kl_divergence, get_device
from otfm.core.loss import rate_kl_divergence, mse_loss, rate_kl_loss
from otfm.core.ot import (
    solve_w1,
    extract_optimal_face,
    solve_tiebreaker,
    SinkhornConvergenceError,
    compute_shortest_paths_and_geodesics,
)
