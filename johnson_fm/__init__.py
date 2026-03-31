"""Johnson graph flow matching: sample-level CFM on J(n,k)."""

from johnson_fm.energy import ising_energy, mcmc_kawasaki, uniform_sample
from johnson_fm.flow import sample_intermediate, compute_target_rates
from johnson_fm.model import SwapRatePredictor
from johnson_fm.dfm_baseline import DFMBitFlipPredictor
