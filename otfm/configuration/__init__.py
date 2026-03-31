"""Configuration-level flow matching: spaces, sampling."""

from otfm.configuration.spaces.base import ConfigurationSpace
from otfm.configuration.spaces.johnson import JohnsonSpace
from otfm.configuration.spaces.kawasaki import KawasakiSpace
from otfm.configuration.spaces.kawasaki_mcmc import (
    kawasaki_mcmc,
    generate_kawasaki_pool,
    ising_energy_lattice,
    compute_kawasaki_dE,
    get_neighbors,
)
from otfm.configuration.spaces.dfm import DFMSpace
from otfm.configuration.sample import generate_samples
