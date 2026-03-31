"""DFM on {0,1}^n: unconstrained binary strings (baseline).

From config_fm/spaces/dfm.py.
"""

import numpy as np
from otfm.configuration.spaces.base import ConfigurationSpace


class DFMSpace(ConfigurationSpace):
    """DFM on {0,1}^n (unconstrained binary strings).

    Position graph: complete graph K_n
    Vocabulary: {0, 1}
    Invariant: none
    Transitions: single bit flip (order=1)
    """

    def __init__(self, n, J_coupling, h_field):
        self.n = n
        self.J = J_coupling
        self.h = h_field
        self._edge_index = None
        self._edge_features = None

    @property
    def n_positions(self):
        return self.n

    @property
    def vocab_size(self):
        return 2

    @property
    def transition_order(self):
        return 1

    def position_graph_edges(self):
        if self._edge_index is None:
            src, dst = [], []
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        src.append(i)
                        dst.append(j)
            self._edge_index = np.array([src, dst], dtype=np.int64)
        return self._edge_index

    def position_edge_features(self):
        if self._edge_features is None:
            edges = self.position_graph_edges()
            feats = np.array([self.J[edges[0, e], edges[1, e]]
                              for e in range(edges.shape[1])],
                             dtype=np.float32)
            self._edge_features = feats[:, None]
        return self._edge_features

    def node_features(self, config):
        return np.stack([config, self.h], axis=-1).astype(np.float32)

    def global_features(self, t=0.0, beta=1.0, **kwargs):
        return np.array([t, beta], dtype=np.float32)

    def transition_mask(self, config):
        """(n,) mask: all positions can flip."""
        return np.ones(self.n, dtype=np.float32)

    def apply_transition(self, config, transition_idx):
        new = config.copy()
        new[transition_idx] = 1.0 - new[transition_idx]
        return new

    def geodesic_distance(self, config_a, config_b):
        return int(np.sum(config_a != config_b))

    def sample_intermediate(self, config_0, config_T, t, rng):
        diff = np.where(config_0 != config_T)[0]
        d = len(diff)
        if d == 0:
            return config_0.copy(), 0, 0
        ell = int(rng.binomial(d, min(t, 0.999)))
        flipped = rng.choice(diff, size=ell, replace=False) if ell > 0 else []
        config_t = config_0.copy()
        if len(flipped) > 0:
            config_t[flipped] = config_T[flipped]
        return config_t, ell, d - ell

    def compute_target_rates(self, config_0, config_T, config_t, t):
        diff_rem = np.where(config_t != config_T)[0]
        d_rem = len(diff_rem)
        rates = np.zeros(self.n, dtype=np.float32)
        if d_rem > 0:
            rates[diff_rem] = 1.0 / d_rem
        return rates

    def sample_source(self, rng):
        return rng.binomial(1, 0.5, size=self.n).astype(np.float32)

    def sample_target(self, rng, beta=1.0, mcmc_pool=None, **kwargs):
        if mcmc_pool is not None:
            return mcmc_pool[int(rng.integers(len(mcmc_pool)))].copy()
        raise ValueError("DFM needs precomputed target pool")
