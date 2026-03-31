"""Johnson graph J(n,k): binary strings with fixed Hamming weight.

From config_fm/spaces/johnson.py.
"""

import numpy as np
from otfm.configuration.spaces.base import ConfigurationSpace


class JohnsonSpace(ConfigurationSpace):
    """Configuration space for J(n,k).

    Position graph: complete graph K_n (with J coupling as edge features).
    Vocabulary: {0, 1}
    Invariant: sum(labels) = k
    Transitions: swap one 1-position with one 0-position (order=2)
    """

    def __init__(self, n, k, J_coupling, h_field, beta_range=(0.5, 2.0)):
        self.n = n
        self.k = k
        self.J = J_coupling       # (n, n)
        self.h = h_field           # (n,)
        self.beta_range = beta_range
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
        return 2

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
            self._edge_features = feats[:, None]  # (E, 1)
        return self._edge_features

    def node_features(self, config):
        """[x_i, h_i] per node."""
        return np.stack([config, self.h], axis=-1).astype(np.float32)

    def global_features(self, t=0.0, beta=1.0, **kwargs):
        """[t, beta]."""
        return np.array([t, beta], dtype=np.float32)

    def transition_mask(self, config):
        """(n, n) mask: 1 where config[i]=1 and config[j]=0."""
        return (config[:, None] * (1 - config[None, :])).astype(np.float32)

    def apply_transition(self, config, transition_idx):
        i = transition_idx // self.n
        j = transition_idx % self.n
        if config[i] == 1 and config[j] == 0:
            new = config.copy()
            new[i] = 0.0
            new[j] = 1.0
            return new
        return None

    def geodesic_distance(self, config_a, config_b):
        return int(np.sum((config_a == 1) & (config_b == 0)))

    def sample_intermediate(self, config_0, config_T, t, rng):
        S_plus = np.where((config_0 == 1) & (config_T == 0))[0]
        S_minus = np.where((config_0 == 0) & (config_T == 1))[0]
        d = len(S_plus)

        if d == 0:
            return config_0.copy(), 0, 0

        ell = int(rng.binomial(d, min(t, 0.999)))

        A = rng.choice(S_plus, size=ell, replace=False) if ell > 0 else []
        B = rng.choice(S_minus, size=ell, replace=False) if ell > 0 else []

        config_t = config_0.copy()
        if len(A) > 0:
            config_t[A] = 0.0
        if len(B) > 0:
            config_t[B] = 1.0

        return config_t, ell, d - ell

    def compute_target_rates(self, config_0, config_T, config_t, t):
        S_plus_rem = np.where((config_t == 1) & (config_T == 0))[0]
        S_minus_rem = np.where((config_t == 0) & (config_T == 1))[0]
        d_rem = len(S_plus_rem)

        rates = np.zeros((self.n, self.n), dtype=np.float32)
        if d_rem > 0:
            rate = 1.0 / d_rem
            for i in S_plus_rem:
                for j in S_minus_rem:
                    rates[i, j] = rate
        return rates

    def sample_source(self, rng):
        config = np.zeros(self.n, dtype=np.float32)
        ones = rng.choice(self.n, size=self.k, replace=False)
        config[ones] = 1.0
        return config

    def sample_target(self, rng, beta=1.0, mcmc_pool=None, **kwargs):
        if mcmc_pool is not None:
            return mcmc_pool[int(rng.integers(len(mcmc_pool)))].copy()
        from johnson_fm.energy import mcmc_kawasaki, ising_energy
        energy_fn = lambda x: ising_energy(x, self.J, self.h)
        return mcmc_kawasaki(energy_fn, self.n, self.k, beta, 5000, rng)
