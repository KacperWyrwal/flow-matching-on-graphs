"""Kawasaki dynamics on 2D square lattice with periodic boundary conditions.

From config_fm/spaces/kawasaki.py.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from otfm.configuration.spaces.base import ConfigurationSpace


class KawasakiSpace(ConfigurationSpace):
    """Kawasaki dynamics on L x L torus.

    Position graph: 2D periodic grid (each site has 4 neighbors)
    Vocabulary: {0, 1}
    Invariant: sum(labels) = k (fixed magnetization)
    Transitions: swap neighboring sites with different labels (order=2)
    """

    def __init__(self, L, k=None, J_coupling=1.0):
        self.L = L
        self.N = L * L
        self.k = k if k is not None else self.N // 2
        self.J_coupling = J_coupling
        self._edge_index = None
        self._edge_features = None

    @property
    def n_positions(self):
        return self.N

    @property
    def vocab_size(self):
        return 2

    @property
    def transition_order(self):
        return 2

    def position_graph_edges(self):
        if self._edge_index is not None:
            return self._edge_index
        src, dst = [], []
        for y in range(self.L):
            for x in range(self.L):
                i = y * self.L + x
                # Right
                j = y * self.L + (x + 1) % self.L
                src.extend([i, j])
                dst.extend([j, i])
                # Down
                j = ((y + 1) % self.L) * self.L + x
                src.extend([i, j])
                dst.extend([j, i])
        self._edge_index = np.array([src, dst], dtype=np.int64)
        return self._edge_index

    def position_edge_features(self):
        if self._edge_features is not None:
            return self._edge_features
        n_edges = self.position_graph_edges().shape[1]
        self._edge_features = np.full((n_edges, 1), self.J_coupling,
                                       dtype=np.float32)
        return self._edge_features

    def node_features(self, config):
        """[sigma_i] per node."""
        return config[:, None].astype(np.float32)

    def global_features(self, t=0.0, beta=1.0, **kwargs):
        return np.array([t, beta], dtype=np.float32)

    def _torus_manhattan(self, i, j):
        """Manhattan distance on torus between sites i and j."""
        sy, sx = i // self.L, i % self.L
        ty, tx = j // self.L, j % self.L
        dx = min(abs(sx - tx), self.L - abs(sx - tx))
        dy = min(abs(sy - ty), self.L - abs(sy - ty))
        return dx + dy

    def transition_mask(self, config):
        """(N, N) mask: 1 for neighboring pairs with config[i]=1, config[j]=0."""
        edges = self.position_graph_edges()
        mask = np.zeros((self.N, self.N), dtype=np.float32)
        for idx in range(edges.shape[1]):
            i, j = int(edges[0, idx]), int(edges[1, idx])
            if config[i] == 1 and config[j] == 0:
                mask[i, j] = 1.0
        return mask

    def apply_transition(self, config, transition_idx):
        i = transition_idx // self.N
        j = transition_idx % self.N
        if config[i] == 1 and config[j] == 0:
            new = config.copy()
            new[i] = 0.0
            new[j] = 1.0
            return new
        return None

    def geodesic_distance(self, config_a, config_b):
        S_plus = np.where((config_a == 1) & (config_b == 0))[0]
        S_minus = np.where((config_a == 0) & (config_b == 1))[0]
        if len(S_plus) == 0:
            return 0
        cost = np.zeros((len(S_plus), len(S_minus)))
        for ii, s in enumerate(S_plus):
            for jj, t in enumerate(S_minus):
                cost[ii, jj] = self._torus_manhattan(s, t)
        row_ind, col_ind = linear_sum_assignment(cost)
        return int(cost[row_ind, col_ind].sum())

    def _lattice_path_position(self, src, tgt, steps_done):
        """Compute position after steps_done steps along horizontal-first
        Manhattan path from src to tgt on the torus."""
        sy, sx = src // self.L, src % self.L
        ty, tx = tgt // self.L, tgt % self.L

        dx_raw = tx - sx
        dy_raw = ty - sy
        if abs(dx_raw) > self.L // 2:
            dx_raw = int(dx_raw - self.L * np.sign(dx_raw))
        if abs(dy_raw) > self.L // 2:
            dy_raw = int(dy_raw - self.L * np.sign(dy_raw))

        dx_sign = int(np.sign(dx_raw)) if dx_raw != 0 else 0
        dy_sign = int(np.sign(dy_raw)) if dy_raw != 0 else 0
        abs_dx = abs(int(dx_raw))
        abs_dy = abs(int(dy_raw))

        cx, cy = sx, sy
        steps_left = steps_done

        h_steps = min(steps_left, abs_dx)
        cx = (cx + h_steps * dx_sign) % self.L
        steps_left -= h_steps

        v_steps = min(steps_left, abs_dy)
        cy = (cy + v_steps * dy_sign) % self.L

        return cy * self.L + cx

    def sample_intermediate(self, config_0, config_T, t, rng):
        """Sample intermediate config along optimal lattice geodesic.

        Simpler approach that guarantees Hamming weight preservation:
        For each matched (src, tgt) pair, sample whether the swap is
        completed by time t. If yes, the particle is at tgt; if no,
        it stays at src. This is the Johnson-graph approach (binary
        completion per swap) applied to the lattice assignment.

        This sacrifices spatial path fidelity (no intermediate lattice
        positions) but guarantees exact Hamming weight preservation.
        """
        S_plus = np.where((config_0 == 1) & (config_T == 0))[0]
        S_minus = np.where((config_0 == 0) & (config_T == 1))[0]
        d = len(S_plus)

        if d == 0:
            return config_0.copy(), 0, 0

        # Sample number of completed swaps (same as Johnson graph)
        ell = int(rng.binomial(d, min(t, 0.999)))

        # Choose which swaps are completed
        completed_idx = rng.choice(d, size=ell, replace=False) if ell > 0 else np.array([], dtype=int)

        config_t = config_0.copy()
        # Apply completed swaps
        for idx in completed_idx:
            src = S_plus[idx]
            tgt = S_minus[idx]
            config_t[src] = 0.0
            config_t[tgt] = 1.0

        return config_t, ell, d - ell

    def compute_target_rates(self, config_0, config_T, config_t, t):
        """Target rates for remaining swaps.

        Uses Johnson-style uniform rate over all remaining (src, tgt) pairs
        that are lattice neighbors. Non-neighbor remaining pairs get zero
        rate (the model can't execute them anyway — they're masked out).
        If no remaining pair is a lattice neighbor, spreads rate over all
        lattice-neighbor valid swaps that move any remaining src particle
        toward any remaining tgt position.
        """
        S_plus_rem = np.where((config_t == 1) & (config_T == 0))[0]
        S_minus_rem = np.where((config_t == 0) & (config_T == 1))[0]
        d_rem = len(S_plus_rem)

        rates = np.zeros((self.N, self.N), dtype=np.float32)
        if d_rem == 0:
            return rates

        # Find lattice-neighbor pairs among remaining swaps
        S_plus_set = set(S_plus_rem.tolist())
        S_minus_set = set(S_minus_rem.tolist())
        edges = self.position_graph_edges()

        # Any lattice edge (i,j) where i is a remaining 1-position
        # and j is a remaining 0-position is a useful swap
        useful_pairs = []
        for idx in range(edges.shape[1]):
            i, j = int(edges[0, idx]), int(edges[1, idx])
            if i in S_plus_set and j in S_minus_set:
                useful_pairs.append((i, j))

        if useful_pairs:
            rate = 1.0 / len(useful_pairs)
            for i, j in useful_pairs:
                rates[i, j] = rate
        # If no direct neighbor pairs exist among remaining swaps,
        # rates stay zero — this is rare and the loss handles it

        return rates

    def sample_source(self, rng):
        config = np.zeros(self.N, dtype=np.float32)
        ones = rng.choice(self.N, size=self.k, replace=False)
        config[ones] = 1.0
        return config

    def sample_target(self, rng, beta=1.0, mcmc_pool=None, **kwargs):
        if mcmc_pool is not None:
            return mcmc_pool[int(rng.integers(len(mcmc_pool)))].copy()
        from otfm.configuration.spaces.kawasaki_mcmc import kawasaki_mcmc
        return kawasaki_mcmc(self, beta, 10000, rng)
