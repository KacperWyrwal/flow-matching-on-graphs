"""Degree-preserving graph generation via double edge swaps."""

import numpy as np
from collections import defaultdict

from otfm.configuration.spaces.base import ConfigurationSpace


class DegreeSequenceSpace(ConfigurationSpace):
    """Fixed degree sequence graph generation.

    Config is the flattened upper-triangle of the adjacency matrix:
    n*(n-1)/2 binary edge variables.

    Transitions: double edge swap (k=4 on edge variables).
    Remove edges (a,b) and (c,d), add edges (a,c) and (b,d)
    [or (a,d) and (b,c)], preserving every node's degree.
    """

    def __init__(self, n, degree_sequence, communities=None,
                 betas=None, pools=None):
        self.n = n
        self.degree_seq = np.asarray(degree_sequence, dtype=int)
        self.communities = communities  # (n,) int array or None
        self.n_potential_edges = n * (n - 1) // 2
        self.betas = betas
        self.pools = pools
        self._edge_index = None

    @property
    def n_positions(self):
        return self.n

    @property
    def vocab_size(self):
        return 2

    @property
    def transition_order(self):
        return 4

    def position_graph_edges(self):
        """Complete graph K_n."""
        if self._edge_index is not None:
            return self._edge_index
        src, dst = [], []
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    src.append(i)
                    dst.append(j)
        self._edge_index = np.array([src, dst], dtype=np.int64)
        return self._edge_index

    def position_edge_features(self):
        return None

    def _upper_tri_indices(self):
        """Cached upper-triangle index arrays."""
        if not hasattr(self, '_ut_i') or self._ut_i is None:
            self._ut_i, self._ut_j = np.triu_indices(self.n, k=1)
        return self._ut_i, self._ut_j

    def _config_to_adj(self, config):
        """Flat upper-triangle → full adjacency matrix (vectorized)."""
        A = np.zeros((self.n, self.n), dtype=np.float32)
        ui, uj = self._upper_tri_indices()
        A[ui, uj] = config
        A[uj, ui] = config
        return A

    def _adj_to_config(self, A):
        """Full adjacency → flat upper-triangle (vectorized)."""
        ui, uj = self._upper_tri_indices()
        return A[ui, uj].astype(np.float32)

    def _edge_pair_to_flat_idx(self, i, j):
        """Map edge (i,j) with i<j to flat index."""
        if i > j:
            i, j = j, i
        return i * self.n - i * (i + 1) // 2 + j - i - 1

    def node_features(self, config):
        """Node features: [degree/n, community_one_hot...]."""
        A = self._config_to_adj(config)
        degrees = A.sum(axis=1)
        feats = [degrees / self.n]
        if self.communities is not None:
            n_comm = int(self.communities.max()) + 1
            for c in range(n_comm):
                feats.append((self.communities == c).astype(np.float32))
        return np.stack(feats, axis=-1).astype(np.float32)

    def global_features(self, t=0.0, beta=1.0, **kwargs):
        return np.array([t, beta], dtype=np.float32)

    def dynamic_edge_features(self, config):
        """Dynamic edge features: [A_ij, same_community]."""
        A = self._config_to_adj(config)
        edges = self.position_graph_edges()
        a_ij = A[edges[0], edges[1]]
        if self.communities is not None:
            same_comm = (self.communities[edges[0]]
                         == self.communities[edges[1]]).astype(np.float32)
            return np.stack([a_ij, same_comm], axis=-1).astype(np.float32)
        return a_ij[:, None].astype(np.float32)

    def enumerate_transitions(self, config):
        """Enumerate all valid double edge swaps (vectorized numpy).

        Returns list of (a, b, c, d, rewiring) tuples.
        """
        A = self._config_to_adj(config)
        ei, ej = np.where(np.triu(A, k=1) > 0)
        edges = np.stack([ei, ej], axis=1)  # (m, 2)
        m = len(edges)
        if m < 2:
            return []

        # All pairs of edges
        idx1, idx2 = np.triu_indices(m, k=1)
        a = edges[idx1, 0]
        b = edges[idx1, 1]
        c = edges[idx2, 0]
        d = edges[idx2, 1]

        # All 4 nodes distinct
        distinct = (a != c) & (a != d) & (b != c) & (b != d)
        a, b, c, d = a[distinct], b[distinct], c[distinct], d[distinct]

        # Check rewirings via vectorized adjacency lookup
        can_ac_bd = (A[a, c] == 0) & (A[b, d] == 0)
        can_ad_bc = (A[a, d] == 0) & (A[b, c] == 0)

        swaps = []
        for i in np.where(can_ac_bd)[0]:
            swaps.append((int(a[i]), int(b[i]), int(c[i]), int(d[i]), 'ac_bd'))
        for i in np.where(can_ad_bc)[0]:
            swaps.append((int(a[i]), int(b[i]), int(c[i]), int(d[i]), 'ad_bc'))
        return swaps

    def apply_transition_by_descriptor(self, config, descriptor):
        """Apply a double edge swap."""
        a, b, c, d, rewiring = descriptor
        A = self._config_to_adj(config)
        A[a, b] = A[b, a] = 0
        A[c, d] = A[d, c] = 0
        if rewiring == 'ac_bd':
            A[a, c] = A[c, a] = 1
            A[b, d] = A[d, b] = 1
        else:
            A[a, d] = A[d, a] = 1
            A[b, c] = A[c, b] = 1
        return self._adj_to_config(A)

    def compute_target_rates_enumerated(self, config_0, config_T,
                                         config_t, t):
        """Target rates over enumerated transitions.

        Returns (transitions, rates) where transitions is from
        enumerate_transitions and rates is a parallel array.
        """
        transitions = self.enumerate_transitions(config_t)
        if not transitions:
            return transitions, np.array([], dtype=np.float32)

        A_t = self._config_to_adj(config_t)
        A_T = self._config_to_adj(config_T)

        # Vectorized symmetric difference
        diff = np.triu(A_t, k=1) - np.triu(A_T, k=1)
        ri, rj = np.where(diff > 0)
        ai, aj = np.where(diff < 0)
        r_set = set(zip(ri.tolist(), rj.tolist()))
        a_set = set(zip(ai.tolist(), aj.tolist()))

        # A swap is geodesic-progressing if it removes an R-edge and adds
        # an A-edge
        progressing = []
        for idx, (a, b, c, d, rewiring) in enumerate(transitions):
            removes = {(min(a, b), max(a, b)), (min(c, d), max(c, d))}
            if rewiring == 'ac_bd':
                adds = {(min(a, c), max(a, c)), (min(b, d), max(b, d))}
            else:
                adds = {(min(a, d), max(a, d)), (min(b, c), max(b, c))}
            # At least one removal is an R-edge and one addition is an A-edge
            if (removes & r_set) and (adds & a_set):
                progressing.append(idx)

        rates = np.zeros(len(transitions), dtype=np.float32)
        if progressing:
            rate = 1.0 / len(progressing)
            for idx in progressing:
                rates[idx] = rate

        return transitions, rates

    # ── Alternating cycle decomposition ───────────────────────────────��──────

    def find_alternating_cycles(self, config_0, config_T):
        """Decompose symmetric difference into alternating cycles.

        Returns (cycles, remove_edges, add_edges).
        Each cycle is a list of (u, v, edge_type, edge_id).
        """
        A0 = self._config_to_adj(config_0)
        AT = self._config_to_adj(config_T)

        # Vectorized symmetric difference
        diff = np.triu(A0, k=1) - np.triu(AT, k=1)
        ri, rj = np.where(diff > 0)
        ai, aj = np.where(diff < 0)
        remove = list(zip(ri.tolist(), rj.tolist()))
        add = list(zip(ai.tolist(), aj.tolist()))

        # Build alternating adjacency
        adj = defaultdict(list)
        for idx, (i, j) in enumerate(remove):
            adj[i].append((j, 'R', idx))
            adj[j].append((i, 'R', idx))
        for idx, (i, j) in enumerate(add):
            adj[i].append((j, 'A', idx))
            adj[j].append((i, 'A', idx))

        used_R = set()
        used_A = set()
        cycles = []

        for start in range(self.n):
            while True:
                # Find an unused R-edge at this node
                start_edge = None
                for (nb, etype, eid) in adj[start]:
                    if etype == 'R' and eid not in used_R:
                        start_edge = (nb, etype, eid)
                        break
                if start_edge is None:
                    break

                cycle = []
                current = start
                next_type = 'R'

                while True:
                    found = False
                    for (nb, etype, eid) in adj[current]:
                        if etype != next_type:
                            continue
                        if etype == 'R' and eid in used_R:
                            continue
                        if etype == 'A' and eid in used_A:
                            continue

                        if etype == 'R':
                            used_R.add(eid)
                        else:
                            used_A.add(eid)
                        cycle.append((current, nb, etype, eid))
                        current = nb
                        next_type = 'A' if next_type == 'R' else 'R'
                        found = True
                        break

                    if not found:
                        break
                    if current == start and next_type == 'R' and len(cycle) >= 2:
                        break

                if len(cycle) >= 4:
                    cycles.append(cycle)

        return cycles, remove, add

    def _resolve_cycle(self, cycle, remove, add):
        """Resolve an alternating cycle into double edge swaps.

        Each swap removes one R-edge and adds one A-edge from the cycle.
        A 2k-cycle needs k-1 swaps.
        Returns list of (remove_edge, add_edge) tuples.
        """
        r_edges = [(u, v, eid) for u, v, etype, eid in cycle if etype == 'R']
        a_edges = [(u, v, eid) for u, v, etype, eid in cycle if etype == 'A']

        swaps = []
        while len(r_edges) > 1 and len(a_edges) > 0:
            r = r_edges.pop(0)
            a = a_edges.pop(0)
            swaps.append((remove[r[2]], add[a[2]]))

        if r_edges and a_edges:
            swaps.append((remove[r_edges[0][2]], add[a_edges[0][2]]))

        return swaps

    def sample_geodesic(self, config_0, config_T, rng):
        """Sample a geodesic as a sequence of (remove_edge, add_edge) swaps."""
        cycles, remove, add = self.find_alternating_cycles(config_0, config_T)
        swaps = []
        for cycle in cycles:
            swaps.extend(self._resolve_cycle(cycle, remove, add))
        rng.shuffle(swaps)
        return swaps

    # ── ConfigurationSpace interface ─────────────────────────────────────────

    def geodesic_distance(self, config_a, config_b):
        cycles, _, _ = self.find_alternating_cycles(config_a, config_b)
        return sum(len(c) // 2 - 1 for c in cycles)

    def sample_intermediate(self, config_0, config_T, t, rng):
        """Sample intermediate by completing a random subset of alternating cycles.

        Each alternating cycle is either fully resolved or not. This
        guarantees exact degree preservation since resolving a full cycle
        preserves every node's degree.

        The number of completed cycles is Bin(n_cycles, t).
        """
        cycles, remove, add = self.find_alternating_cycles(config_0, config_T)
        n_cycles = len(cycles)

        if n_cycles == 0:
            return config_0.copy(), 0, 0

        # Sample number of completed cycles
        ell = int(rng.binomial(n_cycles, min(t, 0.999)))

        # Choose which cycles to complete
        completed_idx = (rng.choice(n_cycles, size=ell, replace=False)
                         if ell > 0 else [])

        A_t = self._config_to_adj(config_0.copy())
        total_swaps = 0
        total_remaining = 0

        for ci in range(n_cycles):
            cycle = cycles[ci]
            if ci in set(completed_idx if len(completed_idx) > 0 else []):
                # Resolve this cycle: remove all R-edges, add all A-edges
                for u, v, etype, eid in cycle:
                    if etype == 'R':
                        A_t[u, v] = A_t[v, u] = 0
                    else:
                        A_t[u, v] = A_t[v, u] = 1
                total_swaps += len(cycle) // 2
            else:
                total_remaining += len(cycle) // 2

        return self._adj_to_config(A_t), ell, n_cycles - ell

    def transition_mask(self, config):
        """(n, n) mask over node pairs.

        For k=4 the true output is over edge-pairs, but we use the
        PairwiseAttentionHead (n×n) as a proxy: predict rates for
        (existing_edge, non_edge) pairs, indexed by their shared-node
        structure. The mask marks node pairs (i,j) where a valid swap
        exists involving i and j.

        Specifically: mask[i,j]=1 if there exists a swap removing an edge
        incident to i and adding an edge incident to j (or vice versa).
        For simplicity, we use: mask[i,j]=1 if edge(i,j) exists (potential
        removal) OR edge(i,j) doesn't exist (potential addition), AND
        i != j. This is permissive — the model learns which swaps are good.
        """
        A = self._config_to_adj(config)
        # All node pairs (complete graph)
        mask = np.ones((self.n, self.n), dtype=np.float32)
        np.fill_diagonal(mask, 0)
        return mask

    def apply_transition(self, config, transition_idx):
        """Apply a swap indexed by flattened (n, n) output.

        The transition_idx encodes a node pair (i, j). We interpret this
        as: find the best valid swap involving nodes i and j, and execute it.

        For generation, we sample from the predicted rates (already masked),
        so this is called with specific (i, j). We attempt to find and
        execute a valid double swap where edge (i, ?) is removed and
        edge (j, ?) is added.
        """
        i = transition_idx // self.n
        j = transition_idx % self.n
        A = self._config_to_adj(config)

        # Try to find a valid double swap involving i and j
        # Case 1: edge (i, j) exists → remove it and add a non-edge
        if A[i, j] == 1:
            # Find another existing edge to pair with
            for a in range(self.n):
                for b in range(a + 1, self.n):
                    if a == i and b == j:
                        continue
                    if A[a, b] == 1:
                        # Try swap: remove (i,j) and (a,b), add (i,a) and (j,b)
                        if (A[i, a] == 0 and A[j, b] == 0
                                and i != a and j != b and i != b and j != a):
                            A_new = A.copy()
                            A_new[i, j] = A_new[j, i] = 0
                            A_new[a, b] = A_new[b, a] = 0
                            A_new[i, a] = A_new[a, i] = 1
                            A_new[j, b] = A_new[b, j] = 1
                            return self._adj_to_config(A_new)
                        # Try other rewiring
                        if (A[i, b] == 0 and A[j, a] == 0
                                and i != b and j != a and i != a and j != b):
                            A_new = A.copy()
                            A_new[i, j] = A_new[j, i] = 0
                            A_new[a, b] = A_new[b, a] = 0
                            A_new[i, b] = A_new[b, i] = 1
                            A_new[j, a] = A_new[a, j] = 1
                            return self._adj_to_config(A_new)
        # Case 2: edge (i, j) doesn't exist → add it by removing two others
        elif A[i, j] == 0:
            # Find edges (i, a) and (j, b) to remove, add (i, j) and (a, b)
            nbrs_i = np.where(A[i] > 0)[0]
            nbrs_j = np.where(A[j] > 0)[0]
            for a in nbrs_i:
                for b in nbrs_j:
                    if a != b and a != j and b != i and A[a, b] == 0:
                        A_new = A.copy()
                        A_new[i, a] = A_new[a, i] = 0
                        A_new[j, b] = A_new[b, j] = 0
                        A_new[i, j] = A_new[j, i] = 1
                        A_new[a, b] = A_new[b, a] = 1
                        return self._adj_to_config(A_new)

        return None  # no valid swap found

    def compute_target_rates(self, config_0, config_T, config_t, t):
        """Target rates over (n, n) node pairs.

        Nonzero at node pairs (i, j) involved in geodesic-progressing swaps.
        """
        A_t = self._config_to_adj(config_t)
        A_T = self._config_to_adj(config_T)

        # Find remaining R and A edges
        r_edges = []
        a_edges = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if A_t[i, j] == 1 and A_T[i, j] == 0:
                    r_edges.append((i, j))
                elif A_t[i, j] == 0 and A_T[i, j] == 1:
                    a_edges.append((i, j))

        rates = np.zeros((self.n, self.n), dtype=np.float32)
        if not r_edges or not a_edges:
            return rates

        # Find valid swaps: pairs of (R-edge, A-edge) that share a node
        valid_pairs = []
        for r in r_edges:
            for a in a_edges:
                shared = set(r) & set(a)
                if shared:
                    valid_pairs.append((r, a))

        if not valid_pairs:
            # No adjacent R-A pairs — spread rate across all R and A edges
            n_r = len(r_edges)
            n_a = len(a_edges)
            rate = 1.0 / (n_r + n_a) if (n_r + n_a) > 0 else 0.0
            for i, j in r_edges:
                rates[i, j] = rates[j, i] = rate
            for i, j in a_edges:
                rates[i, j] = rates[j, i] = rate
        else:
            rate = 1.0 / len(valid_pairs)
            for r, a in valid_pairs:
                # Mark both the R-edge and A-edge nodes
                for i, j in [r, a]:
                    rates[i, j] = max(rates[i, j], rate)
                    rates[j, i] = max(rates[j, i], rate)

        return rates

    def _canonical_graph(self):
        """Build a canonical graph with the target degree sequence."""
        A = np.zeros((self.n, self.n), dtype=np.float32)
        stubs = []
        for v, d in enumerate(self.degree_seq):
            stubs.extend([v] * d)
        # Pair stubs greedily
        used = set()
        i = 0
        while i < len(stubs) - 1:
            u = stubs[i]
            for j in range(i + 1, len(stubs)):
                v = stubs[j]
                if u != v and (u, v) not in used and (v, u) not in used:
                    A[u, v] = A[v, u] = 1
                    used.add((min(u, v), max(u, v)))
                    stubs.pop(j)
                    stubs.pop(i)
                    break
            else:
                i += 1
        return A

    def _random_swap(self, A, rng):
        """Perform one random double edge swap."""
        edges = list(zip(*np.where(np.triu(A) > 0)))
        if len(edges) < 2:
            return A
        idx = rng.choice(len(edges), size=2, replace=False)
        a, b = edges[idx[0]]
        c, d = edges[idx[1]]
        if a == c or a == d or b == c or b == d:
            return A
        if A[a, c] == 0 and A[b, d] == 0:
            A_new = A.copy()
            A_new[a, b] = A_new[b, a] = 0
            A_new[c, d] = A_new[d, c] = 0
            A_new[a, c] = A_new[c, a] = 1
            A_new[b, d] = A_new[d, b] = 1
            return A_new
        if A[a, d] == 0 and A[b, c] == 0:
            A_new = A.copy()
            A_new[a, b] = A_new[b, a] = 0
            A_new[c, d] = A_new[d, c] = 0
            A_new[a, d] = A_new[d, a] = 1
            A_new[b, c] = A_new[c, b] = 1
            return A_new
        return A

    def sample_source(self, rng):
        """Random graph with target degree sequence."""
        A = self._canonical_graph()
        for _ in range(self.n * 10):
            A = self._random_swap(A, rng)
        return self._adj_to_config(A)

    def sample_target(self, rng, beta=1.0, mcmc_pool=None, **kwargs):
        if mcmc_pool is not None:
            idx = int(rng.integers(len(mcmc_pool)))
            return mcmc_pool[idx].copy(), {'beta': beta}
        if self.pools and beta in self.pools:
            idx = int(rng.integers(len(self.pools[beta])))
            return self.pools[beta][idx].copy(), {'beta': beta}
        raise ValueError("Need precomputed pools")
