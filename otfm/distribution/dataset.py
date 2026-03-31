"""
Dataset for meta-level flow matching over graph distributions.

From meta_fm/dataset.py.
"""

import numpy as np
import torch
import ot
from scipy.linalg import expm

from otfm.graph.structure import (
    GraphStructure,
    GeodesicCache,
    rate_matrix_to_edge_index,
)
from otfm.graph.coupling import (
    compute_cost_matrix,
    compute_ot_coupling,
    compute_meta_cost_matrix_batch,
)
from otfm.graph.flow import marginal_distribution_fast, marginal_rate_matrix_fast


class MetaFlowMatchingDataset(torch.utils.data.Dataset):
    """
    Generates training data for the meta-level flow matching.

    Stores pre-generated tuples (mu_tau, tau, R_target) where:
        - mu_tau is the marginal distribution at time tau
        - tau is the time in [0, 1]
        - R_target is the target rate matrix at that time

    Constructor args:
        graph: GraphStructure
        source_distributions: list of np.ndarray, each shape (N,)
        target_distributions: list of np.ndarray, each shape (N,)
        n_samples: int
        cache: GeodesicCache or None
            If None, one is built automatically.
        use_sinkhorn: bool = True
            Use Sinkhorn (faster) for the meta-cost matrix computation.
            Set False to use exact LP.
        meta_coupling: np.ndarray or None
            If provided, use this (n_sources, n_targets) coupling directly and
            skip the meta-cost / meta-OT computation. Useful when the ground-
            truth pairing is known (e.g. diagonal coupling for Experiment 6).
        seed: int = 42
    """

    def __init__(
        self,
        graph: GraphStructure,
        source_distributions,
        target_distributions,
        n_samples: int,
        cost_type: str = 'wasserstein',
        cache: GeodesicCache | None = None,
        use_sinkhorn: bool = True,
        meta_coupling: np.ndarray | None = None,
        seed: int = 42,
    ):
        self.graph = graph
        self.N = graph.N
        rng = np.random.default_rng(seed)

        n_sources = len(source_distributions)
        n_targets = len(target_distributions)

        # Step 1: graph-level cost matrix (for meta-OT only)
        cost = compute_cost_matrix(graph)

        # Step 2 & 3: meta-OT coupling
        if meta_coupling is not None:
            Pi_meta = meta_coupling
        else:
            W_meta = compute_meta_cost_matrix_batch(
                source_distributions, target_distributions, cost,
                use_sinkhorn=use_sinkhorn,
            )
            uniform_s = np.ones(n_sources) / n_sources
            uniform_t = np.ones(n_targets) / n_targets
            Pi_meta = ot.emd(uniform_s, uniform_t, W_meta)

        # Step 4: build GeodesicCache once, then precompute for each coupling
        # in the meta-coupling support (at most 2*max(n_s,n_t)-1 nonzero entries)
        if cache is None:
            cache = GeodesicCache(graph)

        meta_nonzero = np.argwhere(Pi_meta > 1e-12)

        # Precompute graph-level couplings and populate cache
        graph_couplings: dict = {}
        for s_idx, t_idx in meta_nonzero:
            s_idx, t_idx = int(s_idx), int(t_idx)
            if (s_idx, t_idx) not in graph_couplings:
                pi = compute_ot_coupling(
                    source_distributions[s_idx],
                    target_distributions[t_idx],
                    graph_struct=graph,
                )
                graph_couplings[(s_idx, t_idx)] = pi
                cache.precompute_for_coupling(pi)

        # Step 5: sample from Pi_meta and generate training tuples
        pi_flat = Pi_meta.flatten()
        pi_flat = pi_flat / pi_flat.sum()

        mu_list = []
        tau_list = []
        R_list = []

        for _ in range(n_samples):
            idx = int(rng.choice(n_sources * n_targets, p=pi_flat))
            s = idx // n_targets
            t_idx = idx % n_targets

            pi_st = graph_couplings[(s, t_idx)]
            tau = float(rng.uniform(0.0, 0.999))

            mu_tau = marginal_distribution_fast(cache, pi_st, tau)
            R_target = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi_st, tau)

            mu_list.append(mu_tau)
            tau_list.append(tau)
            R_list.append(R_target)

        self.mu_data = torch.tensor(np.array(mu_list), dtype=torch.float32)
        self.tau_data = torch.tensor(np.array(tau_list), dtype=torch.float32).unsqueeze(1)
        self.R_data = torch.tensor(np.array(R_list), dtype=torch.float32)

    def __len__(self):
        return len(self.mu_data)

    def __getitem__(self, idx):
        return self.mu_data[idx], self.tau_data[idx], self.R_data[idx]


class ConditionalMetaFlowMatchingDataset(torch.utils.data.Dataset):
    """
    Like MetaFlowMatchingDataset but each sample also carries per-node context.

    Designed for conditional flow matching where context (e.g. an observation
    and diffusion time) is fixed while the flow time tau varies.

    Constructor args:
        graph: GraphStructure
        source_obs_pairs: list of dicts with keys:
            'mu_source': np.ndarray (N,)  -- target of the backward flow
            'mu_obs':    np.ndarray (N,)  -- starting point / observation
            'tau_diff':  float            -- diffusion time (scalar context)
        n_samples: int
        seed: int = 42

    Uses diagonal meta-coupling: pair k transports mu_obs_k -> mu_source_k.

    __getitem__ returns:
        mu:       (N,)          distribution at flow time tau
        tau:      (1,)          flow time
        context:  (N, 2)        [mu_obs(a), tau_diff] per node
        R_target: (N, N)        target rate matrix
    """

    def __init__(self, graph, source_obs_pairs, n_samples: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        N = graph.N
        cache = GeodesicCache(graph)

        n_pairs = len(source_obs_pairs)

        # Precompute graph-level couplings: mu_obs -> mu_source
        couplings = []
        for pair in source_obs_pairs:
            pi = compute_ot_coupling(pair['mu_obs'], pair['mu_source'], graph_struct=graph)
            couplings.append(pi)
            cache.precompute_for_coupling(pi)

        mu_list, tau_list, ctx_list, R_list = [], [], [], []

        for _ in range(n_samples):
            k = int(rng.integers(0, n_pairs))
            pair = source_obs_pairs[k]
            pi = couplings[k]
            tau = float(rng.uniform(0.0, 0.999))

            mu_tau = marginal_distribution_fast(cache, pi, tau)
            R_target = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi, tau)

            # Context: [mu_obs(a), tau_diff] for each node
            context = np.stack([pair['mu_obs'],
                                 np.full(N, pair['tau_diff'])], axis=-1)  # (N, 2)

            mu_list.append(mu_tau)
            tau_list.append(tau)
            ctx_list.append(context)
            R_list.append(R_target)

        self.mu_data = torch.tensor(np.array(mu_list), dtype=torch.float32)
        self.tau_data = torch.tensor(np.array(tau_list), dtype=torch.float32).unsqueeze(1)
        self.ctx_data = torch.tensor(np.array(ctx_list), dtype=torch.float32)
        self.R_data = torch.tensor(np.array(R_list), dtype=torch.float32)

    def __len__(self):
        return len(self.mu_data)

    def __getitem__(self, idx):
        return self.mu_data[idx], self.tau_data[idx], self.ctx_data[idx], self.R_data[idx]


class InpaintingDataset(torch.utils.data.Dataset):
    """
    Training data for conditional distribution inpainting.

    Each sample: OT flow from a corrupted distribution back to the clean one,
    conditioned on the corruption mask.

    Constructor args:
        graph: GraphStructure
        clean_distributions: list of np.ndarray (N,)
        n_masks_per_dist: int = 50 -- random masks per clean distribution
        n_masked_nodes: int = 3
        n_samples: int = 10000
        seed: int = 42

    __getitem__ returns:
        mu:       (N,)      distribution at flow time tau
        tau:      (1,)      flow time
        context:  (N, 2)    [mu_corrupted(a), mask(a)] per node
        R_target: (N, N)    target rate matrix
    """

    def __init__(self, graph, clean_distributions, n_masks_per_dist: int = 50,
                 n_masked_nodes: int = 3, n_samples: int = 10000, seed: int = 42):
        rng = np.random.default_rng(seed)
        N = graph.N
        cache = GeodesicCache(graph)
        n_clean = len(clean_distributions)

        # Pre-generate (corrupted, clean, mask) triples
        triples = []
        for mu_clean in clean_distributions:
            for _ in range(n_masks_per_dist):
                mask_idx = rng.choice(N, size=n_masked_nodes, replace=False)
                mask = np.ones(N, dtype=float)
                mask[mask_idx] = 0.0

                mu_corr = mu_clean.copy()
                mu_corr[mask_idx] = 0.0
                s = mu_corr.sum()
                if s > 1e-12:
                    mu_corr /= s
                else:
                    mu_corr = np.ones(N) / N  # fallback

                pi = compute_ot_coupling(mu_corr, mu_clean, graph_struct=graph)
                cache.precompute_for_coupling(pi)
                triples.append((mu_corr, mu_clean, mask, pi))

        mu_list, tau_list, ctx_list, R_list = [], [], [], []

        for _ in range(n_samples):
            mu_corr, mu_clean, mask, pi = triples[int(rng.integers(0, len(triples)))]
            tau = float(rng.uniform(0.0, 0.999))

            mu_tau = marginal_distribution_fast(cache, pi, tau)
            R_target = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi, tau)

            # Context: [mu_corrupted(a), mask(a)] per node
            context = np.stack([mu_corr, mask], axis=-1)  # (N, 2)

            mu_list.append(mu_tau)
            tau_list.append(tau)
            ctx_list.append(context)
            R_list.append(R_target)

        self.mu_data = torch.tensor(np.array(mu_list), dtype=torch.float32)
        self.tau_data = torch.tensor(np.array(tau_list), dtype=torch.float32).unsqueeze(1)
        self.ctx_data = torch.tensor(np.array(ctx_list), dtype=torch.float32)
        self.R_data = torch.tensor(np.array(R_list), dtype=torch.float32)

    def __len__(self):
        return len(self.mu_data)

    def __getitem__(self, idx):
        return self.mu_data[idx], self.tau_data[idx], self.ctx_data[idx], self.R_data[idx]


class TopologyGeneralizationDataset(torch.utils.data.Dataset):
    """
    Generates training data across multiple graph topologies for topology
    generalization experiments.

    Each sample includes the graph structure (edge_index, n_nodes) alongside
    the usual (mu_tau, tau, context, R_target).

    Constructor args:
        graphs:              list of (name: str, R: np.ndarray) pairs
        n_samples_per_graph: int = 1000
        n_pairs_per_graph:   int = 50  -- source-obs pairs per graph
        tau_diff_range:      tuple = (0.3, 1.5)
        seed:                int = 42

    __getitem__ returns tuple:
        mu:         (N,)       distribution at flow time tau
        tau:        (1,)       flow time
        context:    (N, 2)     [mu_obs(a), tau_diff] per node
        R_target:   (N, N)     target u_tilde rate matrix
        edge_index: (2, E)     graph edges
        n_nodes:    int
    """

    def __init__(self, graphs, n_samples_per_graph: int = 1000,
                 n_pairs_per_graph: int = 50,
                 tau_diff_range: tuple = (0.3, 1.5),
                 seed: int = 42):
        rng = np.random.default_rng(seed)
        self.samples = []

        for _g_idx, (name, R) in enumerate(graphs):
            N = R.shape[0]
            graph_struct = GraphStructure(R)
            cache = GeodesicCache(graph_struct)
            edge_index = rate_matrix_to_edge_index(R)

            # Generate source-obs pairs for this graph
            pairs = []
            couplings = []
            for _ in range(n_pairs_per_graph):
                n_peaks = int(rng.integers(1, 4))  # 1-3 peaks
                peak_nodes = rng.choice(N, size=n_peaks, replace=False)
                weights = rng.dirichlet(np.full(n_peaks, 2.0))
                mu_source = np.ones(N) * 0.2 / N
                for node, w in zip(peak_nodes, weights):
                    mu_source[node] += 0.8 * w
                mu_source = np.clip(mu_source, 1e-6, None)
                mu_source /= mu_source.sum()

                tau_diff = float(rng.uniform(*tau_diff_range))
                from scipy.linalg import expm as _expm
                mu_obs = mu_source @ _expm(tau_diff * R)
                mu_obs = np.clip(mu_obs, 1e-12, None)
                mu_obs /= mu_obs.sum()

                pi = compute_ot_coupling(mu_obs, mu_source, graph_struct=graph_struct)
                cache.precompute_for_coupling(pi)

                pairs.append({'mu_obs': mu_obs, 'tau_diff': tau_diff})
                couplings.append(pi)

            # Sample training tuples from these pairs
            for _ in range(n_samples_per_graph):
                k = int(rng.integers(0, n_pairs_per_graph))
                pair = pairs[k]
                pi = couplings[k]
                tau = float(rng.uniform(0.0, 0.999))

                mu_tau = marginal_distribution_fast(cache, pi, tau)
                R_target_np = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi, tau)

                context = np.stack(
                    [pair['mu_obs'], np.full(N, pair['tau_diff'])], axis=-1)  # (N, 2)

                self.samples.append((
                    torch.tensor(mu_tau, dtype=torch.float32),
                    torch.tensor([tau], dtype=torch.float32),
                    torch.tensor(context, dtype=torch.float32),
                    torch.tensor(R_target_np, dtype=torch.float32),
                    edge_index,
                    N,
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PosteriorSamplingDataset(torch.utils.data.Dataset):
    """
    Training data for posterior sampling via flow matching.

    Each sample starts from a Dirichlet(1,...,1) draw and flows toward
    the source distribution, conditioned on the sensor observation.
    Multiple Dirichlet starts per source distribution ensure the model
    sees diverse starting points for the same target.

    Constructor args:
        R:                   (N, N) rate matrix
        source_obs_pairs:    list of dicts with keys:
                               'mu_source':   (N,) numpy array
                               'mu_backproj': (N,) backprojected sensor reading
                               'tau_diff':    float
        n_starts_per_pair:   int = 10 -- Dirichlet starts per source
        n_samples:           int = 15000
        seed:                int = 42

    __getitem__ returns (mu, tau, context, R_target, edge_index, n_nodes)
    compatible with train_flexible_conditional.
    """

    def __init__(self, R: np.ndarray, source_obs_pairs,
                 n_starts_per_pair: int = 10,
                 n_samples: int = 15000, seed: int = 42):
        rng = np.random.default_rng(seed)
        N = R.shape[0]
        graph_struct = GraphStructure(R)
        cache = GeodesicCache(graph_struct)
        edge_index = rate_matrix_to_edge_index(R)

        all_triples = []  # (mu_source, mu_backproj, tau_diff, coupling)
        for pair in source_obs_pairs:
            mu_source   = pair['mu_source']
            mu_backproj = pair['mu_backproj']
            tau_diff    = pair['tau_diff']
            for _ in range(n_starts_per_pair):
                mu_start = rng.dirichlet(np.ones(N))
                pi = compute_ot_coupling(mu_start, mu_source, graph_struct=graph_struct)
                cache.precompute_for_coupling(pi)
                all_triples.append((mu_source, mu_backproj, tau_diff, pi))

        self.samples = []
        for _ in range(n_samples):
            mu_source, mu_backproj, tau_diff, pi = \
                all_triples[int(rng.integers(len(all_triples)))]
            tau = float(rng.uniform(0.0, 0.999))

            mu_tau      = marginal_distribution_fast(cache, pi, tau)
            R_target_np = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi, tau)
            context     = np.stack([mu_backproj, np.full(N, tau_diff)], axis=-1)

            self.samples.append((
                torch.tensor(mu_tau,      dtype=torch.float32),
                torch.tensor([tau],       dtype=torch.float32),
                torch.tensor(context,     dtype=torch.float32),
                torch.tensor(R_target_np, dtype=torch.float32),
                edge_index,
                N,
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CubeBoundaryDataset(torch.utils.data.Dataset):
    """
    Training data for boundary observation on a cube (or any fixed-mask) graph.

    The OT flow source can be:
      'observation' -- flow from boundary-masked observation -> clean  (v1)
      'uniform'     -- flow from uniform -> clean; observation only in context (v2)

    Context per node: [observed_value(a), mask(a)].

    Returns (mu, tau, context, R_target, edge_index, n_nodes) per sample so it
    is compatible with train_flexible_conditional.

    Constructor args:
        R:                    (N, N) rate matrix for the graph
        mask:                 (N,) binary array -- 1 = observed, 0 = hidden
        clean_distributions:  list of np.ndarray (N,)
        boundary_observations: list of np.ndarray (N,) or None.
            If provided, used as the context observation (e.g. diffused boundary
            values). If None, observations are computed as apply_boundary_mask
            of each clean distribution.
        tau_diffs:            list of float or None.
            If provided, each entry is the diffusion time for that distribution
            and is included as a third context channel per node: context shape
            becomes (N, 3) = [mu_obs, mask, tau_diff]. If None, context is
            (N, 2) = [mu_obs, mask].
        n_samples:            int = 10000
        start_from:           'uniform' (default) or 'observation'
        seed:                 int = 42
    """

    def __init__(self, R: np.ndarray, mask: np.ndarray,
                 clean_distributions, boundary_observations=None,
                 tau_diffs=None,
                 n_samples: int = 10000,
                 start_from: str = 'uniform', seed: int = 42):
        rng = np.random.default_rng(seed)
        N = R.shape[0]
        graph_struct = GraphStructure(R)
        cache = GeodesicCache(graph_struct)
        edge_index = rate_matrix_to_edge_index(R)

        # Pre-compute (mu_obs, tau_diff, mu_clean, coupling) for each distribution
        pairs = []
        for k, mu_clean in enumerate(clean_distributions):
            if boundary_observations is not None:
                mu_obs = boundary_observations[k]
            else:
                mu_obs = mu_clean * mask
                s = mu_obs.sum()
                mu_obs = mu_obs / s if s > 1e-12 else mask / mask.sum()

            td = float(tau_diffs[k]) if tau_diffs is not None else None

            if start_from == 'uniform':
                mu_start = np.ones(N) / N
            else:
                mu_start = mu_obs

            pi = compute_ot_coupling(mu_start, mu_clean, graph_struct=graph_struct)
            cache.precompute_for_coupling(pi)
            pairs.append((mu_obs, td, mu_clean, pi))

        self.samples = []
        for _ in range(n_samples):
            mu_obs, td, mu_clean, pi = pairs[int(rng.integers(0, len(pairs)))]
            tau = float(rng.uniform(0.0, 0.999))

            mu_tau = marginal_distribution_fast(cache, pi, tau)
            R_target_np = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi, tau)
            if td is not None:
                context = np.stack([mu_obs, mask, np.full(N, td)], axis=-1)  # (N, 3)
            else:
                context = np.stack([mu_obs, mask], axis=-1)  # (N, 2)

            self.samples.append((
                torch.tensor(mu_tau, dtype=torch.float32),
                torch.tensor([tau], dtype=torch.float32),
                torch.tensor(context, dtype=torch.float32),
                torch.tensor(R_target_np, dtype=torch.float32),
                edge_index,
                N,
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CubePosteriorDataset(torch.utils.data.Dataset):
    """
    Training data for posterior sampling with cube boundary observations.

    Like CubeBoundaryDataset but each sample starts from a configurable
    prior distribution over the simplex instead of a fixed uniform.

    Constructor args:
        R:                   (N, N) rate matrix
        mask:                (N,) binary array -- 1 = observed, 0 = hidden
        source_obs_pairs:    list of dicts with keys:
                               'mu_source': (N,)
                               'mu_obs':    (N,)  boundary observation
                               'tau_diff':  float
        n_starts_per_pair:   int = 10
        n_samples:           int = 15000
        seed:                int = 42
        init_dist_fn:        callable(rng) -> (N,) or None.
                             If None, defaults to Dirichlet(1,...,1).

    Context per node: [mu_obs(a), mask(a), tau_diff] -- shape (N, 3).
    __getitem__ returns (mu, tau, context, R_target, edge_index, n_nodes).
    """

    def __init__(self, R: np.ndarray, mask: np.ndarray,
                 source_obs_pairs,
                 n_starts_per_pair: int = 10,
                 n_samples: int = 15000, seed: int = 42,
                 init_dist_fn=None):
        rng = np.random.default_rng(seed)
        N = R.shape[0]
        if init_dist_fn is None:
            init_dist_fn = lambda rng: rng.dirichlet(np.ones(N))
        graph_struct = GraphStructure(R)
        cache = GeodesicCache(graph_struct)
        edge_index = rate_matrix_to_edge_index(R)

        all_triples = []
        for pair in source_obs_pairs:
            mu_source = pair['mu_source']
            mu_obs    = pair['mu_obs']
            tau_diff  = pair['tau_diff']
            for _ in range(n_starts_per_pair):
                mu_start = init_dist_fn(rng)
                pi = compute_ot_coupling(mu_start, mu_source, graph_struct=graph_struct)
                cache.precompute_for_coupling(pi)
                all_triples.append((mu_source, mu_obs, tau_diff, pi))

        self.samples = []
        for _ in range(n_samples):
            mu_source, mu_obs, tau_diff, pi = \
                all_triples[int(rng.integers(len(all_triples)))]
            tau = float(rng.uniform(0.0, 0.999))

            mu_tau      = marginal_distribution_fast(cache, pi, tau)
            R_target_np = (1.0 - tau) * marginal_rate_matrix_fast(cache, pi, tau)
            context     = np.stack([mu_obs, mask, np.full(N, tau_diff)], axis=-1)

            self.samples.append((
                torch.tensor(mu_tau,      dtype=torch.float32),
                torch.tensor([tau],       dtype=torch.float32),
                torch.tensor(context,     dtype=torch.float32),
                torch.tensor(R_target_np, dtype=torch.float32),
                edge_index,
                N,
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
