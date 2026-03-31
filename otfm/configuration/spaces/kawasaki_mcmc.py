"""MCMC sampling utilities for Kawasaki dynamics on 2D lattice.

From config_fm/spaces/kawasaki_mcmc.py.
"""

import numpy as np


def get_neighbors(i, L):
    """Return 4 neighbors of site i on L x L torus."""
    y, x = i // L, i % L
    return [
        y * L + (x + 1) % L,        # right
        y * L + (x - 1) % L,        # left
        ((y + 1) % L) * L + x,      # down
        ((y - 1) % L) * L + x,      # up
    ]


def ising_energy_lattice(config, L, J=1.0):
    """Ising energy on L x L torus. config in {0,1}^N, uses +/-1 convention."""
    N = L * L
    E = 0.0
    for y in range(L):
        for x in range(L):
            i = y * L + x
            si = 2 * config[i] - 1
            # Right neighbor
            j = y * L + (x + 1) % L
            sj = 2 * config[j] - 1
            E -= J * si * sj
            # Down neighbor
            j = ((y + 1) % L) * L + x
            sj = 2 * config[j] - 1
            E -= J * si * sj
    return float(E)


def compute_kawasaki_dE(config, i, j, L, J=1.0):
    """Energy change from swapping sites i and j. O(degree) cost."""
    si = 2 * config[i] - 1
    sj = 2 * config[j] - 1

    if si == sj:
        return 0.0

    nbrs_i = get_neighbors(i, L)
    nbrs_j = get_neighbors(j, L)

    dE = 0.0
    for nb in nbrs_i:
        if nb != j:
            s_nb = 2 * config[nb] - 1
            dE += -J * (sj - si) * s_nb
    for nb in nbrs_j:
        if nb != i:
            s_nb = 2 * config[nb] - 1
            dE += -J * (si - sj) * s_nb

    return dE


def kawasaki_mcmc(space, beta, n_steps, rng, config=None):
    """Kawasaki MCMC on the lattice.

    Args:
        space: KawasakiSpace instance (needs .L, .N, .k, .J_coupling)
        beta: inverse temperature
        n_steps: number of proposed swaps
        rng: numpy random generator
        config: initial configuration (if None, starts from random)
    Returns: (N,) float32 configuration
    """
    L = space.L
    N = space.N
    J = space.J_coupling
    edges = space.position_graph_edges()
    n_edges = edges.shape[1]

    if config is None:
        config = space.sample_source(rng)
    else:
        config = config.copy()

    for _ in range(n_steps):
        e = int(rng.integers(n_edges))
        i, j = int(edges[0, e]), int(edges[1, e])

        if config[i] == config[j]:
            continue

        dE = compute_kawasaki_dE(config, i, j, L, J)

        if dE < 0 or rng.uniform() < np.exp(-beta * dE):
            config[i], config[j] = config[j], config[i]

    return config


def generate_kawasaki_pool(space, beta, pool_size, chain_length, seed=42):
    """Pre-generate MCMC pool for a given beta.

    Runs multiple chains with thinning for decorrelation.
    Returns (pool_size, N) float32 array.
    """
    rng = np.random.default_rng(seed)
    pool = []

    n_chains = max(1, pool_size // 10)
    samples_per_chain = (pool_size + n_chains - 1) // n_chains

    for _ in range(n_chains):
        # Burn-in from random
        config = kawasaki_mcmc(space, beta, chain_length, rng)
        for _ in range(samples_per_chain):
            # Continue from previous config (thinning)
            config = kawasaki_mcmc(space, beta,
                                   max(chain_length // 10, space.N * 5),
                                   rng, config=config)
            pool.append(config.copy())
            if len(pool) >= pool_size:
                break
        if len(pool) >= pool_size:
            break

    return np.array(pool[:pool_size], dtype=np.float32)
