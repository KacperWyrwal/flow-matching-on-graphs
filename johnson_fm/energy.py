"""Energy functions and MCMC sampling for Ising model on J(n,k)."""

import numpy as np


def ising_energy(x, J, h):
    """Ising energy E(x) = -0.5 * x @ J @ x - h @ x."""
    return -0.5 * x @ J @ x - h @ x


def uniform_sample(n, k, rng):
    """Sample binary string with exactly k ones uniformly."""
    x = np.zeros(n, dtype=np.float32)
    ones = rng.choice(n, size=k, replace=False)
    x[ones] = 1.0
    return x


def mcmc_kawasaki(energy_fn, n, k, beta, n_steps, rng, x_init=None,
                  J=None, h=None):
    """Kawasaki swap MCMC on J(n,k) with Metropolis-Hastings.

    If J and h are provided, uses fast local dE computation (O(n) per step).
    Otherwise falls back to full energy recomputation (O(n²) per step).

    Returns final configuration (n,) float32 array.
    """
    if x_init is not None:
        x = x_init.copy()
    else:
        x = uniform_sample(n, k, rng)

    use_fast = J is not None and h is not None
    if not use_fast:
        E_current = energy_fn(x)

    # Use numpy arrays for O(1) random choice
    ones_arr = np.where(x == 1)[0].astype(np.intp)
    zeros_arr = np.where(x == 0)[0].astype(np.intp)
    # Position-in-array lookup for O(1) swap
    ones_pos = np.full(n, -1, dtype=np.intp)  # site -> index in ones_arr
    zeros_pos = np.full(n, -1, dtype=np.intp)
    for idx_o in range(len(ones_arr)):
        ones_pos[ones_arr[idx_o]] = idx_o
    for idx_z in range(len(zeros_arr)):
        zeros_pos[zeros_arr[idx_z]] = idx_z
    n_ones = len(ones_arr)
    n_zeros = len(zeros_arr)

    # For fast path: precompute sparse neighbor lists
    if use_fast:
        nbrs = [np.where(J[a] != 0)[0] for a in range(n)]
        J_vals = [J[a, nbrs[a]] for a in range(n)]

    # Pre-generate random numbers in batches for speed
    batch_rand = 4096
    rand_idx = batch_rand  # trigger initial generation

    for step in range(n_steps):
        if rand_idx >= batch_rand:
            r_ones = rng.integers(n_ones, size=batch_rand)
            r_zeros = rng.integers(n_zeros, size=batch_rand)
            r_unif = rng.uniform(size=batch_rand)
            rand_idx = 0

        i = int(ones_arr[r_ones[rand_idx]])
        j = int(zeros_arr[r_zeros[rand_idx]])

        if use_fast:
            dE = 0.0
            for nb, jv in zip(nbrs[i], J_vals[i]):
                if nb != j:
                    dE += jv * x[nb]
            for nb, jv in zip(nbrs[j], J_vals[j]):
                if nb != i:
                    dE -= jv * x[nb]
            dE += J[i, j]
            dE -= (h[j] - h[i])
        else:
            x[i] = 0.0
            x[j] = 1.0
            E_prop = energy_fn(x)
            dE = E_prop - E_current
            x[i] = 1.0
            x[j] = 0.0

        accept = dE < 0 or r_unif[rand_idx] < np.exp(-beta * dE)
        rand_idx += 1

        if accept:
            x[i] = 0.0
            x[j] = 1.0

            # Update ones_arr: swap i out, j in (O(1))
            pos_i = ones_pos[i]
            ones_arr[pos_i] = j
            ones_pos[j] = pos_i
            ones_pos[i] = -1

            # Update zeros_arr: swap j out, i in (O(1))
            pos_j = zeros_pos[j]
            zeros_arr[pos_j] = i
            zeros_pos[i] = pos_j
            zeros_pos[j] = -1

            if not use_fast:
                E_current += dE

    return x.astype(np.float32)


def generate_mcmc_pool(J, h, n, k, beta, pool_size, chain_length, seed=42):
    """Pre-generate a pool of MCMC samples for a given (J, h, beta).

    Runs multiple independent chains for decorrelation.
    Returns (pool_size, n) float32 array.
    """
    rng = np.random.default_rng(seed)
    energy_fn = lambda x: ising_energy(x, J, h)
    pool = []

    n_chains = min(10, pool_size)
    samples_per_chain = (pool_size + n_chains - 1) // n_chains

    for c in range(n_chains):
        x = uniform_sample(n, k, rng)
        # Burn-in (fast path with J, h)
        x = mcmc_kawasaki(energy_fn, n, k, beta, chain_length, rng,
                          x_init=x, J=J, h=h)
        for s in range(samples_per_chain):
            # Thinning
            x = mcmc_kawasaki(energy_fn, n, k, beta,
                              max(chain_length // 10, 100), rng,
                              x_init=x, J=J, h=h)
            pool.append(x.copy())
            if len(pool) >= pool_size:
                break
        if len(pool) >= pool_size:
            break

    return np.array(pool[:pool_size], dtype=np.float32)


def compute_exact_boltzmann(J, h, n, k, beta):
    """Compute exact Boltzmann distribution over J(n,k) (small n only).

    Returns (configs, probs) where configs is (M, n) and probs is (M,).
    M = C(n, k).
    """
    from itertools import combinations
    configs = []
    energies = []
    for combo in combinations(range(n), k):
        x = np.zeros(n, dtype=np.float32)
        x[list(combo)] = 1.0
        configs.append(x)
        energies.append(ising_energy(x, J, h))

    configs = np.array(configs)
    energies = np.array(energies)

    # Log-sum-exp for numerical stability
    log_probs = -beta * energies
    log_probs -= log_probs.max()
    probs = np.exp(log_probs)
    probs /= probs.sum()

    return configs, probs
