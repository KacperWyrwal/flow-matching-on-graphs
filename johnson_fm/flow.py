"""Conditional flow computations on the Johnson graph J(n,k)."""

import numpy as np


def sample_intermediate(x_0, x_T, t, rng):
    """Sample x_t along geodesic on J(n,k).

    Args:
        x_0: (n,) source binary string
        x_T: (n,) target binary string
        t: flow time in [0, 1)
        rng: numpy random generator

    Returns:
        x_t: (n,) intermediate binary string (same Hamming weight)
        S_plus_rem: indices remaining in S+ (positions to turn off)
        S_minus_rem: indices remaining in S- (positions to turn on)
        d: geodesic distance
        ell: number of completed swaps
    """
    S_plus = np.where((x_0 == 1) & (x_T == 0))[0]   # positions to turn off
    S_minus = np.where((x_0 == 0) & (x_T == 1))[0]   # positions to turn on
    d = len(S_plus)

    if d == 0:
        return x_0.copy(), np.array([], dtype=int), np.array([], dtype=int), 0, 0

    # Sample number of completed swaps
    ell = int(rng.binomial(d, min(t, 0.999)))

    # Sample which swaps are done
    A_idx = rng.choice(len(S_plus), size=ell, replace=False) if ell > 0 else np.array([], dtype=int)
    B_idx = rng.choice(len(S_minus), size=ell, replace=False) if ell > 0 else np.array([], dtype=int)
    A = S_plus[A_idx]
    B = S_minus[B_idx]

    # Construct x_t
    x_t = x_0.copy()
    if len(A) > 0:
        x_t[A] = 0.0
    if len(B) > 0:
        x_t[B] = 1.0

    # Remaining sets
    mask_plus = np.ones(len(S_plus), dtype=bool)
    mask_minus = np.ones(len(S_minus), dtype=bool)
    if len(A_idx) > 0:
        mask_plus[A_idx] = False
    if len(B_idx) > 0:
        mask_minus[B_idx] = False
    S_plus_rem = S_plus[mask_plus]
    S_minus_rem = S_minus[mask_minus]

    return x_t, S_plus_rem, S_minus_rem, d, ell


def compute_target_rates(n, S_plus_rem, S_minus_rem, d, ell):
    """Compute target swap rates (u_tilde) at x_t.

    The target rate for each valid geodesic-progressing swap (i, j) where
    i in S_plus_rem and j in S_minus_rem is 1/(d - ell).

    Returns:
        target_rates: (n, n) float32 array, sparse (only valid swaps nonzero)
    """
    d_rem = d - ell
    target_rates = np.zeros((n, n), dtype=np.float32)

    if d_rem > 0:
        rate = 1.0 / d_rem
        for i in S_plus_rem:
            for j in S_minus_rem:
                target_rates[i, j] = rate

    return target_rates
