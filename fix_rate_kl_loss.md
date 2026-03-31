# Fix: Rate KL Divergence Loss (Principled Discrete Loss)

## Motivation

The theoretically correct loss for matching CTMC path measures is the
KL divergence between rate vectors, not MSE. The KL between path measures
of two CTMCs with rates Q and Q^theta decomposes as:

    KL(Q || Q^theta) = E_Q ∫ Σ_{b≠a} [Q_ab log(Q_ab/Q^theta_ab) - Q_ab + Q^theta_ab] dt

The integrand is the generalized KL divergence (Bregman divergence with
generating function φ(x) = x log x - x) between rate vectors from the
current state.

This is the discrete analogue of the L2 velocity loss in Euclidean flow
matching, which arises from Girsanov's theorem. Using it makes the
training objective consistent with the theoretical derivation.

## Implementation

Add to `meta_fm/train.py`:

```python
def rate_kl_divergence(target, predicted, eps=1e-8):
    """
    Generalized KL divergence between rate vectors.
    
    For rates r (target) and r_theta (predicted):
        D(r || r_theta) = Σ [r * log(r / r_theta) - r + r_theta]
    
    This is non-negative, zero iff r == r_theta.
    Equivalent to KL between Poisson processes with rates r and r_theta.
    
    Args:
        target:    (*, N, N) target rate matrices (off-diagonal entries)
        predicted: (*, N, N) predicted rate matrices (off-diagonal entries)
        eps:       small constant for numerical stability
    
    Returns:
        (*, ) per-sample loss (summed over off-diagonal entries)
    """
    # Only compute on off-diagonal entries
    N = target.shape[-1]
    off_diag = ~torch.eye(N, dtype=torch.bool, device=target.device)
    
    r = target[..., off_diag].clamp(min=eps)        # target rates
    r_theta = predicted[..., off_diag].clamp(min=eps)  # predicted rates
    
    # Generalized KL: r * log(r/r_theta) - r + r_theta
    per_entry = r * torch.log(r / r_theta) - r + r_theta
    
    return per_entry.sum(dim=-1)  # sum over edges


def mse_loss(target, predicted):
    """
    MSE between rate matrices (off-diagonal entries).
    The previous default loss.
    """
    N = target.shape[-1]
    off_diag = ~torch.eye(N, dtype=torch.bool, device=target.device)
    
    diff_sq = (target[..., off_diag] - predicted[..., off_diag]) ** 2
    return diff_sq.sum(dim=-1)
```

## Integration into Training Loops

```python
def train_flexible_conditional(model, dataset, n_epochs=1000, ...,
                                loss_type='rate_kl',
                                loss_weighting='uniform',
                                ema_decay=0.999):
    """
    loss_type: 'rate_kl' (default, principled) or 'mse' (fallback)
    """
    ...
    for epoch in range(n_epochs):
        for batch in dataloader:
            R_pred = model(...)
            R_target = batch['R_target']
            
            if loss_type == 'rate_kl':
                per_sample = rate_kl_divergence(R_target, R_pred)
            elif loss_type == 'mse':
                per_sample = mse_loss(R_target, R_pred)
            
            # Normalize by number of off-diagonal entries
            per_sample = per_sample / (N * (N - 1))
            
            # Apply time weighting
            if loss_weighting == 'original':
                weights = 1.0 / (1.0 - tau).clamp(min=0.001) ** 2
            elif loss_weighting == 'uniform':
                weights = torch.ones_like(tau)
            elif loss_weighting == 'linear':
                weights = 1.0 / (1.0 - tau).clamp(min=0.001)
            
            loss = (weights * per_sample).mean()
            ...
```

## CLI

```python
parser.add_argument('--loss-type', type=str, default='rate_kl',
                    choices=['rate_kl', 'mse'],
                    help='Loss function: rate_kl (principled) or mse (fallback)')
```

## Properties of Rate KL vs MSE

                    Rate KL                 MSE
Theoretical basis   Path measure KL         None (heuristic)
Zero iff            r == r_theta            r == r_theta
Symmetry            Asymmetric              Symmetric
Small rate errors   Less penalized          Equally penalized
Large rate errors   More penalized          Equally penalized
Requires r > 0     Yes (use eps)           No
Scale invariance    Yes (relative error)    No (absolute error)

The asymmetry of rate KL is actually desirable: underestimating a large
rate (missing an important transition) is penalized more than overestimating
a small rate (adding a spurious weak transition). This matches the
importance of transitions — missing a geodesic transition is worse than
adding a tiny non-geodesic leak.

## Note on Factorization

We train on the factorized targets: u_tilde = (1-t) * u. The rate KL
applies to u_tilde directly — both target and predicted u_tilde are
non-negative (the factorization preserves non-negativity of off-diagonal
entries since rates are non-negative and (1-t) >= 0).

The factorized targets can have zero entries (when a transition is not
along a geodesic). The eps clamp handles this — a zero target rate
contributes eps * log(eps/r_theta) - eps + r_theta ≈ r_theta, penalizing
the predicted rate for being nonzero where the target is zero. This is
correct behavior — spurious transitions should be suppressed.

## Apply To

All training functions:
- train()
- train_conditional()
- train_flexible_conditional()
- train_film_conditional()
- train_direct_gnn() — N/A (uses KL on distributions, not rates)

## Expected Impact

The rate KL should produce models that are better calibrated in terms
of relative rate magnitudes. Important transitions (large rates) will
be predicted more accurately at the expense of less important ones
(small rates). This should improve the quality of the generated flows,
especially at late times when rates are concentrated on a few transitions.

If rate KL training is unstable (possible due to the log), fall back
to MSE which we know works.
