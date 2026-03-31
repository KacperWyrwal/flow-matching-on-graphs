# Fix: Factorize 1/(1-t) From the Learning Target

## Problem

The model is trying to learn rate matrices that scale as 1/(1-t), which
diverge as t -> 1. No finite neural network can output values approaching
infinity, so training on the raw rate matrices fails for the concentrating
direction.

## Fix

Factor out the 1/(1-t) singularity analytically. The model learns the
bounded residual, and 1/(1-t) is applied at inference time.

The marginal rate matrix decomposes as:

    u_t(a,b) = 1/(1-t) * u_tilde_t(a,b)

where u_tilde_t(a,b) is bounded for all t in [0,1] and varies smoothly.

## Changes

### 1. Dataset generation (`meta_fm/dataset.py`)

In all dataset classes (MetaFlowMatchingDataset, ConditionalMetaFlowMatchingDataset,
InpaintingDataset), store the rescaled target:

```python
# Current:
R_target = marginal_rate_matrix_fast(cache, pi, tau)

# New:
R_target_raw = marginal_rate_matrix_fast(cache, pi, tau)
R_target = (1.0 - tau) * R_target_raw  # bounded for all tau
```

This is the ONLY change in dataset generation. Everything else (mu_tau,
context, etc.) stays the same.

### 2. Model output interpretation

No architecture change. The model output is now interpreted as u_tilde
(the bounded part) rather than u (the full rate matrix).

The softplus activation on edge readouts still ensures non-negative off-diagonal
entries, which is correct since u_tilde also has non-negative off-diagonal entries
(it's just u scaled by a positive constant).

### 3. Sampling / inference (`meta_fm/sample.py`)

In sample_trajectory, sample_trajectory_conditional, and
sample_trajectory_guided, multiply the model output by 1/(1-t):

```python
def sample_trajectory(model, mu_start, n_steps=200, device=None):
    dt = 0.999 / n_steps
    t = 0.0
    mu = mu_start.copy()
    trajectory = [mu.copy()]
    
    for _ in range(n_steps):
        # Model predicts u_tilde (bounded)
        u_tilde = model_forward(mu, t)  # (N, N)
        
        # Apply the 1/(1-t) factor analytically
        R = u_tilde / (1.0 - t)
        
        # Euler step
        mu = mu + dt * mu @ R
        mu = np.clip(mu, 0, None)
        mu /= mu.sum()
        
        t += dt
        trajectory.append(mu.copy())
    
    return times, trajectory
```

Same change for conditional and guided variants.

### 4. Training loss

Standard MSE on the rescaled target — NO time weighting needed:

```python
loss = mse_off_diagonal(R_pred, R_target_rescaled)
```

The loss is now well-behaved because R_target_rescaled is bounded.

## Why This Works

The conditional rate matrix is:

    R_t(a,b | i,j) = d(a,j)/(1-t) * R_ab * N_b / N_a

After rescaling by (1-t):

    R_tilde(a,b | i,j) = d(a,j) * R_ab * N_b / N_a

This is INDEPENDENT of t. The only t-dependence in u_tilde comes from the
mixture weights p_t(a|i,j)/p_t(a) in the marginal, which vary smoothly
(binomial in t). So u_tilde is a smooth, bounded function of (mu, t) —
exactly what a neural network can learn.

## Impact

This fix should resolve the training instability in ALL concentrating-direction
experiments (5, 6, 8) without any hyperparameter tuning, architecture changes,
or loss function modifications.

For forward-direction experiments (3, 4, 7) that already work, the rescaling
doesn't hurt — it just changes the scale of the loss, which the optimizer
adapts to automatically.

## Rerun Plan

1. Apply changes to dataset.py and sample.py.
2. Delete all checkpoints (the stored targets have changed scale).
3. Rerun Experiment 8 (conditional source recovery) as the primary validation.
4. If successful, proceed to Experiment 9 (inpainting).
5. Optionally rerun Experiments 3, 4, 7 to verify no regression.
