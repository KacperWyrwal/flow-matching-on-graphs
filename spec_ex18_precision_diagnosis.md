# Spec: Ex18 Precision Diagnosis

## Problem

In Experiment 18, the "Laplacian sharpening" baseline computes the exact inverse
`mu_source = obs @ expm(-tau * R)`. This should recover the source perfectly
(TV = 0), but in practice gives TV ~ 0.04–0.45 depending on tau. We suspect
float32 precision loss in the forward step is the cause. This spec describes a
diagnostic script to verify that hypothesis and, if confirmed, fix the baseline.

## Root Cause Hypothesis

The forward observation is computed in float64 but cast to float32:

```python
def generate_observation(mu_source, R, tau):
    obs = mu_source @ expm(tau * R)
    obs = np.clip(obs, 0, None)
    obs /= obs.sum() + 1e-15
    return obs.astype(np.float32)  # <-- precision loss here
```

The inverse `expm(-tau * R)` amplifies this truncation error, especially for
large tau and sharp source distributions. Additionally, clipping negative values
and renormalizing in both forward and inverse steps introduces further bias.

## Diagnostic Script

Create `experiments/ex18_precision_check.py` that does the following:

### 1. Round-trip test across dtypes and tau values

For a representative set of graphs (reuse `generate_training_graphs` from
`ex17_ot_generalization.py`) and source types (reuse `generate_source_distribution`
from `ex18_source_recovery.py`):

```
tau_values = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
source_types: generate 10 sources per graph (use seed=42)
```

For each (graph, source, tau), compute:

a) **float64 round-trip (no clipping):**
   ```python
   obs64 = mu_source @ expm(tau * R)          # float64
   recovered64 = obs64 @ expm(-tau * R)        # float64, no clip
   tv_64_noclip = total_variation(recovered64 / recovered64.sum(), mu_source)
   ```

b) **float64 round-trip (with clipping, as in current code):**
   ```python
   obs64 = mu_source @ expm(tau * R)
   obs64 = np.clip(obs64, 0, None); obs64 /= obs64.sum()
   recovered64 = obs64 @ expm(-tau * R)
   recovered64 = np.clip(recovered64, 0, None); recovered64 /= recovered64.sum()
   tv_64_clip = total_variation(recovered64, mu_source)
   ```

c) **float32 round-trip (current code path):**
   ```python
   obs32 = (mu_source @ expm(tau * R)).astype(np.float32)
   obs32 = np.clip(obs32, 0, None); obs32 /= obs32.sum()
   recovered32 = obs32.astype(np.float64) @ expm(-tau * R)
   recovered32 = np.clip(recovered32, 0, None); recovered32 /= recovered32.sum()
   tv_32 = total_variation(recovered32, mu_source)
   ```

d) **Condition number of diffusion operator:**
   ```python
   P = expm(tau * R)
   cond = np.linalg.cond(P)
   ```

### 2. Output

**Console table:**
```
tau | TV (f64, no clip) | TV (f64, clip) | TV (f32) | cond(P) | n_negative_entries
----|-------------------|----------------|----------|---------|-------------------
0.1 |     ...           |    ...         |   ...    |   ...   |    ...
...
```

Report mean ± std across all (graph, source) pairs for each tau.

**Plot:** `experiments/ex18_precision_check.png` with 2 panels:

- **Panel A:** TV vs tau for the three round-trip variants (lines with error bands).
  Log-scale y-axis. This directly shows whether float64 eliminates the error.

- **Panel B:** TV (float32 round-trip) vs condition number (scatter, colored by tau).
  This shows whether ill-conditioning explains the error pattern.

### 3. Imports and reuse

```python
# Reuse from existing codebase:
from experiments.ex17_ot_generalization import generate_training_graphs
from experiments.ex18_source_recovery import (
    generate_source_distribution,
    generate_observation,      # for comparison only
    baseline_laplacian_sharpening,
)
from graph_ot_fm import total_variation
```

### 4. What to look for in the results

- If `tv_64_noclip ≈ 0` (< 1e-10) for all tau: the inverse is mathematically
  exact and all error comes from clipping/precision. This confirms the hypothesis.

- If `tv_64_noclip > 0` for large tau: there may be numerical issues in `expm`
  itself for large tau (unlikely but possible for ill-conditioned R).

- If `tv_64_clip ≈ 0` but `tv_32 >> 0`: the float32 cast is the sole culprit.

- If `tv_64_clip >> 0`: clipping in the forward step is also contributing,
  which would mean `expm(tau * R)` produces negative entries (shouldn't happen
  for a valid rate matrix, but worth checking).

- The condition number should correlate with float32 error magnitude.

### 5. If confirmed: fix the baseline

If float64 round-trip gives TV ≈ 0, then:

1. **In `ex18_source_recovery.py`:** Change `generate_observation` to keep
   float64 precision, or at minimum store a float64 copy for baseline evaluation:
   ```python
   def generate_observation(mu_source, R, tau):
       obs = mu_source @ expm(tau * R)
       obs = np.clip(obs, 0, None)
       obs /= obs.sum() + 1e-15
       return obs  # keep float64
   ```

2. **In `baseline_laplacian_sharpening`:** Ensure input is float64 before
   inverting.

3. **In the dataset and model pipeline:** Cast to float32 only when creating
   tensors for model input, not when computing baselines.

4. **Important:** The FM model still receives float32 observations as input
   (this is correct — the model should work with float32). The fix only affects
   the **baseline evaluation** so that it reflects the true capability of exact
   inversion, not float32 artifacts.

5. After fixing, rerun Ex18 evaluation (not training — the training data is
   fine as-is since the model sees float32 observations). The Laplacian baseline
   should now show TV ≈ 0 for all tau, making it the true "oracle with full
   model knowledge" baseline.

### 6. Impact on paper narrative

If confirmed, the paper should present the Laplacian baseline as an **oracle**
that has access to the exact rate matrix R — something the learned methods don't
need. The comparison becomes: FM achieves TV of X without knowing R, while
perfect recovery requires knowing R exactly. This is a cleaner story than the
current one where Laplacian appears to beat FM but is actually hampered by a
precision bug.
