# Spec: Dirichlet Start Ablation (Ex11e)

## Goal

Find the best starting distribution for posterior sampling by ablating over:
1. **Uniform Dirichlet** with varying concentration: Dirichlet(α, ..., α)
2. **Observation-centered Dirichlet** with varying concentration: Dirichlet(α · N · obs)

Compare all variants on accuracy (TV), peak recovery, calibration (r), and
posterior diversity. Use the Ex11 setup for direct comparison with existing
numbers.

## Background

Ex11d showed that Dirichlet(1, ..., 1) starts cost ~70% relative TV increase
over obs-starts (0.093 vs 0.054), while providing calibration r=0.69. We want
to know if a better choice of starting distribution can close this gap.

Reference numbers:
```
FM obs-start:           TV=0.054, Peak=98%, no uncertainty
FM Dirichlet(1):        TV=0.093, Peak=97%, r=0.69, diversity=0.064
DirectGNN:              TV=0.038, Peak=99%, no uncertainty
```

## Ablation design

### Arm 1: Uniform Dirichlet — Dirichlet(α, ..., α)

The start distribution mean is always uniform (1/N, ..., 1/N). α controls
concentration around uniform.

```python
mu_start = rng.dirichlet(np.full(N, alpha))
```

Values: α ∈ {0.1, 0.5, 1.0, 5.0, 10.0}

- α = 0.1: very spiky starts (near-delta on random nodes)
- α = 0.5: moderately spiky
- α = 1.0: uniform over simplex (current Ex11d setting)
- α = 5.0: concentrated near uniform distribution
- α = 10.0: tightly concentrated near uniform

Expected tradeoff: small α → high diversity but hard transport, large α → easy
transport but low diversity (all starts look the same → no uncertainty).

### Arm 2: Observation-centered Dirichlet — Dirichlet(α · N · obs)

The start distribution mean is the observation. α controls concentration
around the observation.

```python
mu_start = rng.dirichlet(alpha * N * obs)
```

Note: `alpha * N * obs` sums to `alpha * N` (since obs sums to 1), so the
effective Dirichlet concentration parameter is α · N. The mean of this
Dirichlet is exactly `obs`. As α → ∞, starts concentrate on `obs` (recovering
obs-start). As α → 0, starts become diffuse.

Values: α ∈ {0.5, 1.0, 5.0, 10.0, 50.0}

- α = 0.5: fairly diffuse around observation
- α = 1.0: moderate spread around observation
- α = 5.0: fairly tight around observation
- α = 10.0: tight around observation
- α = 50.0: very tight (near obs-start)

Expected: this should dominate Arm 1 because starts are always informative
(centered on observation). The question is where the sweet spot is — enough
noise for meaningful diversity, not so much that accuracy degrades.

### Numerical guard

When obs has near-zero entries, `alpha * N * obs` can underflow. Use:
```python
obs_safe = np.clip(obs, 1e-4, None)
obs_safe /= obs_safe.sum()
mu_start = rng.dirichlet(alpha * N * obs_safe)
```

## Training

**Critical: train a separate model for each (arm, α) combination.** The
starting distribution affects the OT couplings in the training data, so models
trained with different starts see different flow matching targets. You cannot
train one model and evaluate with different starts at test time — the starts
must match between training and inference.

For each combination:
- Use FiLMConditionalGNNRateMatrixPredictor (node_context_dim=1, global_dim=1)
- Same training graphs, source generation, tau range as Ex11/Ex11d
- Same dataset parameters: n_pairs_per_graph=75, n_starts_per_pair=5

### Quick-scan settings (for initial ablation)

Use reduced training to scan the landscape quickly:
- **100 epochs** at **lr=5e-3** (instead of 1000 epochs at 5e-4)
- This gives ~5x higher final loss than full training but should preserve the
  relative ranking across configurations
- Checkpoint: `checkpoints/ex11e_{arm}_{alpha}_100ep_quick.pt`
  e.g., `ex11e_uniform_0.1_100ep_quick.pt`, `ex11e_obsdir_5.0_100ep_quick.pt`

This is 10 quick training runs, each ~10x cheaper than Ex11d.

### Follow-up: full training on best configurations

After identifying the 2-3 most promising (arm, α) combinations from the quick
scan, retrain those at full scale (1000 epochs, lr=5e-4) for final numbers.
Checkpoint: `checkpoints/ex11e_{arm}_{alpha}_1000ep.pt`

**Caveat:** configurations with harder transport (uniform Dirichlet, small α)
may converge slower and look worse in the quick scan than they would at full
training. Interpret the quick-scan ranking with this in mind — if a uniform
Dirichlet configuration looks close to an obs-centered one at 100 epochs, it
may actually win at 1000 epochs.

## Evaluation

For each trained model, evaluate on the same 180 test cases as Ex11/Ex11d.
Use posterior_k=20 samples with the **matching** start distribution.

### Metrics per (arm, α)
- Mean TV (overall and by topology/tau)
- Peak recovery %
- Calibration r (Pearson between posterior std and |error|)
- Posterior diversity (mean pairwise TV among K samples)

### Output

**Console table:**
```
Arm              α      TV     Peak%   Cal-r   Diversity
---------------------------------------------------------
Uniform         0.1     ...    ...     ...     ...
Uniform         0.5     ...    ...     ...     ...
Uniform         1.0     ...    ...     ...     ... (= Ex11d)
Uniform         5.0     ...    ...     ...     ...
Uniform        10.0     ...    ...     ...     ...
Obs-centered    0.5     ...    ...     ...     ...
Obs-centered    1.0     ...    ...     ...     ...
Obs-centered    5.0     ...    ...     ...     ...
Obs-centered   10.0     ...    ...     ...     ...
Obs-centered   50.0     ...    ...     ...     ...
---------------------------------------------------------
FM obs-start     -      0.054  98%     -       -
DirectGNN        -      0.038  99%     -       -
```

**Figure:** `experiments/ex11e_dirichlet_ablation.png` with 4 panels:

**Panel A: TV vs α (two lines, one per arm).**
X-axis: α (log scale). Y-axis: TV. Horizontal reference lines for FM obs-start
(0.054) and DirectGNN (0.038). This is the main result — shows the accuracy
frontier for each arm.

**Panel B: Calibration r vs α (two lines).**
X-axis: α (log scale). Y-axis: Pearson r. Shows how uncertainty quality
degrades with increasing α.

**Panel C: TV vs calibration r (Pareto frontier).**
Each point is one (arm, α) combination. X-axis: calibration r. Y-axis: TV.
Lower-left is better (low TV, high calibration). This directly shows the
accuracy-uncertainty tradeoff and identifies the Pareto-optimal configurations.

**Panel D: Posterior diversity vs α (two lines).**
X-axis: α (log scale). Y-axis: diversity. Shows collapse of diversity at
high α.

## Expected outcomes

**Arm 1 (uniform Dirichlet):** TV should be U-shaped or monotonically
decreasing with α. At very small α, starts are far from sources (high TV).
At very large α, starts are all near-uniform (low TV but low diversity).
Calibration r should decrease with α as diversity collapses.

**Arm 2 (obs-centered Dirichlet):** Should dominate Arm 1 at every α. TV
should decrease monotonically with α (tighter around observation = easier
transport). Calibration should remain reasonable even at moderate α because
the noise is centered on the right place. The sweet spot is likely α ∈ [5, 20]
where TV approaches obs-start levels while calibration is still meaningful.

**Best case scenario:** Obs-centered Dirichlet at some α achieves TV close to
FM obs-start (~0.05) with calibration r > 0.5. This would mean posterior
sampling is nearly free — you get uncertainty at minimal accuracy cost.

**This would change the paper story:** instead of presenting a stark
accuracy-uncertainty tradeoff, we present obs-centered Dirichlet as the
default that provides calibrated uncertainty at negligible cost.

## Implementation

### File: `experiments/ex11e_dirichlet_ablation.py`

The script should:
1. Loop over all (arm, α) combinations
2. For each: build dataset, train model (or load checkpoint), evaluate
3. Collect all results into the summary table and figure
4. Support `--resume` to skip already-trained checkpoints

Reuse infrastructure from ex11d_dirichlet_variant.py. The main change is
parameterizing the start distribution:

```python
def sample_start(obs, N, alpha, arm, rng):
    if arm == 'uniform':
        return rng.dirichlet(np.full(N, alpha))
    elif arm == 'obs_centered':
        obs_safe = np.clip(obs, 1e-4, None)
        obs_safe /= obs_safe.sum()
        return rng.dirichlet(alpha * N * obs_safe)
```

This function is used in both dataset generation (for OT couplings) and
inference (for posterior sampling).
