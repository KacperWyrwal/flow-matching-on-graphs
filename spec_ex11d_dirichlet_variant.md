# Spec: Ex11 Dirichlet-Start Variant (Ex11d)

## Goal

Isolate the performance cost of Dirichlet-start posterior sampling vs
observation-start point estimation. Run Ex11 with Dirichlet random starts
instead of starting from the observation, keeping everything else identical.
This directly explains the performance gap between Ex11 and Ex18.

## Background

- **Ex11 (current):** Flow goes from observation → source. The model learns a
  correction to the diffused signal. No uncertainty quantification.
- **Ex18:** Flow goes from Dirichlet random draw → source, with observation as
  side context. Produces posterior samples but is a harder learning problem.
- **Ex11d (this experiment):** Flow goes from Dirichlet random draw → source,
  with observation as side context. Same graphs, sources, and tau range as Ex11.

## What to change from Ex11

### Training data generation

Replace the current flow matching sample generation (which uses observation as
the starting distribution) with the Ex18-style Dirichlet-start approach:

```python
# CURRENT Ex11: OT coupling from observation to source
coupling = compute_ot_coupling(mu_obs, mu_source, graph_struct)

# NEW Ex11d: OT coupling from Dirichlet draw to source
mu_start = rng.dirichlet(np.full(N, dirichlet_alpha))
coupling = compute_ot_coupling(mu_start, mu_source, graph_struct)
```

For each (graph, source, tau) training pair, generate `n_starts_per_pair=5`
independent Dirichlet starts (matching Ex18's setup).

### Context

Switch to FiLM-style context to match Ex18:
- `node_ctx = obs[:, None]` — shape (N, 1), observation value per node
- `global_ctx = [tau]` — shape (1,), diffusion time

This requires using `FiLMConditionalGNNRateMatrixPredictor` with
`node_context_dim=1, global_dim=1` instead of
`FlexibleConditionalGNNRateMatrixPredictor` with `context_dim=2`.

### Inference

Use posterior sampling (matching Ex18):
```python
posterior_k = 20
dirichlet_alpha = 1.0
mu_starts = [rng.dirichlet(np.full(N, dirichlet_alpha)) for _ in range(posterior_k)]
fm_samples = sample_posterior_film(model, mu_starts, node_ctx, global_ctx, ...)
fm_mean = fm_samples.mean(axis=0)
```

### Everything else stays the same

- Same 13 training topologies, same 4 test topologies
- Same tau range: training τ ∈ [0.3, 1.5], test τ ∈ {0.5, 1.0, 1.4}
- Same source distributions: 1, 2, 3 peaks with Dirichlet weights
- Same hyperparameters: hidden_dim=64, n_layers=4, lr=5e-4
- Same number of epochs: 1000
- Same evaluation metrics: TV, peak recovery, peak location

## Evaluation

### DirectGNN baseline: reuse from Ex11, do NOT retrain

DirectGNN is a one-shot predictor — it takes (observation, tau) as input and
outputs the source estimate directly. It has no starting distribution and no
trajectory. Therefore it is **identical** in the obs-start and Dirichlet-start
settings. Do not retrain DirectGNN for Ex11d.

Instead, either:
- Load Ex11's DirectGNN checkpoint and re-evaluate on the same test cases, or
- Hard-code Ex11's DirectGNN numbers for the comparison table

The Ex11 DirectGNN reference numbers are:
```
Overall:   TV=0.0377, Peak=98.7%
grid_3x5:  TV=0.0171, Peak=100%
cycle_15:  TV=0.0246, Peak=100%
barbell:   TV=0.0965, Peak=96%
petersen:  TV=0.0126, Peak=99%
tau=0.5:   TV=0.0200, Peak=99%
tau=1.0:   TV=0.0296, Peak=100%
tau=1.4:   TV=0.0635, Peak=97%
```

### Three-way comparison

The goal is a table with three columns: FM obs-start (Ex11), FM Dirichlet-start
(Ex11d), and DirectGNN (from Ex11). This shows both the cost of posterior
sampling AND the gap to the direct regression baseline.

### Summary figure with 4 panels

**Panel A: TV comparison bar chart.**
Three side-by-side bars (FM obs-start, FM Dirichlet-start, DirectGNN) grouped
by test topology. Include exact inverse (TV≈0) as a horizontal reference line.

**Panel B: TV by tau.**
Three lines (FM obs-start, FM Dirichlet-start, DirectGNN) across
τ ∈ {0.5, 1.0, 1.4}. Shows whether the Dirichlet-start penalty grows with
difficulty and how both FM variants compare to DirectGNN.

**Panel C: Peak recovery comparison.**
Same three-way layout as Panel A but for peak recovery accuracy.

**Panel D: Calibration (Ex11d only).**
Scatter of posterior std vs |error|, with Pearson r. This is the unique payoff
of the Dirichlet approach — something neither FM obs-start nor DirectGNN can
provide. If calibration is strong (r > 0.7), this justifies the accuracy cost.

### Console output

Print a three-way comparison table:
```
                    FM obs-start    FM Dirichlet     DirectGNN
                    TV    Peak%     TV    Peak%  r   TV    Peak%
grid_3x5           0.035 97%       ...   ...   ...  0.017 100%
cycle_15           0.013 100%      ...   ...   ...  0.025 100%
barbell            0.118 97%       ...   ...   ...  0.097 96%
petersen           0.048 99%       ...   ...   ...  0.013 99%
Overall            0.054 98%       ...   ...   ...  0.038 99%
```

## Implementation notes

### File: `experiments/ex11d_dirichlet_variant.py`

This should be a standalone script that:
1. Imports graph definitions and helper functions from `ex11_combined.py`
2. Builds a new dataset with Dirichlet starts (reuse `DiffusionSourceDataset`
   from `ex18_source_recovery.py` as a reference, but adapted to Ex11's graph
   set and source generation)
3. Trains a FiLMConditionalGNNRateMatrixPredictor
4. Evaluates with posterior sampling
5. Loads Ex11's checkpoint to get the observation-start results for comparison
   (or re-evaluates Ex11's model on the same test cases)

### Checkpoint path
`checkpoints/ex11d_dirichlet_film_1000ep_h64_l4.pt`

### Dataset parameters (match Ex18 structure)
```python
n_samples_per_graph = 1000  # match Ex11
n_pairs_per_graph = 75      # match Ex11
n_starts_per_pair = 5       # Dirichlet starts per pair
dirichlet_alpha = 1.0       # uninformative Dirichlet
n_samples = n_samples_per_graph * len(TRAINING_GRAPHS)  # total FM samples
```

### Key: ensure fair comparison

The comparison is only valid if:
- Both models see the same (graph, source, tau) pairs during training
- Both models train for the same number of epochs with the same optimizer
- The only difference is the starting distribution and context mechanism

To ensure this, use the same random seed (42) for source generation in both.
The Dirichlet variant just adds the random starts on top.

## Expected outcomes

The key numbers to watch are FM Dirichlet-start TV relative to both FM obs-start
(0.054) and DirectGNN (0.038):

**If Dirichlet starts cause a large TV increase** (e.g., Ex11d TV > 0.15):
The posterior framing is expensive. FM falls far behind DirectGNN. The paper
should use obs-start FM for point estimation experiments and reserve Dirichlet
starts only for experiments where uncertainty quantification is the explicit
goal (e.g., Ex13b, Ex14b posterior sampling).

**If Dirichlet starts cause a moderate TV increase** (e.g., Ex11d TV ~ 0.07-0.10):
There is a real accuracy-uncertainty tradeoff. FM Dirichlet is worse than
DirectGNN on accuracy but provides calibrated uncertainty. The paper frames
this honestly: use DirectGNN or FM obs-start when you only need a point
estimate, use FM Dirichlet when you need posterior samples.

**If Dirichlet starts cause a small TV increase** (e.g., Ex11d TV ~ 0.055-0.07):
The posterior framing is nearly free. FM Dirichlet is still worse than DirectGNN
on accuracy, but the gap is modest and the uncertainty payoff justifies it.
The paper uses Dirichlet starts as the default since you get calibrated
uncertainty at minimal cost.

**Calibration quality matters independently.** Even if Ex11d TV is worse than
DirectGNN, a strong calibration r (> 0.7) makes the Dirichlet approach
scientifically valuable — you can't get uncertainty from DirectGNN at all.
If calibration is weak (r < 0.5), the Dirichlet approach is paying an accuracy
cost for unreliable uncertainty, which is a bad trade.

**Bonus:** If calibration r in Ex11d is substantially higher than Ex18's
r=0.505, that suggests the calibration issue in Ex18 is also about undertraining
rather than a fundamental limitation of the approach.
