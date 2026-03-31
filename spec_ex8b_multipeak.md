# Experiment 8.2: `ex8b_multipeak_recovery.py`

## Motivation

Experiment 8 recovered single-peaked distributions from diffused observations.
This worked but the sources were trivial — one dominant node. Experiment 8.2
recovers multi-peak distributions (1-3 peaks) from diffused observations. The
model must reconstruct how many peaks exist, where they are, and how mass is
split among them.

This is a harder and more realistic test of conditional flow matching: the
source is a complex distribution, not just a point.

## Setup

- **Graph**: 5x5 grid graph (25 nodes), unweighted.
- **Diffusion time range**: tau_diff sampled uniformly from [0.3, 1.5]
  (same as updated Ex8).

### Source distributions: 1-3 random peaks

To generate a source distribution:
1. Sample n_peaks uniformly from {1, 2, 3}.
2. Pick n_peaks distinct nodes uniformly at random.
3. Draw mass weights from Dirichlet(alpha=2, size=n_peaks).
   This gives varied splits — sometimes one peak dominates, sometimes
   mass is roughly equal.
4. Scale: peak nodes get 80% of total mass (split by Dirichlet weights),
   remaining 20% distributed uniformly across all other nodes.
5. Add small Gaussian noise (std=0.01), clip, renormalize.

Examples of what this produces:
- 1 peak: similar to Ex8 (one dominant node, easy)
- 2 peaks: mass concentrated at two distant nodes (medium — model must
  identify both locations)
- 3 peaks: mass spread across three nodes (hard — model must reconstruct
  a complex pattern from a blurry observation)

### Training data

Generate 300 source-observation-tau triples:
- 100 with n_peaks=1
- 100 with n_peaks=2
- 100 with n_peaks=3

For each: generate source, sample tau_diff in [0.3, 1.5], diffuse.

### Context

Same as Ex8: per-node features [mu_obs(a), tau_diff].
context_dim = 2.

## Training

- ConditionalGNNRateMatrixPredictor with context_dim=2
- hidden_dim=64, n_layers=4
- 500 epochs, Adam lr=1e-3, batch_size=64
- 7500 training samples (2500 per peak count)
- Diagonal meta-coupling (known pairing)
- 1/(1-t) factorization applied

## Evaluation

### Test cases

30 held-out triples (10 per n_peaks), at tau_diff = 0.8 (moderate diffusion).

For each test case:
1. Start from mu_obs at t=0.
2. Integrate forward conditioned on (mu_obs, tau_diff).
3. Output at t=1 is the recovered source.
4. Compare to true source and exact inverse.

### Metrics

- **TV distance**: between recovered and true source, grouped by n_peaks.
- **Peak count recovery**: the recovered distribution's number of dominant
  nodes (nodes with mass > 0.1, say) should match the true n_peaks.
  This tests whether the model learns to reconstruct the right number of
  modes.
- **Peak location recovery**: for each true peak node, is there a
  recovered peak within Manhattan distance 1? This is a softer metric
  that allows slight spatial shifts.
- **Mass split accuracy**: for 2- and 3-peak sources, compare the
  recovered mass ratios among peaks to the true ratios. Use L1 distance
  between sorted mass vectors.

### Exact baseline

Same as Ex8: mu_source_exact = mu_obs @ expm(-tau_diff * R), clip, renormalize.

## Plots (single figure, 2x3 grid)

- **Panel A**: Training loss curve.

- **Panel B**: Example recoveries. Three rows (one per n_peaks), each showing
  four 5x5 heatmaps: "Observation", "Recovered (learned)", "Recovered (exact)",
  "True source". Shows the model handling 1, 2, and 3 peaks.

- **Panel C**: TV distance grouped by n_peaks. Grouped bar chart with two bars
  per group (learned, exact). TV should increase with n_peaks.

- **Panel D**: Peak count recovery accuracy by true n_peaks. Bar chart.
  How often does the model get the right number of peaks?

- **Panel E**: Peak location recovery (fraction of true peaks with a recovered
  peak within Manhattan distance 1), grouped by n_peaks.

- **Panel F**: Trajectory visualization for a 2-peak case. Five 5x5 heatmaps
  at t = 0, 0.25, 0.5, 0.75, 1.0. Should show mass gradually separating from
  a diffused blob into two distinct peaks.

## Validation checks (print to console)

```
=== Experiment 8.2: Multi-Peak Recovery Results ===
Training: initial loss = X, final loss = Y

By number of peaks (tau_diff=0.8):
  n_peaks=1: TV_learned=X.XX, TV_exact=X.XX, peak_count_acc=XX%
  n_peaks=2: TV_learned=X.XX, TV_exact=X.XX, peak_count_acc=XX%
  n_peaks=3: TV_learned=X.XX, TV_exact=X.XX, peak_count_acc=XX%

Peak location recovery (within Manhattan dist 1):
  n_peaks=1: XX%
  n_peaks=2: XX%
  n_peaks=3: XX%
```

## Expected Outcome

- n_peaks=1 should match Ex8 performance (~0.06-0.10 TV, 100% peak recovery).
- n_peaks=2 should be harder but still work: the model identifies both peaks
  and roughly recovers the mass split. TV ~0.15-0.25.
- n_peaks=3 is the hardest: more peaks to locate and the diffused observation
  is blurrier. TV ~0.20-0.35. Peak count recovery may drop below 100%.
- The trajectory visualization for 2 peaks should show the model learning to
  split a single diffused blob into two separate concentrations — a qualitatively
  interesting behavior.

## Dependencies

No new dependencies. Reuses ConditionalGNNRateMatrixPredictor,
ConditionalMetaFlowMatchingDataset, and train_conditional from Ex8.
