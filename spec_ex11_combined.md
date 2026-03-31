# Experiment 11: `ex11_combined_generalization.py`

## Motivation

Experiments 8.2 and 10 demonstrated multi-peak recovery and topology
generalization separately. This experiment combines them: recover multi-peak
sources with variable diffusion times across unseen graph topologies. This is
the hardest synthetic task we've attempted — the model must simultaneously:

1. Adapt to an unseen graph topology
2. Infer the number and locations of peaks from a blurry observation
3. Account for different amounts of diffusion
4. Reconstruct the full mass distribution, not just peak locations

A skeptic can no longer dismiss this as "just sharpening a bump" — with
multiple overlapping peaks at high diffusion times on unfamiliar graphs,
the task requires genuine understanding of transport dynamics on graphs.

## Setup

### Training graphs (same as Ex10)

13 topologies: grid (3x3, 4x4, 5x5), cycle (8, 10, 12), path (6, 8, 10),
star (7, 9), complete bipartite (3_3, 4_4).

### Test graphs (same as Ex10)

4 held-out topologies: grid_3x5, cycle_15, barbell(4,3), Petersen.

### Source distributions

1-3 peaks per source, mass split via Dirichlet(alpha=2), 80/20 peak/background
split. Same as Ex8.2 and Ex10.

### Diffusion time

tau_diff sampled uniformly from [0.3, 1.5] during training.
Test at tau_diff in {0.5, 1.0, 1.4} to show performance across difficulty.

### Context

Per node: [mu_obs(a), tau_diff] — same as Ex8 and Ex10.
context_dim = 2.

## Training

- FlexibleConditionalGNNRateMatrixPredictor with context_dim=2
- hidden_dim=64, n_layers=4
- Use TopologyGeneralizationDataset (already supports multi-peak and
  variable tau_diff)
- 1000 samples per graph × 13 graphs = 13000 total samples
- 75 source-obs pairs per graph (more than Ex10's 50 to cover the
  combinatorial space of n_peaks × tau_diff)
- 1000 epochs (with gradient clipping max_norm=1.0, lr=5e-4)
- 1/(1-t) factorization applied

This is essentially the same training setup as Ex10 but with more pairs
per graph to cover the multi-peak × variable-tau space.

## Evaluation

### Test matrix

For each of the 4 test topologies, evaluate at:
- tau_diff in {0.5, 1.0, 1.4}
- n_peaks in {1, 2, 3}
- 5 test cases per (topology, tau_diff, n_peaks) combination
- Total: 4 × 3 × 3 × 5 = 180 test cases

### Baselines

1. **Exact inverse**: mu_obs @ expm(-tau_diff * R), clip, renormalize.
2. **Argmax baseline**: for peak location only — take the top-k nodes by
   mass in the observation (where k = true n_peaks). This is what a skeptic
   would suggest as the simple approach.

### Metrics

- **TV distance** to true source (primary metric)
- **Peak recovery (top-k)**: fraction of true peaks found in top-k of recovered
- **Peak location (top-k, Manhattan ≤ 1)**: softer spatial metric

## Plots (single figure, 2x3 grid)

- **Panel A**: Training loss curve.

- **Panel B**: TV heatmap. Rows = test topologies (4), columns = tau_diff (3).
  Cell color = mean TV. Marginal means on edges. Shows which combinations
  are hardest. A clean gradient from easy (low tau, simple topology) to hard
  (high tau, complex topology).

- **Panel C**: TV by n_peaks, aggregated across test topologies and tau_diff.
  Three bars (1 peak, 2 peaks, 3 peaks) for learned model, plus three bars
  for exact inverse. Shows the difficulty scaling with source complexity.

- **Panel D**: Peak recovery (top-k) heatmap. Same layout as Panel B.
  Should be mostly green (high accuracy) with degradation only at the
  hardest combinations.

- **Panel E**: Comparison to argmax baseline for peak LOCATION recovery.
  At low tau_diff, argmax on the observation should work well (peaks are
  still visible). At high tau_diff, argmax fails (peaks merge) but the
  learned model should still recover them. Two lines: learned vs argmax,
  plotted against tau_diff. This is where the model proves its value over
  simple heuristics.

- **Panel F**: Trajectory gallery. Show 3 examples on different test graphs:
  one 1-peak case, one 2-peak case where peaks visibly separate during the
  flow, and one 3-peak case. Each example shows the graph at t=0, 0.5, 1.0.
  Use networkx graph drawings with node colors.

## Validation checks (print to console)

```
=== Experiment 11: Combined Generalization ===

Overall (180 test cases):
  Mean TV (learned):    X.XXXX
  Mean TV (exact):      X.XXXX
  Peak recovery (learned): XX.X%
  Peak recovery (argmax):  XX.X%

By test topology:
  grid_3x5:  TV=X.XX, peak=XX%
  cycle_15:  TV=X.XX, peak=XX%
  barbell:   TV=X.XX, peak=XX%
  petersen:  TV=X.XX, peak=XX%

By tau_diff:
  tau=0.5: TV=X.XX, peak_learned=XX%, peak_argmax=XX%
  tau=1.0: TV=X.XX, peak_learned=XX%, peak_argmax=XX%
  tau=1.4: TV=X.XX, peak_learned=XX%, peak_argmax=XX%

By n_peaks:
  1 peak:  TV=X.XX, peak=XX%
  2 peaks: TV=X.XX, peak=XX%
  3 peaks: TV=X.XX, peak=XX%
```

## Expected Outcome

- At tau=0.5 with 1 peak: both learned and argmax should do well.
- At tau=1.4 with 3 peaks: argmax should fail (overlapping diffused bumps
  are indistinguishable) while the learned model recovers peaks using its
  understanding of transport dynamics. This is the key comparison.
- Across topologies: performance should be comparable between training and
  test topologies (as demonstrated in Ex10).
- The TV heatmap should show a smooth gradient from easy to hard, with no
  catastrophic failures on any test topology.

## Dependencies

No new dependencies. Reuses all infrastructure from Ex10.
Same FlexibleConditionalGNNRateMatrixPredictor, TopologyGeneralizationDataset,
train_flexible_conditional, sample_trajectory_flexible.
