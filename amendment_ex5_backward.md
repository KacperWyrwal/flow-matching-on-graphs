# Amendment: Experiment 5 — Source Localization via Dedicated Backward Model

## Motivation

The original Experiment 5 attempted source localization by integrating the
forward-trained model backward in time. This failed catastrophically (10% peak
recovery, TV ~0.8) because backward integration through the 1/(1-t) singularity
is numerically unstable — the forward flow is contractive near t=1, making the
backward flow expansive and amplifying small model errors exponentially.

The fix: train a separate backward model that directly learns to transport
observed (spread) distributions to source (peaked) distributions. This keeps
all integration in the stable forward direction.

## Key Insight

On an undirected graph, the reference random walk is identical to its time
reversal (R is symmetric off-diagonal). So the entire framework — geodesic
structure, conditional rate matrices, costs, OT couplings — applies unchanged.
We just swap source and target distributions.

## Changes to `ex5_source_localization.py`

### Setup

- Cycle graph with N=6 nodes, unweighted (same as before)
- **Backward model source (Pi_0_back):** 50 approximately uniform distributions
  (these represent observed/measured endpoints). Generate identically to the
  target distributions in Experiment 3.
- **Backward model target (Pi_1_back):** 50 peaked distributions at various
  nodes (these represent the sources we want to recover). Generate identically
  to the source distributions in Experiment 3.
- Note: this is literally the Experiment 3 setup with source and target swapped.

### Training the Backward Model

1. Compute the graph-level cost matrix c(i,j) (same as before — graph is the
   same, cost is symmetric for undirected graphs so c(i,j) = c(j,i)).
2. For each pair (mu_spread, mu_peaked) of backward-source and backward-target:
   a. Compute the graph-level OT coupling pi(i,j) transporting mu_spread to
      mu_peaked.
   b. Compute the graph-level OT distance W(mu_spread, mu_peaked).
3. Solve the meta-OT coupling between Pi_0_back and Pi_1_back.
4. Pre-generate 5000 training samples:
   For each sample:
     a. Draw (mu_spread, mu_peaked) from the meta-OT coupling
     b. Draw tau ~ U[0, 1]
     c. Compute mu_tau = marginal_distribution(graph, pi, tau)
        where pi is the graph-level coupling from mu_spread to mu_peaked
     d. Compute R_target = marginal_rate_matrix(graph, pi, tau)
     e. Store (mu_tau, tau, R_target)
5. Instantiate a NEW GNNRateMatrixPredictor (same architecture as Ex3/Ex4).
6. Train for 500 epochs (same hyperparameters). If loss hasn't converged,
   extend to 1000 epochs.
7. Save checkpoint as checkpoints/meta_model_backward_gnn.pt

### Source Localization Procedure

For each test case:
1. Start with a ground truth peaked source mu_source.
2. Compute the exact forward flow from mu_source to get mu_observed at t=1.
   (Use the forward graph-level solver — compute coupling between mu_source
   and some target, evolve forward.)
   
   Actually simpler: we already know from Ex3 that the forward flow takes
   peaked distributions to near-uniform ones. So mu_observed is approximately
   the marginal_distribution at t=0.99 from the forward solver.
   
   Even simpler for this experiment: just use mu_observed directly as a
   near-uniform distribution and verify recovery of a peaked source. We don't
   strictly need to run the forward model — we can define test pairs:
   
   - mu_observed: a near-uniform distribution (draw from same process as
     Pi_0_back but held out from training)
   - mu_source_true: the peaked distribution it should map to (draw from same
     process as Pi_1_back but held out from training)
   - Verify: does the backward model, starting from mu_observed and integrating
     forward to t=1, recover something close to mu_source_true?

3. Add observation noise to mu_observed:
   mu_noisy = mu_observed + N(0, sigma^2), clip, renormalize
4. Run the backward model FORWARD from mu_noisy:
   Integrate dp/dt = p @ R_backward(p, t) from t=0 to t=1 using Euler method.
   This is standard forward integration — stable, no singularity issues.
5. The output at t=1 is the recovered source mu_recovered.
6. Compare mu_recovered to mu_source_true.

### Test Cases

Generate 10 held-out test pairs:
- 10 held-out near-uniform distributions (not in training set)
- For each, a "true source" peaked distribution that is the OT-optimal
  partner (solve the graph-level OT coupling between the held-out uniform
  and each training peaked distribution, pick the best match, or simply
  generate a new peaked distribution as ground truth and verify the backward
  model transports toward it)

Simpler approach for clean evaluation:
- Generate 10 peaked distributions (held out) as ground truth sources.
- For each, compute the exact forward flow to get the corresponding
  near-uniform observation at t=0.99.
- Run the backward model starting from this observation.
- Compare the backward model's output to the true peaked source.

This creates a clean round-trip test: forward exact solver produces the
observation, backward learned model recovers the source.

### Noise Levels

Same as before: sigma = 0.0, 0.02, 0.05

### Plots (single figure, 2x2 grid)

- **Panel A**: Forward trajectory of the backward model for ONE test case at
  sigma=0. Show bar plots at t = 0.0, 0.25, 0.5, 0.75, 1.0 (5 snapshots).
  The distribution should start near-uniform and progressively concentrate
  toward the true source node. Title should indicate the true peak node.

- **Panel B**: Overlay of recovered source vs true source for 3 test cases at
  sigma=0. Grouped bar chart: solid = true, hatched = recovered.
  These should now match closely (unlike the failed backward-integration version).

- **Panel C**: TV distance between recovered and true source vs noise level.
  Scatter plot with 10 dots per noise level, horizontal bars for means.
  Expected: TV ~0.1-0.2 at sigma=0 (comparable to Ex3/Ex4 forward accuracy),
  graceful degradation with noise.

- **Panel D**: Peak node recovery accuracy (%) vs noise level. Bar chart.
  Expected: high accuracy (>80%) at sigma=0, gradual decrease with noise.

### Validation Checks (print to console)

For each noise level:
- Mean TV distance (recovered vs true source)
- Std TV distance
- Peak recovery accuracy (%)
- List of (true peak node, recovered peak node) for each test case

### Expected Outcome

The backward model should perform comparably to the forward model in Ex3/Ex4,
since it's the same framework with swapped endpoints. At zero noise, TV distances
should be in the 0.05-0.15 range and peak recovery should be >80%. The key
demonstration is that this works where naive backward integration completely
failed.

### Comparison to Failed Approach

The final output of this experiment should include a brief printed comparison:

```
=== Source Localization Results ===
Method 1 (backward integration of forward model):
  Mean TV at sigma=0: ~0.81  [FAILED]
  Peak recovery at sigma=0: 10%

Method 2 (dedicated backward model):
  Mean TV at sigma=0: [result]
  Peak recovery at sigma=0: [result]%
```

To produce the Method 1 numbers, either load them from the previous Ex5 run
or re-run the backward integration briefly. This comparison is the punchline.

## File Changes

- `ex5_source_localization.py`: Complete rewrite per above spec.
- `checkpoints/meta_model_backward_gnn.pt`: New checkpoint for backward model.
- No changes to library code (`graph_ot_fm/` or `meta_fm/`). The backward model
  uses the exact same GNNRateMatrixPredictor class and training loop — only the
  training data has swapped source/target roles.
