# Experiment 9: `ex9_inpainting.py`

## Motivation

Apply conditional flow matching to inpainting: given a distribution on a graph
with some node values masked (corrupted), reconstruct the full distribution.
No exact solution exists — the model must use its learned prior over valid
distributions to fill in the gaps.

This is the payoff experiment: it demonstrates a genuinely novel capability
of the framework that simpler methods cannot replicate.

Depends on Experiment 8 validating the conditional architecture.

---

## Setup

- **Graph**: 3x3 grid graph (9 nodes). Same as Experiment 7.
- **Data distribution**: Community-structured distributions from Experiment 7
  (4 corner communities: A={0,1,3}, B={1,2,5}, C={3,6,7}, D={5,7,8},
  Dirichlet(alpha=5) weights on community nodes, small noise elsewhere).
- **Corruption process**: Randomly mask M=3 nodes (out of 9) by setting their
  mass to 0 and renormalizing the remaining nodes.

## Context Features

Per node: [mu_corrupted(a), mask(a)]

context_dim = 2. The first feature is the observed value (0 for masked nodes,
renormalized value for observed nodes). The second is the binary mask
(1 = observed, 0 = masked).

Total input per node: [mu(a), t, mu_corrupted(a), mask(a)] = 4 features.

## Training Data Generation

Generate 200 clean community distributions (50 per community).

For each clean distribution, generate multiple corruption variants with
different random masks. This gives the model experience with different
masking patterns.

For each training sample:
1. Draw a clean community distribution mu_clean.
2. Draw a random mask: choose 3 of 9 nodes to mask.
3. Corrupt: set masked nodes to 0, renormalize remaining nodes to get
   mu_corrupted.
4. Compute graph-level OT coupling from mu_corrupted to mu_clean.
5. Sample flow time tau ~ U[0, 0.999].
6. Compute mu_tau and R_target along the OT flow at time tau.
7. Build context: for each node a, context = [mu_corrupted(a), mask(a)].
8. Store (mu_tau, tau, context, R_target).

Use diagonal meta-coupling: each (corrupted_k, clean_k) pair is matched
to itself.

Total: 10000 training samples (multiple mask variants per clean distribution).

## Dataset

Reuse ConditionalMetaFlowMatchingDataset from Experiment 8, or extend it
slightly to handle the mask-based corruption:

```python
class InpaintingDataset(torch.utils.data.Dataset):
    """
    Generates training data for distribution inpainting.
    
    Constructor args:
        graph: GraphStructure
        clean_distributions: list of np.ndarray (N,)
        n_masks_per_dist: int = 50 — number of random masks per distribution
        n_masked_nodes: int = 3
        n_samples: int = 10000
        seed: int = 42
    
    For each sample:
        1. Pick a random clean distribution
        2. Pick a random mask (or reuse a pre-generated one)
        3. Corrupt the distribution
        4. Compute OT flow from corrupted to clean
        5. Sample along the flow
        6. Store (mu_tau, tau, context=[corrupted, mask], R_target)
    
    __getitem__ returns:
        mu: (N,) — distribution at time tau
        tau: (1,) — flow time
        context: (N, 2) — [corrupted value, mask] per node
        R_target: (N, N) — target rate matrix
    """
```

## Training

- ConditionalGNNRateMatrixPredictor with context_dim=2
  (same architecture as Experiment 8, same context_dim by coincidence)
- hidden_dim=64, n_layers=4
- 500 epochs, Adam lr=1e-3, batch_size=64
- 10000 training samples

Uses train_conditional from Experiment 8 — no new training code needed.

## Evaluation

### Test cases

40 held-out clean distributions (10 per community), each with a random mask
of 3 nodes. These specific distributions and masks were not seen during training.

For each test case:
1. Corrupt the clean distribution with the mask.
2. Start from mu_corrupted at t=0.
3. Integrate forward conditioned on (mu_corrupted, mask).
4. Output at t=1 is the inpainted distribution.
5. Compare to the true clean distribution.

### Naive baseline

Fill masked nodes with the mean value of observed nodes:
```python
mu_naive = mu_corrupted.copy()
mean_observed = mu_corrupted[mask == 1].mean()
mu_naive[mask == 0] = mean_observed
mu_naive /= mu_naive.sum()
```

This baseline has no knowledge of community structure. The learned model
should substantially outperform it.

### Metrics

- **TV distance**: between inpainted and true clean distribution.
- **Masked node TV**: TV computed only on the masked nodes.
  Isolates inpainting quality from unchanged observed nodes.
  masked_tv = 0.5 * sum_{masked a} |inpainted(a) - true(a)|
- **Community preservation**: does the inpainted distribution still belong
  to the same community as the original? Classify by highest total mass
  in each corner region.

### Plots (single figure, 2x3 grid)

- **Panel A**: Training loss curve.

- **Panel B**: Example inpainting cases. For 4 test cases (one per community),
  show three 3x3 heatmaps in a row: "Corrupted", "Inpainted", "True".
  Mark masked nodes with an X overlay on the corrupted heatmap.
  Layout: 4 rows x 3 cols of mini heatmaps.

- **Panel C**: TV distance histogram for all 40 test cases. Overlaid with
  naive baseline TV distances. Learned model should be substantially better.

- **Panel D**: Community preservation accuracy. Bar chart showing fraction
  of inpainted distributions correctly classified per community, plus overall.
  Should be >80%.

- **Panel E**: Masked-node-only TV by community. Box plot grouped by community.
  Tests whether some communities are harder to inpaint (corner/edge effects).

- **Panel F**: Ablation — performance vs number of masked nodes.
  Run evaluation with M = 1, 2, 3, 4, 5 masked nodes.
  Plot mean TV and community accuracy vs M.
  M=1 should be near-perfect, M=5 (majority masked) harder but still above
  naive baseline. Shows graceful degradation.

### Validation checks (print to console)

```
=== Experiment 9: Inpainting Results ===
Mean TV (learned):  X.XXXX ± X.XXXX
Mean TV (naive):    X.XXXX ± X.XXXX
Improvement:        XX.X%

Community preservation accuracy:
  Overall:  XX%
  Com A:    XX%
  Com B:    XX%
  Com C:    XX%
  Com D:    XX%

Masked-node TV (learned): X.XXXX ± X.XXXX
Masked-node TV (naive):   X.XXXX ± X.XXXX

Ablation (M = number of masked nodes):
  M=1: TV=X.XX, community_acc=XX%
  M=2: TV=X.XX, community_acc=XX%
  M=3: TV=X.XX, community_acc=XX%
  M=4: TV=X.XX, community_acc=XX%
  M=5: TV=X.XX, community_acc=XX%
```

## Expected Outcome

The model should successfully inpaint masked nodes by leveraging community
structure. Key behaviors:
- If unmasked nodes indicate Community A (mass on nodes 0, 1), the model fills
  node 3 with high mass even though it's masked.
- The naive baseline fills uniformly and cannot do this.
- Community preservation should be high (>80%) at M=3.
- Graceful degradation: performance decreases with more masked nodes but stays
  above baseline even at M=5 (only 4 of 9 observed).

This demonstrates that conditional flow matching on graphs can solve
underdetermined inverse problems by leveraging learned structural priors —
a capability unique to the meta-level framework.

## Dependencies

No new dependencies. Reuses ConditionalGNNRateMatrixPredictor, train_conditional,
and sample_trajectory_conditional from Experiment 8.
