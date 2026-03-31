# Updated Specs: Experiments 8 and 9

---

## Experiment 8 Update: Narrower Diffusion Range

### Problem

The 5x5 grid has a short mixing time. At tau_diff >= 2.0, the diffused
distribution is essentially uniform and the source is unrecoverable — the
high TV at these diffusion times reflects an impossible problem, not a model
failure.

### Changes

1. **Diffusion time range**: Change from [0.5, 4.0] to [0.3, 1.5].
   At tau_diff=1.5 on a 5x5 grid, the bump is spread but still clearly
   localized. At tau_diff=0.3, it's barely spread.

2. **Test diffusion times**: Change from {1.0, 2.0, 3.0, 3.5} to
   {0.5, 0.8, 1.2, 1.4}. The last value (1.4) is near the top of the
   training range and tests the hardest well-posed regime. None of these
   exact values appear in training (continuous uniform sampling).

3. **Remove the interpolation framing**: All test tau values are within the
   training range [0.3, 1.5], so there's no out-of-range interpolation test.
   Instead, the experiment shows performance across a range of difficulty
   levels, all of which should be solvable.

### Expected outcome with these changes

- Peak recovery should be high (>80%) at all test diffusion times.
- TV should increase with tau_diff but stay reasonable (<0.4).
- The learned model should match or slightly trail the exact inverse at all
  levels (since the exact inverse is well-behaved in this range).
- Training loss should converge more cleanly because the target rate matrices
  are more moderate in magnitude (less diffusion to undo).

### Code changes in `ex8_conditional_source_recovery.py`

```python
# Change these lines:
tau_diff_range = (0.3, 1.5)  # was (0.5, 4.0)

# Change test diffusion times:
test_tau_diffs = [0.5]*5 + [0.8]*5 + [1.2]*5 + [1.4]*5
```

Update plot labels and console output accordingly. Everything else unchanged.

Delete the old checkpoint and rerun.

---

## Experiment 9 Update: Inpainting with Classifier-Free Guidance

### Classifier-Free Guidance Overview

During training, randomly drop the context (replace with zeros) with
probability p_drop = 0.15. This teaches the model to produce both conditional
and unconditional rate matrices.

At inference, compute both and interpolate:

    R_guided = (1 + w) * R_conditional - w * R_unconditional

where w is the guidance weight:
- w = 0: pure conditional (use observations as-is)
- w > 0: amplify the effect of conditioning (trust observations more)
- w < 0: dampen conditioning (rely more on prior)

### Architecture Changes

ConditionalGNNRateMatrixPredictor needs no architecture changes. The context
dropping is handled purely in the training loop and sampling code.

### Training Changes

In `train_conditional`, add context dropping:

```python
def train_conditional(model, dataset, n_epochs=500, batch_size=64, lr=1e-3,
                      device=None, context_drop_prob=0.0):
    """
    If context_drop_prob > 0, randomly replace context with zeros for that
    fraction of samples in each batch. This enables classifier-free guidance
    at inference time.
    """
    # In the training loop:
    for mu, tau, context, R_target in dataloader:
        # Context dropping
        if context_drop_prob > 0:
            drop_mask = torch.rand(mu.shape[0], 1, 1, device=device) < context_drop_prob
            context = context * (~drop_mask).float()  # zero out entire context for dropped samples
        
        R_pred = model(mu, tau, context)
        loss = mse_off_diagonal(R_pred, R_target)
        # ... rest unchanged
```

### Sampling Changes

Add guided sampling to `meta_fm/sample.py`:

```python
def sample_trajectory_guided(model, mu_start, context, guidance_weight=0.0,
                              n_steps=200, device=None):
    """
    Integrate with classifier-free guidance.
    
    At each step:
        R_cond = model(mu, t, context)
        R_uncond = model(mu, t, zeros_like(context))
        R = (1 + w) * R_cond - w * R_uncond
        
        # Ensure valid rate matrix after interpolation
        R_off_diag = R with diagonal zeroed
        R_off_diag = clamp(R_off_diag, min=0)  # guidance can create negatives
        R_diagonal = -row_sum(R_off_diag)
        
        mu_next = mu + dt * mu @ R
        clip, renormalize
    
    When guidance_weight=0, this reduces to standard conditional sampling
    (R_cond only, no unconditional evaluation — skip for efficiency).
    """
```

Note the clamping step: the linear combination (1+w)*R_cond - w*R_uncond can
produce negative off-diagonal entries, which violate rate matrix validity.
Clamping to non-negative and recomputing the diagonal fixes this.

### Experiment 9 Training

```python
# In ex9_inpainting.py:
history = train_conditional(model, dataset, n_epochs=500, batch_size=64,
                            lr=1e-3, device=device,
                            context_drop_prob=0.15)  # enable CFG
```

### Experiment 9 Evaluation

Evaluate at multiple guidance weights: w in {0.0, 0.5, 1.0, 2.0, 3.0}.

For each test case and each guidance weight:
1. Start from corrupted distribution.
2. Integrate with sample_trajectory_guided at the given w.
3. Compare to true clean distribution.

### Updated Plots for Experiment 9 (single figure, 2x3 grid)

- **Panel A**: Training loss curve.

- **Panel B**: Example inpainting cases. For 4 test cases (one per community),
  show three 3x3 heatmaps per row: "Corrupted", "Inpainted (w=best)", "True".
  Mark masked nodes with X overlay on the corrupted heatmap.
  Use the guidance weight that gives best average performance.
  Layout: 4 rows x 3 cols of mini heatmaps.

- **Panel C**: TV distance vs guidance weight. Line plot with error bars
  (mean ± std over 40 test cases). Should show a U-shape or monotonic trend
  with an optimal w somewhere in the tested range.

- **Panel D**: Community preservation accuracy vs guidance weight. Line plot.
  Shows how guidance affects the model's ability to maintain community structure.

- **Panel E**: Comparison to naive baseline. At the best guidance weight,
  show histograms of TV for learned model vs naive baseline (fill masked nodes
  with mean of observed). Learned should be substantially better.

- **Panel F**: Ablation — performance vs number of masked nodes (M=1,2,3,4,5)
  at the best guidance weight. Plot mean TV and community accuracy vs M.
  Shows graceful degradation.

### Validation Checks

```
=== Experiment 9: Inpainting Results ===
Training: initial loss = X, final loss = Y

Guidance weight sweep:
  w=0.0: TV=X.XX ± X.XX, community_acc=XX%
  w=0.5: TV=X.XX ± X.XX, community_acc=XX%
  w=1.0: TV=X.XX ± X.XX, community_acc=XX%
  w=2.0: TV=X.XX ± X.XX, community_acc=XX%
  w=3.0: TV=X.XX ± X.XX, community_acc=XX%

Best guidance weight: w=X.X

At best w:
  Mean TV (learned):  X.XXXX ± X.XXXX
  Mean TV (naive):    X.XXXX ± X.XXXX
  Improvement:        XX.X%
  Community preservation: XX%

Ablation (M masked nodes, best w):
  M=1: TV=X.XX, community_acc=XX%
  M=2: TV=X.XX, community_acc=XX%
  M=3: TV=X.XX, community_acc=XX%
  M=4: TV=X.XX, community_acc=XX%
  M=5: TV=X.XX, community_acc=XX%
```

### Expected Outcome

- Moderate guidance (w=0.5-1.0) should outperform both no guidance (w=0)
  and excessive guidance (w=3.0).
- At the optimal w, the model should substantially beat the naive baseline
  on both TV and community preservation.
- The guidance weight ablation is the novel contribution: it shows that
  controlling the conditioning strength matters for inpainting quality,
  just as it does in image generation with diffusion models.

### Dependencies

No new dependencies.
