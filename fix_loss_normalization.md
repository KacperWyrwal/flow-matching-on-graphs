# Fix: Normalize Loss by Number of Off-Diagonal Entries

## Problem

The MSE loss sums over all off-diagonal entries of the N×N rate matrix.
For a 125-node graph this is ~15,600 entries vs ~625 for a 25-node graph.
The raw loss is ~25x larger, making loss values incomparable across graph
sizes and effectively scaling up the learning rate for larger graphs.

## Fix

Divide the per-sample loss by N*(N-1) (the number of off-diagonal entries):

In `train_flexible_conditional`:

```python
per_sample = (diff_sq * off_diag_mask.float()).sum(dim=(-2, -1))
per_sample = per_sample / (N * (N - 1))  # <-- add this line
```

In `train` and `train_conditional` (fixed-graph versions):

```python
per_sample = (diff_sq * mask.unsqueeze(0)).sum(dim=(-2, -1))
N = R_pred.shape[-1]
per_sample = per_sample / (N * (N - 1))  # <-- add this line
```

## Impact

- Loss values become comparable across graph sizes
- Learning rate has consistent effective scale regardless of N
- No change to optimization dynamics for fixed-graph experiments
  (just a constant rescaling)
- For topology generalization (mixed graph sizes in one batch),
  this prevents large graphs from dominating the gradient

## Apply to all training functions

- `train()` in train.py
- `train_conditional()` in train.py
- `train_flexible_conditional()` in train.py

## Rerun

No need to rerun previous experiments (results are unchanged, just loss
scale differs). Apply before rerunning Ex12 v2.
