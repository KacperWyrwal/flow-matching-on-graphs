# Amendment: Experiment 8.2 — Top-k Peak Metric and Configurable Epochs

## Fix 1: Threshold-Free Peak Metrics

The current `count_peaks` and `peak_location_recovery` functions use a fixed
threshold (0.10) to detect peaks. This produces false negatives when peaks
have low mass (common with 3 peaks and Dirichlet(alpha=2) mass splits),
making even the exact inverse appear to fail.

Replace with top-k metrics that use the known number of true peaks:

```python
def peak_recovery_topk(recovered, true_peaks):
    """
    Check if the top-k nodes by mass in the recovered distribution
    match the true peak nodes (exact match).
    k = len(true_peaks).
    Returns fraction of true peaks found in the top-k.
    """
    k = len(true_peaks)
    top_k = set(np.argsort(recovered)[-k:].tolist())
    return len(top_k & set(true_peaks)) / k


def peak_location_topk(recovered, true_peaks):
    """
    Check if each true peak has a top-k recovered node within Manhattan
    distance 1. Threshold-free version of peak_location_recovery.
    """
    k = len(true_peaks)
    top_k = np.argsort(recovered)[-k:].tolist()
    if not true_peaks:
        return 1.0
    covered = 0
    for tp in true_peaks:
        if any(manhattan(tp, rp) <= 1 for rp in top_k):
            covered += 1
    return covered / len(true_peaks)
```

### Changes to evaluation code

Replace all uses of `count_peaks` and `peak_location_recovery` with the
top-k versions:

```python
# Old:
'peak_count_ok_learned': int(count_peaks(mu_learned) == n_peaks),
'loc_recovery_learned': peak_location_recovery(mu_learned, true_peaks),

# New:
'peak_topk_learned': peak_recovery_topk(mu_learned, true_peaks),
'peak_topk_exact': peak_recovery_topk(mu_exact, true_peaks),
'loc_topk_learned': peak_location_topk(mu_learned, true_peaks),
'loc_topk_exact': peak_location_topk(mu_exact, true_peaks),
```

### Updated plot labels

- Panel D: "Peak recovery (top-k exact match)" instead of "Peak count accuracy"
- Panel E: "Peak location recovery (top-k, Manhattan ≤ 1)"

The exact inverse should now get 100% on both metrics at all peak counts.

## Fix 2: Configurable Number of Epochs

The loss was still decreasing at epoch 500. Add `--n-epochs` CLI argument:

```python
parser.add_argument('--n-epochs', type=int, default=1000,
                    help='Number of training epochs (default: 1000)')
```

Increase default from 500 to 1000. Pass to training:

```python
history = train_conditional(
    model, dataset, n_epochs=args.n_epochs, batch_size=64, lr=1e-3,
    device=device, loss_weighting=args.loss_weighting)
```

Usage:
```bash
# Default (1000 epochs)
python experiments/ex8b_multipeak_recovery.py

# Quick test
python experiments/ex8b_multipeak_recovery.py --n-epochs 500

# Extended training
python experiments/ex8b_multipeak_recovery.py --n-epochs 2000
```

The checkpoint filename should include the epoch count to avoid conflicts:
```python
ckpt_path = os.path.join(
    checkpoint_dir,
    f'meta_model_ex8b_multipeak_{args.loss_weighting}_{args.n_epochs}ep.pt')
```

Add the same `--n-epochs` argument to all other experiment scripts (ex3
through ex9) for consistency.

## Rerun

Delete old checkpoint and rerun with:
```bash
python experiments/ex8b_multipeak_recovery.py --n-epochs 1000
```
