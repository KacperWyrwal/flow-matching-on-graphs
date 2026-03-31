# Fix: MetaFlowMatchingDataset Precomputed Coupling + Experiment 6 Correction

## Problem

MetaFlowMatchingDataset always computes its own meta-OT coupling with uniform
marginals. For Experiment 6 (source localization), the meta-OT coupling matches
diffused distributions to peaked distributions based on transport cost, NOT based
on the known ground-truth pairing (diffused_k came from peaked_k). This means
the model is trained on incorrect transport paths.

## Fix: Add `meta_coupling` parameter to MetaFlowMatchingDataset

In `meta_fm/dataset.py`, add an optional `meta_coupling` argument. When provided,
skip the meta-cost and meta-OT computation and use the given coupling directly.

```python
class MetaFlowMatchingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        graph: GraphStructure,
        source_distributions,
        target_distributions,
        n_samples: int,
        cost_type: str = 'wasserstein',
        cache: GeodesicCache | None = None,
        use_sinkhorn: bool = True,
        meta_coupling: np.ndarray | None = None,   # <-- NEW PARAMETER
        seed: int = 42,
    ):
        # ... existing setup ...
        
        # Step 2 & 3: meta-cost and meta-OT coupling
        if meta_coupling is not None:
            # Use the provided coupling directly
            Pi_meta = meta_coupling
        else:
            # Existing behavior: compute meta-cost and solve meta-OT
            W_meta = compute_meta_cost_matrix_batch(
                source_distributions, target_distributions, cost,
                use_sinkhorn=use_sinkhorn,
            )
            uniform_s = np.ones(n_sources) / n_sources
            uniform_t = np.ones(n_targets) / n_targets
            Pi_meta = ot.emd(uniform_s, uniform_t, W_meta)
        
        # ... rest unchanged ...
```

## Fix: Experiment 6 uses diagonal coupling

In `ex6_grid_source_localization.py`, construct a diagonal meta-coupling and
pass it to the dataset. Since each diffused distribution k should pair with
peaked distribution k:

```python
# Diagonal coupling: diffused_k <-> peaked_k
n_train = len(diffused_dists)
meta_coupling = np.eye(n_train) / n_train  # uniform weight on each pair

dataset = MetaFlowMatchingDataset(
    graph=graph,
    source_distributions=diffused_dists,
    target_distributions=peaked_dists,
    n_samples=5000,
    meta_coupling=meta_coupling,    # <-- use known pairing
    seed=42,
)
```

This ensures each training sample comes from the correct diffusion-reversal path:
diffused_k is transported back to peaked_k, not to some other peaked distribution
that happened to be cheaper in OT distance.

## Impact on Other Experiments

- **Experiments 3, 4, 5**: No change. These use unrelated source/target distributions
  where the meta-OT coupling is the correct approach.
- **Experiment 7**: No change. Uniform noise has no ground-truth pairing to the
  community distributions, so meta-OT is correct.
- **Experiment 6**: Must use diagonal coupling. Delete the checkpoint and rerun
  after applying this fix.

## Rerun Experiment 6

After applying the fix:
1. Delete `checkpoints/meta_model_grid_backward_gnn.pt`
2. Rerun `ex6_grid_source_localization.py`
3. The training loss should now decrease meaningfully (the model is learning
   a consistent mapping rather than a confused mixture of different pairings)
