# Updated Fix: Factorize 1/(1-t) With Configurable Loss Weighting

## Summary

The model learns the bounded residual u_tilde = (1-t) * u. The 1/(1-t) factor
is applied analytically at inference. The loss weighting on the factorized
targets is configurable.

## Loss Weighting Options

```python
def train(model, dataset, n_epochs=500, batch_size=64, lr=1e-3, device=None,
          loss_weighting='original'):
    """
    loss_weighting options:
        'original'  — weight by 1/(1-t)^2. Equivalent to unweighted MSE on
                       the raw (unfactorized) rate matrices. Preserves the
                       original objective while benefiting from bounded model
                       outputs. This is the default.
        'uniform'   — no weighting. Equal importance across all times.
                       Downweights late times relative to the original objective.
        'linear'    — weight by 1/(1-t). A middle ground.
    """
```

Implementation in the training loop:

```python
# After computing per-sample MSE on off-diagonal entries:
per_sample_loss = diff_sq.sum(dim=(-1, -2))  # (batch,)
tau = tau.squeeze(-1)  # (batch,)

if loss_weighting == 'original':
    weights = 1.0 / (1.0 - tau).clamp(min=0.001) ** 2
elif loss_weighting == 'linear':
    weights = 1.0 / (1.0 - tau).clamp(min=0.001)
elif loss_weighting == 'uniform':
    weights = torch.ones_like(tau)
else:
    raise ValueError(f"Unknown loss_weighting: {loss_weighting}")

loss = (weights * per_sample_loss).mean()
```

The .clamp(min=0.001) prevents division by zero when tau is very close to 1.

Same parameter added to train_conditional.

## Experiment Scripts

Each experiment accepts a `--loss-weighting` command line argument:

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--loss-weighting', type=str, default='uniform',
                    choices=['original', 'uniform', 'linear'],
                    help='Loss weighting scheme (default: uniform)')
args = parser.parse_args()

# Then pass to training:
history = train(model, dataset, n_epochs=500, loss_weighting=args.loss_weighting)
# or:
history = train_conditional(model, dataset, n_epochs=500,
                            loss_weighting=args.loss_weighting)
```

Usage:
```bash
# Default (uniform — no weighting)
python experiments/ex8_conditional_source_recovery.py

# Original objective (1/(1-t)^2 weighting)
python experiments/ex8_conditional_source_recovery.py --loss-weighting original

# Middle ground
python experiments/ex8_conditional_source_recovery.py --loss-weighting linear
```

Add this to all experiment scripts that train a model: ex3, ex4, ex5, ex6, ex7,
ex8, ex9. The default is 'uniform' to match the current working experiments.

The checkpoint filename should include the weighting scheme to avoid conflicts:
```python
ckpt_path = os.path.join(checkpoint_dir,
    f'meta_model_ex8_cond_gnn_{args.loss_weighting}.pt')
```

## Recommendation

Use 'uniform' as the default — it's what produced the good results on
Experiments 3 and 4 after factorization. Try 'original' on backward/conditional
experiments (8, 9) if 'uniform' underperforms at late times. The checkpoint
naming convention makes it easy to compare runs side by side.
