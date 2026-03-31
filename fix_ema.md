# Fix: Add EMA to Training Loops

## Implementation

Add to `meta_fm/train.py`:

```python
class EMA:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        """Call after each optimizer.step()."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay)
    
    def apply(self, model):
        """Swap model weights with EMA weights (for evaluation)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model):
        """Restore original weights (after evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
```

## Integration into Training Loops

In `train_flexible_conditional`, `train_film_conditional`, and
`train_direct_gnn`:

```python
def train_flexible_conditional(model, dataset, n_epochs=1000, ...,
                                ema_decay=0.999):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ema = EMA(model, decay=ema_decay)
    
    for epoch in range(n_epochs):
        # ... existing training loop ...
        optimizer.step()
        ema.update(model)  # <-- add this line after every step
    
    # Apply EMA weights before returning
    ema.apply(model)
    
    return {'losses': losses, 'ema': ema}
```

## CLI

```python
parser.add_argument('--ema-decay', type=float, default=0.999,
                    help='EMA decay rate (0 to disable)')
```

## Checkpoint

Save the EMA state alongside the model:

```python
torch.save({
    'model_state_dict': model.state_dict(),  # these are EMA weights after ema.apply()
    'ema_shadow': ema.shadow,
}, ckpt_path)
```

## Apply To

All training functions in `meta_fm/train.py`:
- `train()`
- `train_conditional()`
- `train_flexible_conditional()`
- `train_film_conditional()`
- `train_direct_gnn()`

## Expected Impact

Smoother evaluation metrics, especially for experiments with training
spikes (Ex12b, Ex13). The EMA model should give slightly better test
performance than the raw model at any given epoch.
