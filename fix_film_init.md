# Fix: FiLM Identity Initialization

## Problem

The FiLM conditioning layer starts with random gamma and beta values,
which disrupts the hidden features from the first forward pass. The model
effectively starts from a random initialization rather than building on
the working non-FiLM architecture. This explains the training slowdown
(loss stuck at 0.016 vs 0.0005 for the non-FiLM version).

## Fix

Initialize the FiLM MLP's last layer so that gamma=1, beta=0 at the start.
This makes the FiLM layer act as an identity at initialization:

    h <- 1 * h + 0 = h (no modulation)

The model starts by ignoring the global conditioning and behaves like
the non-FiLM version, then gradually learns to use the conditioning
as training progresses.

## Code Change

In `FiLMRateMessagePassing.__init__`, after creating `self.film_mlp`:

```python
class FiLMRateMessagePassing(MessagePassing):
    def __init__(self, in_dim: int, hidden_dim: int, global_dim: int):
        super().__init__(aggr='add')
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.film_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        
        # === ADD THIS: Identity FiLM initialization ===
        # Last layer of film_mlp outputs [gamma, beta], each of dim hidden_dim.
        # Initialize to output gamma=1, beta=0 so FiLM starts as identity.
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)
        self.film_mlp[-1].bias.data[:hidden_dim] = 1.0  # gamma = 1
        # beta half of bias already 0 from zeros_ init
```

## Why This Works

With this initialization:
- film_mlp[-1].weight = 0: output doesn't depend on input (initially)
- film_mlp[-1].bias[:hidden_dim] = 1.0: gamma = 1
- film_mlp[-1].bias[hidden_dim:] = 0.0: beta = 0
- So h <- 1 * h + 0 = h for all inputs

As training progresses, the weights become nonzero and the model learns
to modulate features based on the global conditioning. The gradient
signal flows through the film_mlp and the encoder, allowing both to
learn jointly with the rest of the network.

## Expected Impact

The model should now start training at a similar loss as the non-FiLM
version (~0.017 initial, dropping to ~0.001 within a few hundred epochs)
and then potentially improve further as the FiLM conditioning kicks in
and provides additional information from the sensor readings.

## Apply To

`meta_fm/model.py`, class `FiLMRateMessagePassing.__init__`

## Rerun

Delete the existing FiLM checkpoint and rerun:
```bash
rm checkpoints/meta_model_ex13_film_*.pt
python experiments/ex13_sparse_sensors.py --conditioning film
```
