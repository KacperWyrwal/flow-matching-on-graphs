# Amendment: Experiment 13 — GNN + Softmax Baseline

## Motivation

The flow matching formulation is more complex and expensive than a direct
GNN predictor. We should verify it provides value over the simpler approach.

## Baseline 4: GNN + Softmax (Direct Prediction)

Same GNN architecture as the flow matching model, but instead of predicting
rate matrices and integrating over time, directly predict the source
distribution in one forward pass.

### Architecture

```python
class DirectGNNPredictor(nn.Module):
    """
    Same message-passing backbone as FlexibleConditionalGNNRateMatrixPredictor,
    but the readout is per-node logits → softmax → distribution.
    
    Input per node: [mu_backproj(a), tau_diff] (same context as flow model)
    Message passing: same layers, same hidden_dim, same n_layers
    Readout: MLP per node → scalar logit → softmax over all nodes
    
    forward(context, edge_index):
        context: (N, 2)
        edge_index: (2, E)
        returns: (N,) distribution (sums to 1)
    """
    def __init__(self, context_dim=2, hidden_dim=128, n_layers=6):
        super().__init__()
        # Same message passing layers as FlexibleConditionalGNNRateMatrixPredictor
        self.mp_layers = nn.ModuleList()
        self.mp_layers.append(RateMessagePassing(in_dim=context_dim, hidden_dim=hidden_dim))
        for _ in range(n_layers - 1):
            self.mp_layers.append(RateMessagePassing(in_dim=hidden_dim, hidden_dim=hidden_dim))
        
        # Per-node readout → scalar logit
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, context, edge_index):
        # context: (N, context_dim)
        h = context
        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index)
        logits = self.readout(h).squeeze(-1)  # (N,)
        return torch.softmax(logits, dim=0)    # (N,) distribution
```

### Training

```python
def train_direct_gnn(model, train_pairs, n_epochs=1000, lr=5e-4, device='cpu'):
    """
    Train with KL divergence loss: KL(true || predicted).
    
    train_pairs: list of (context, mu_source, edge_index) tuples
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        rng.shuffle(train_pairs)
        epoch_loss = 0
        for context, mu_true, edge_index in train_pairs:
            mu_pred = model(context.to(device), edge_index.to(device))
            # KL divergence: sum p_true * log(p_true / p_pred)
            mu_true_t = torch.tensor(mu_true, dtype=torch.float32, device=device)
            loss = (mu_true_t * (mu_true_t.clamp(min=1e-10).log() 
                    - mu_pred.clamp(min=1e-10).log())).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
```

### Key differences from flow model

| Aspect | Flow Matching | GNN + Softmax |
|--------|--------------|---------------|
| Training data | OT rate matrices (expensive) | Source distributions (cheap) |
| Training objective | MSE on rate matrices | KL divergence on distributions |
| Inference | 200-step integration | Single forward pass |
| Output guarantee | Valid distribution (rate matrix) | Valid distribution (softmax) |
| Stochastic sampling | Possible (vary initial dist) | Deterministic |
| Interpretable trajectory | Yes | No |

### Evaluation

Run the GNN + softmax baseline on the same 90 test cases. Report all the
same metrics (Full TV, Interior TV, peak recovery by depth).

### Updated plots

Add GNN+softmax as a 5th method (orange) to all comparison plots:
- Panel C: 5 methods × 3 peak counts
- Panel D: 5 methods for interior TV
- Panel E: 5 methods × 3 depth categories

### Updated console output

```
Full TV:
  Learned (flow):  X.XXXX ± X.XXXX
  GNN+softmax:     X.XXXX ± X.XXXX
  LASSO:           X.XXXX ± X.XXXX
  MNE:             X.XXXX ± X.XXXX
  Backprojection:  X.XXXX ± X.XXXX
```

## CLI

```python
parser.add_argument('--train-direct-gnn', action='store_true',
                    help='Also train and evaluate GNN+softmax baseline')
```

## Dependencies

No new dependencies.
