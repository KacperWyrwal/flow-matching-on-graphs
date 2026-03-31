# Experiment 8: `ex8_conditional_source_recovery.py`

## Motivation

Validate conditional flow matching on graphs. The model is conditioned on an
observed diffused distribution and a diffusion time, and learns to produce
the backward flow that recovers the source. An exact solution exists (matrix
exponential inverse), providing a precise baseline.

This experiment validates the conditional architecture before applying it to
problems without exact solutions (Experiment 9).

---

## Conditional GNN Architecture

### New model: `meta_fm/model.py` — `ConditionalGNNRateMatrixPredictor`

Extends GNNRateMatrixPredictor by accepting additional per-node conditioning.

```python
class ConditionalGNNRateMatrixPredictor(nn.Module):
    """
    Predicts rate matrices conditioned on additional context.
    
    Input per node: [mu(a), t, context_1(a), context_2(a), ...]
    
    The context is a per-node signal on the same graph — it naturally
    fits as additional node features that the GNN processes alongside
    the current distribution.
    
    Constructor args:
        edge_index: torch.LongTensor (2, num_edges)
        n_nodes: int
        context_dim: int — number of additional context features per node
        hidden_dim: int = 64
        n_layers: int = 4
    
    forward(mu, t, context):
        mu: (batch, N) — current distribution
        t: (batch, 1) — flow time
        context: (batch, N, context_dim) — per-node conditioning
        returns: (batch, N, N) — valid rate matrices
    
    Architecture is identical to GNNRateMatrixPredictor except:
        - Input feature dim is 2 + context_dim instead of 2
        - First message passing layer: in_dim = 2 + context_dim
        - Everything else unchanged
    """
```

The change from GNNRateMatrixPredictor is minimal: concatenate context features
to the initial node features before message passing. No changes to message
passing layers, edge readout, or rate matrix assembly.

---

## Setup

- **Graph**: 5x5 grid graph (25 nodes), unweighted. Same as Experiment 6.
- **Forward process**: Heat diffusion. mu_obs = mu_source @ expm(tau_diff * R).
- **Diffusion times for training**: tau_diff sampled uniformly from [0.5, 4.0].
  Continuous range, not discrete levels.
- **Source distributions**: Peaked at random nodes (same generation as Ex6).
- **Training data**: 200 triples (mu_source, mu_obs, tau_diff).
  For each: pick random peak node, generate peaked distribution, pick random
  tau_diff in [0.5, 4.0], diffuse to get observation.

## Context Features

Per node: [mu_obs(a), tau_diff]

context_dim = 2. The observation value at each node, plus the diffusion time
broadcast to all nodes.

Total input per node: [mu(a), t, mu_obs(a), tau_diff] = 4 features.

## Dataset

Since MetaFlowMatchingDataset doesn't support context features, add a new
dataset class:

```python
class ConditionalMetaFlowMatchingDataset(torch.utils.data.Dataset):
    """
    Like MetaFlowMatchingDataset but stores per-sample context.
    
    Constructor args:
        graph: GraphStructure
        source_obs_pairs: list of dicts, each with keys:
            'mu_source': np.ndarray (N,)
            'mu_obs': np.ndarray (N,)
            'tau_diff': float
        n_samples: int
        seed: int = 42
    
    For each pair, computes graph-level OT coupling mu_obs -> mu_source,
    then samples (mu_tau, tau, R_target) along the flow.
    Context per sample = [mu_obs, tau_diff broadcast] per node.
    
    Uses diagonal meta-coupling: each (mu_obs_k, mu_source_k) is matched
    to itself (known ground-truth pairing).
    
    __getitem__ returns:
        mu: (N,) — distribution at time tau
        tau: (1,) — flow time
        context: (N, context_dim) — per-node conditioning
        R_target: (N, N) — target rate matrix
    """
```

## Training

- ConditionalGNNRateMatrixPredictor with context_dim=2
- hidden_dim=64, n_layers=4
- 500 epochs, Adam lr=1e-3, batch_size=64
- 5000 training samples

### Training loop

Add to `meta_fm/train.py`:

```python
def train_conditional(model, dataset, n_epochs=500, batch_size=64, lr=1e-3,
                      device=None):
    """
    Training loop for ConditionalGNNRateMatrixPredictor.
    Same as train() but dataset returns (mu, tau, context, R_target)
    and model.forward takes (mu, tau, context).
    """
```

### Conditional sampling

Add to `meta_fm/sample.py`:

```python
def sample_trajectory_conditional(model, mu_start, context, n_steps=200,
                                   device=None):
    """
    Integrate the conditional flow forward.
    Context is fixed throughout the trajectory (the observation
    doesn't change as the flow evolves).
    
    At each step:
        R = model(mu_current, t_current, context)
        mu_next = mu_current + dt * mu_current @ R
        clip, renormalize
    """
```

## Evaluation

### Test cases

20 new (source, observation, tau_diff) triples:
- 5 at tau_diff = 1.0
- 5 at tau_diff = 2.0
- 5 at tau_diff = 3.0
- 5 at tau_diff = 3.5 (interpolation — not a trained value)

For each test case:
1. Start from mu_obs at t=0.
2. Integrate forward: dp/dt = p @ R_theta(p, t, context=[mu_obs, tau_diff]).
3. Output at t=1 is the recovered source.
4. Compare to true source AND to exact inverse.

### Exact baseline

mu_source_exact = mu_obs @ expm(-tau_diff * R).
Clip negatives and renormalize if needed. Note: for large tau_diff, the exact
inverse may produce negative values, meaning the problem is ill-conditioned.
The learned model avoids this since softplus ensures non-negative rates.

### Plots (single figure, 2x3 grid)

- **Panel A**: Training loss curve.

- **Panel B**: Example recovery for one test case. Four 5x5 grid heatmaps:
  "Observation", "Recovered (learned)", "Recovered (exact)", "True source".

- **Panel C**: TV distance (recovered vs true) for learned model vs exact
  baseline, grouped by diffusion time. Grouped bar chart with two bars per
  tau_diff value (learned, exact). At low tau_diff both should be good. At
  high tau_diff the exact inverse may degrade (negative entries) while the
  learned model stays valid.

- **Panel D**: Peak recovery accuracy vs diffusion time. Two lines or grouped
  bars: learned and exact.

- **Panel E**: Interpolation test. TV distances for tau_diff = 3.5 (unseen)
  compared to nearby trained values (3.0, 4.0). Shows the model interpolates
  smoothly.

- **Panel F**: One trajectory visualization. 5x5 grid heatmaps at
  t = 0, 0.25, 0.5, 0.75, 1.0 showing the conditioned flow concentrating
  mass from diffused observation toward source peak.

### Validation checks (print to console)

- Per diffusion time: mean TV (learned), mean TV (exact), peak accuracy (both)
- Interpolation: TV at tau_diff=3.5 vs average of 3.0 and 4.0
- Any cases where exact inverse produces negative entries

## Expected Outcome

At low diffusion times (tau_diff=1.0), both learned and exact should perform
well. At high diffusion times (tau_diff=3.0-4.0), the exact inverse may
degrade while the learned model stays robust due to its constrained output
space. The interpolation to tau_diff=3.5 should work smoothly.

## Dependencies

No new dependencies.
