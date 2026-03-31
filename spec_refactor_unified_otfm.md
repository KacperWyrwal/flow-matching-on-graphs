# Spec: Refactor into Unified `otfm` Package

## Overview

Create a new branch `refactor/unified-otfm` and reorganize the entire
codebase into a clean, modular package reflecting the three-level
framework:

- **Level 1 (configuration):** Flow matching on arbitrarily large
  (possibly infinite) configuration graphs via local Markov moves
- **Level 2 (graph-marginal):** Flow matching on small explicit graphs
  with exact OT couplings and marginal rate fields
- **Level 3 (distribution-conditional):** Flow matching in distribution
  space, using level 2 as a subroutine, for posterior sampling with
  uncertainty quantification

## Branch setup

```bash
git checkout -b refactor/unified-otfm
```

## Target structure

```
otfm/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── loss.py                    # Rate KL loss (shared across all levels)
│   ├── schedulers.py              # 1/(1-t) factorization, time sampling utilities
│   └── ot.py                      # OT solvers (W1 + tiebreaker, partial OT)
│
├── graph/                         # Level 2: graph-level marginal
│   ├── __init__.py
│   ├── structure.py               # GraphStructure, GeodesicCache
│   ├── flow.py                    # Conditional/marginal rates, binomial marginals
│   ├── coupling.py                # OT coupling on explicit graphs
│   └── sample.py                  # Trajectory simulation on explicit graphs
│
├── distribution/                  # Level 3: distribution-level conditional
│   ├── __init__.py
│   ├── dataset.py                 # Dirichlet starts (obs-centered), OT coupling
│   │                              # between distributions, dataset construction
│   ├── sample.py                  # Posterior sampling (K starts → posterior mean)
│   └── calibration.py             # Calibration r, diversity, posterior metrics
│
├── configuration/                 # Level 1: graph-level conditional
│   ├── __init__.py
│   ├── spaces/
│   │   ├── __init__.py
│   │   ├── base.py                # ConfigurationSpace ABC
│   │   ├── johnson.py             # Fixed Hamming weight (J(n,k))
│   │   ├── kawasaki.py            # Fixed magnetization on lattice
│   │   └── dfm.py                 # Unconstrained (DFM baseline)
│   ├── flow.py                    # Geodesics, intermediates, conditional rates
│   ├── sample.py                  # CTMC simulation via Markov moves
│   └── mcmc.py                    # MCMC utilities for target pool generation
│
├── models/
│   ├── __init__.py
│   ├── backbone.py                # AttentionBackbone (unified GNN/transformer)
│   ├── heads.py                   # Output heads:
│   │                              #   EdgeRateHead (level 2, per-edge on explicit graph)
│   │                              #   SingleNodeHead (level 1, k=1, DFM-style)
│   │                              #   PairwiseAttentionHead (level 1, k=2)
│   │                              #   PairOfPairsHead (level 1, k=4)
│   ├── conditioning.py            # FiLM layers, context encoding
│   └── predictor.py               # Top-level model classes:
│                                  #   GraphRatePredictor (level 2)
│                                  #   DistributionRatePredictor (level 3)
│                                  #   ConfigurationRatePredictor (level 1)
│
├── train/
│   ├── __init__.py
│   ├── graph_marginal.py          # Level 2 training loop
│   ├── distribution.py            # Level 3 training loop
│   └── configuration.py           # Level 1 training loop
│
└── eval/
    ├── __init__.py
    ├── metrics.py                 # TV, energy stats, correlation RMSE, KS test
    ├── validity.py                # Constraint satisfaction checks
    └── visualization.py           # Graph drawing, distribution plots, galleries

experiments/
├── ex11_combined.py
├── ex17_ot_transport.py
├── ex18_source_recovery.py
├── ex20_johnson.py
├── ex21_kawasaki.py
└── (future experiments)
```

## Migration mapping

### From `graph_ot_fm/` → `otfm/graph/` and `otfm/core/`

| Old | New | Notes |
|-----|-----|-------|
| `graph_ot_fm/graph_structure.py` | `otfm/graph/structure.py` | GraphStructure, GeodesicCache |
| `graph_ot_fm/ot_solver.py` | `otfm/core/ot.py` | OT solver (W1 + tiebreaker) |
| `graph_ot_fm/flow.py` | `otfm/graph/flow.py` | marginal_distribution_fast, marginal_rate_matrix_fast |
| `graph_ot_fm/utils.py` | `otfm/eval/metrics.py` | total_variation, etc. |
| `graph_ot_fm/graphs.py` | `otfm/graph/structure.py` | Graph construction helpers |

### From `meta_fm/` → `otfm/models/`, `otfm/train/`, `otfm/distribution/`

| Old | New | Notes |
|-----|-----|-------|
| `meta_fm/model.py` (FlexibleConditionalGNN) | `otfm/models/predictor.py` | GraphRatePredictor |
| `meta_fm/model.py` (FiLMConditionalGNN) | `otfm/models/predictor.py` | DistributionRatePredictor |
| `meta_fm/model.py` (DirectGNNPredictor) | `otfm/models/predictor.py` | DirectPredictor (baseline) |
| `meta_fm/model.py` (FiLM layers) | `otfm/models/conditioning.py` | FiLM, context encoding |
| `meta_fm/model.py` (rate_matrix_to_edge_index) | `otfm/graph/structure.py` | Utility |
| `meta_fm/train.py` | `otfm/train/graph_marginal.py` | train_flexible_conditional |
| `meta_fm/train.py` | `otfm/train/distribution.py` | train_film_conditional |
| `meta_fm/sample.py` | `otfm/graph/sample.py` | sample_trajectory_flexible |
| `meta_fm/sample.py` | `otfm/distribution/sample.py` | sample_posterior_film |
| `meta_fm/dataset.py` | `otfm/distribution/dataset.py` | TopologyGeneralizationDataset, DiffusionSourceDataset |
| `meta_fm/ema.py` | `otfm/train/` (shared utility) | EMA |

### From `johnson_fm/` and `config_fm/` → `otfm/configuration/`

| Old | New | Notes |
|-----|-----|-------|
| `config_fm/config_space.py` | `otfm/configuration/spaces/base.py` | ConfigurationSpace ABC |
| `config_fm/spaces/johnson.py` | `otfm/configuration/spaces/johnson.py` | JohnsonSpace |
| `config_fm/spaces/kawasaki.py` | `otfm/configuration/spaces/kawasaki.py` | KawasakiSpace |
| `config_fm/spaces/dfm.py` | `otfm/configuration/spaces/dfm.py` | DFMSpace (clean, no monkey-patching) |
| `config_fm/model.py` | `otfm/models/predictor.py` | ConfigurationRatePredictor |
| `config_fm/train.py` | `otfm/train/configuration.py` | train_configuration_fm |
| `config_fm/sample.py` | `otfm/configuration/sample.py` | generate_samples |
| `config_fm/loss.py` | `otfm/core/loss.py` | Merged with graph-level loss |

### Delete after migration

- `graph_ot_fm/` (replaced by `otfm/graph/` + `otfm/core/`)
- `meta_fm/` (replaced by `otfm/models/` + `otfm/train/` + `otfm/distribution/`)
- `johnson_fm/` (replaced by `otfm/configuration/`)
- `config_fm/` (replaced by `otfm/configuration/` + `otfm/models/`)

## Key design decisions

### 1. ConfigurationSpace.sample_target always returns (config, context_dict)

```python
class ConfigurationSpace(ABC):
    @abstractmethod
    def sample_target(self, rng, **kwargs) -> tuple[np.ndarray, dict]:
        """Returns (config, context_kwargs).
        context_kwargs is passed to global_features during training."""
        ...
```

No `MultiBetaSampler` wrapper. The space itself handles multi-beta
sampling internally if needed:

```python
class KawasakiSpace(ConfigurationSpace):
    def __init__(self, L, k, J, betas, mcmc_pools):
        self.betas = betas
        self.pools = mcmc_pools
    
    def sample_target(self, rng, **kwargs):
        beta = float(rng.choice(self.betas))
        idx = rng.integers(len(self.pools[beta]))
        return self.pools[beta][idx].copy(), {'beta': beta}
```

### 2. Unified AttentionBackbone

One implementation that handles both sparse and dense attention:

```python
class AttentionBackbone(nn.Module):
    """Attention-based message passing on any graph.
    
    - Complete graph (no edge_index) → standard transformer
    - Sparse graph (edge_index provided) → graph attention network
    - Edge features → attention biases
    - Global features → FiLM conditioning
    """
    def __init__(self, node_dim, edge_dim=0, global_dim=0,
                 hidden_dim=128, n_layers=4, n_heads=4):
        ...
    
    def forward(self, node_features, edge_index=None, edge_features=None,
                global_features=None, attention_mask=None):
        """
        If edge_index is None: full attention (transformer mode)
        If edge_index provided: sparse attention (GNN mode)
        """
        ...
```

### 3. Loss function shared across levels

The rate KL loss has the same form everywhere:

$$D(r \| r^\theta) = \sum [r \log(r/r^\theta) - r + r^\theta]$$

The only difference is what "sum over" means:
- Level 2: sum over edges of explicit graph
- Level 1: sum over valid transitions in configuration space
- Level 3: same as level 2 (operating on the explicit graph)

One implementation with a mask argument handles all cases:

```python
def rate_kl_loss(pred, target, mask):
    """Rate KL loss, summed over valid entries (mask > 0)."""
    ...
```

### 4. DFMSpace as a proper first-class space

```python
class DFMSpace(ConfigurationSpace):
    """Unconstrained binary/categorical labels.
    
    Works on any position graph. No invariant constraint.
    Transitions: single-node label change.
    """
    def __init__(self, n, vocab_size=2,
                 position_graph_edges=None,
                 position_edge_features=None,
                 betas=None, mcmc_pools=None):
        ...
```

Constructed with explicit position graph and edge features. No
monkey-patching.

### 5. Predictor classes per level

```python
# Level 2: predicts rate matrix on explicit graph edges
class GraphRatePredictor(nn.Module):
    """GNN backbone + EdgeRateHead.
    Input: distribution p_t, context, edge_index
    Output: rate u_t(a,b) for each edge (a,b)"""

# Level 3: predicts rate matrix conditioned on context
class DistributionRatePredictor(nn.Module):
    """GNN backbone + EdgeRateHead + FiLM conditioning.
    Input: distribution p_t, node_context, global_context, edge_index
    Output: rate u_t(a,b) for each edge (a,b)"""

# Level 1: predicts transition rates on configuration space
class ConfigurationRatePredictor(nn.Module):
    """AttentionBackbone + TransitionHead (k-dependent).
    Input: configuration x_t, global_context, position_graph
    Output: rates for valid transitions"""

# Baseline: direct prediction (no flow)
class DirectPredictor(nn.Module):
    """GNN backbone + per-node softmax output.
    Input: context, edge_index
    Output: predicted distribution"""
```

All share the same `AttentionBackbone` internally but differ in
input/output interfaces.

### 6. Training loops

Each level has its own training function with a clear signature:

```python
# Level 2
def train_graph_marginal(model, dataset, n_epochs, batch_size,
                         lr, device, **kwargs):
    """Train on precomputed (p_t, t, context, u_target, edge_index) tuples."""

# Level 3
def train_distribution(model, dataset, n_epochs, batch_size,
                       lr, device, **kwargs):
    """Train on (mu_start, mu_target, context) with Dirichlet starts.
    Uses level 2 internally to compute flow matching targets."""

# Level 1
def train_configuration(model, space, n_epochs, batch_size,
                        lr, device, **kwargs):
    """Train on-the-fly: sample (x_0, x_T) pairs, compute geodesics,
    sample intermediates, compute conditional rates."""
```

## Implementation order

### Phase 1: Core + Level 2 (the foundation)

1. Create `otfm/core/` with loss, schedulers, OT solver
2. Create `otfm/graph/` by migrating from `graph_ot_fm/`
3. Create `otfm/models/backbone.py` and `otfm/models/conditioning.py`
4. Create `otfm/models/predictor.py` with `GraphRatePredictor`
5. Create `otfm/train/graph_marginal.py`
6. Create `otfm/eval/`
7. **Verify:** Ex11 obs-start and Ex17 produce identical results

### Phase 2: Level 3 (distribution-level)

8. Create `otfm/distribution/` by migrating from `meta_fm/`
9. Create `DistributionRatePredictor` in predictor.py
10. Create `otfm/train/distribution.py`
11. **Verify:** Ex11d/e (Dirichlet starts) and Ex18 reproduce results

### Phase 3: Level 1 (configuration-level)

12. Create `otfm/configuration/` with clean spaces
13. Create `ConfigurationRatePredictor` in predictor.py
14. Create `otfm/train/configuration.py`
15. Implement proper `DFMSpace` (no monkey-patching)
16. **Verify:** Ex20 and Ex21 reproduce results

### Phase 4: Cleanup

17. Update all experiment scripts to import from `otfm/`
18. Delete old packages (`graph_ot_fm/`, `meta_fm/`, `johnson_fm/`, `config_fm/`)
19. Run all experiments end-to-end to confirm nothing is broken

## Verification protocol

After each phase, run the relevant experiments and confirm:

**Phase 1 verification (level 2):**
```
Ex11 obs-start: TV=0.054, Peak=98% (FM), TV=0.055, Peak=98% (DirectGNN)
Ex17: FM path TV ~0.14 ID (check against stored results)
```

**Phase 2 verification (level 3):**
```
Ex11e obs-centered α=10: TV=0.037, Peak=98%, Cal-r=0.65
Ex18: FM TV comparable to stored results
```

**Phase 3 verification (level 1):**
```
Ex20 n=16 β=2.0: TV=0.14, E bias=0.14, Corr RMSE=0.016
Ex21: matches pre-refactor results (once available)
```

If any result deviates by more than 10% relative from the reference,
stop and debug before proceeding.

## Do NOT change

- Experiment logic, evaluation metrics, or plotting code (these stay
  in experiment scripts)
- Hyperparameters or model architectures (this is a refactor, not a
  redesign)
- Checkpoint format (old checkpoints should load into new models where
  architectures match)
- Random seeds (reproducibility must be preserved)
