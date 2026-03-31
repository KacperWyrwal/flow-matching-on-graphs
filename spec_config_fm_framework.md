# Spec: Unified Configuration Flow Matching Framework

## Overview

Refactor the Ex20 Johnson graph experiment to use a general configuration
flow matching framework. The framework separates problem-independent
components (GNN backbone, training loop, inference, loss) from
problem-specific components (configuration space definition, transition
enumeration, geodesic computation).

The Johnson graph experiment should be a thin instantiation of the general
framework, demonstrating that new problems (fixed composition, Kawasaki,
degree-preserving, homology) can be added by defining a new
`ConfigurationSpace` subclass with no changes to training or inference.

## Architecture

### Directory structure

```
config_fm/
├── __init__.py
├── config_space.py       # Abstract ConfigurationSpace base class
├── model.py              # GNN backbone + transition scoring head
├── train.py              # Training loop (problem-independent)
├── sample.py             # Inference / generation (problem-independent)
├── loss.py               # Rate KL loss (problem-independent)
├── spaces/
│   ├── __init__.py
│   ├── johnson.py        # Johnson graph J(n,k)
│   ├── dfm.py            # DFM on K_M^L (for baseline comparison)
│   └── (future: kawasaki.py, degree_sequence.py, homology.py, etc.)
experiments/
└── ex20_johnson_graph.py # Thin script using config_fm + johnson space
```

### Core abstraction: `ConfigurationSpace`

```python
# config_fm/config_space.py

from abc import ABC, abstractmethod
import numpy as np

class ConfigurationSpace(ABC):
    """Abstract base class for configuration flow matching.
    
    A configuration is a labeled graph (G, ell) where G is a fixed
    position graph and ell: V -> vocab is a node labeling.
    
    Subclasses define:
    - The position graph and vocabulary
    - The invariant (constraint)
    - Valid transitions and their enumeration
    - Geodesic computation between configurations
    - Sampling source and target configurations
    """
    
    @property
    @abstractmethod
    def n_positions(self) -> int:
        """Number of nodes in the position graph."""
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Size of the label vocabulary."""
        pass
    
    @property
    @abstractmethod
    def transition_order(self) -> int:
        """Number of nodes affected per transition (k).
        k=1 for DFM, k=2 for Johnson/Kawasaki, k=4 for degree-preserving."""
        pass
    
    @abstractmethod
    def position_graph_edges(self) -> np.ndarray:
        """Return edge index of the position graph for GNN message passing.
        Shape: (2, n_edges). This is the graph the GNN operates on."""
        pass
    
    @abstractmethod
    def position_edge_features(self) -> np.ndarray | None:
        """Optional edge features for the position graph.
        Shape: (n_edges, d_edge) or None.
        E.g., J[i,j] coupling strengths for the Ising model."""
        pass
    
    @abstractmethod
    def node_features(self, config: np.ndarray) -> np.ndarray:
        """Compute node features for the GNN given a configuration.
        
        Args:
            config: (n_positions,) or (n_positions, vocab_size) label array
        Returns:
            (n_positions, d_node) feature array
        """
        pass
    
    @abstractmethod
    def global_features(self, **kwargs) -> np.ndarray:
        """Compute global context features (e.g., [t, beta]).
        Returns: (d_global,) feature array."""
        pass
    
    @abstractmethod
    def enumerate_transitions(self, config: np.ndarray) -> dict:
        """Enumerate all valid transitions from the current configuration.
        
        Args:
            config: current configuration (n_positions,)
        Returns:
            dict with:
              'S_minus': list of arrays, each listing nodes flipping
                         from their current label (role: "source" of flip)
              'S_plus':  list of arrays, each listing nodes receiving
                         new labels (role: "target" of flip)
              'labels_old': list of arrays, old labels for affected nodes
              'labels_new': list of arrays, new labels for affected nodes
              'n_transitions': int, total number of valid transitions
              
        For binary vocab, S_minus[i] has nodes going 1->0 and S_plus[i]
        has nodes going 0->1 for transition i.
        """
        pass
    
    @abstractmethod
    def transition_mask(self, config: np.ndarray) -> np.ndarray:
        """Return a mask over the model's output indicating valid transitions.
        
        The shape depends on transition_order:
          k=1: (n_positions, vocab_size) -- per-node, per-target-label
          k=2: (n_positions, n_positions) -- per-pair
          k=4: (n_edges, n_edges) -- per-edge-pair (on position graph)
        
        Args:
            config: current configuration (n_positions,)
        Returns:
            mask array with 1 for valid transitions, 0 otherwise
        """
        pass
    
    @abstractmethod
    def apply_transition(self, config: np.ndarray,
                         transition_idx) -> np.ndarray:
        """Apply a transition to a configuration.
        
        Args:
            config: current configuration (n_positions,)
            transition_idx: index or identifier of the transition to apply
        Returns:
            new configuration (n_positions,)
        """
        pass
    
    @abstractmethod
    def geodesic_distance(self, config_a: np.ndarray,
                          config_b: np.ndarray) -> int:
        """Compute geodesic distance on the configuration graph."""
        pass
    
    @abstractmethod
    def sample_intermediate(self, config_0: np.ndarray,
                            config_T: np.ndarray,
                            t: float, rng) -> tuple:
        """Sample an intermediate configuration along the geodesic.
        
        Args:
            config_0: source configuration
            config_T: target configuration
            t: flow time in [0, 1)
            rng: numpy random generator
        Returns:
            (config_t, n_completed, n_remaining): intermediate config,
            number of completed transitions, number remaining
        """
        pass
    
    @abstractmethod
    def compute_target_rates(self, config_0: np.ndarray,
                             config_T: np.ndarray,
                             config_t: np.ndarray,
                             t: float) -> np.ndarray:
        """Compute conditional target rates at config_t.
        
        Returns array matching the shape of transition_mask output,
        with the target rate for each valid transition. The 1/(1-t)
        factor should NOT be included (model learns tilde_r, the
        bounded version).
        """
        pass
    
    @abstractmethod
    def sample_source(self, rng) -> np.ndarray:
        """Sample a configuration from the source distribution."""
        pass
    
    @abstractmethod
    def sample_target(self, rng, **kwargs) -> np.ndarray:
        """Sample a configuration from the target distribution."""
        pass
```

### GNN Backbone + Scoring Head: `ConfigurationRatePredictor`

```python
# config_fm/model.py

class ConfigurationRatePredictor(nn.Module):
    """Unified model for configuration flow matching.
    
    Architecture:
    1. GNN backbone on position graph -> node embeddings
    2. Transition scoring head -> rates for valid transitions
    
    The scoring head adapts to transition_order:
      k=1: per-node MLP (DFM style)
      k=2: pairwise attention (Johnson/Kawasaki style)
      k=4: pair-of-pairs attention (degree-preserving style)
    """
    
    def __init__(self,
                 node_feature_dim: int,
                 edge_feature_dim: int = 0,
                 global_dim: int = 2,
                 hidden_dim: int = 128,
                 n_layers: int = 4,
                 transition_order: int = 2,
                 vocab_size: int = 2):
        super().__init__()
        
        self.transition_order = transition_order
        self.vocab_size = vocab_size
        
        # GNN backbone (same for all instances)
        # Uses FiLM conditioning for global features (t, beta, etc.)
        self.backbone = FiLMGNNBackbone(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )
        
        # Scoring head (depends on transition_order and vocab_size)
        if transition_order == 1:
            # DFM: per-node rate for each target label
            # Output: (n, vocab_size) rates
            self.head = SingleNodeHead(hidden_dim, vocab_size)
        elif transition_order == 2:
            if vocab_size == 2:
                # Johnson/Kawasaki: pairwise attention between
                # 1-nodes and 0-nodes
                # Output: (n, n) swap rates
                self.head = PairwiseAttentionHead(hidden_dim)
            else:
                # Fixed composition: pairwise attention with label
                # embeddings
                # Output: (n, n) swap rates
                self.head = PairwiseAttentionWithLabelsHead(
                    hidden_dim, vocab_size)
        elif transition_order == 4:
            # Degree-preserving: pair-of-pairs attention
            # Output: (n_edges, n_edges) double-swap rates
            self.head = PairOfPairsAttentionHead(hidden_dim)
        else:
            # General k-body: set-pooled scoring
            self.head = GeneralTransitionHead(hidden_dim, transition_order)
    
    def forward(self, node_features, edge_index, edge_features,
                global_features, transition_mask):
        """
        Args:
            node_features: (batch, n, d_node)
            edge_index: (2, n_edges)
            edge_features: (batch, n_edges, d_edge) or None
            global_features: (batch, d_global)
            transition_mask: (batch, ...) valid transition mask
        Returns:
            rates: (batch, ...) predicted rates, same shape as mask,
                   zero where mask is zero
        """
        # 1. GNN backbone
        h = self.backbone(node_features, edge_index, edge_features,
                          global_features)  # (batch, n, hidden_dim)
        
        # 2. Scoring head
        raw_rates = self.head(h, edge_index)  # (batch, ...)
        
        # 3. Mask and ensure non-negative
        rates = torch.nn.functional.softplus(raw_rates) * transition_mask
        
        return rates
```

### Scoring heads

```python
class SingleNodeHead(nn.Module):
    """k=1: predict rate for each (node, target_label) pair."""
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.mlp = MLP(hidden_dim, hidden_dim, vocab_size)
    
    def forward(self, h, edge_index):
        # h: (batch, n, d)
        return self.mlp(h)  # (batch, n, vocab_size)


class PairwiseAttentionHead(nn.Module):
    """k=2, binary vocab: attention between 1-nodes and 0-nodes."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, h, edge_index):
        # h: (batch, n, d)
        Q = self.W_Q(h)  # (batch, n, d) -- queries from 1-nodes
        K = self.W_K(h)  # (batch, n, d) -- keys from 0-nodes
        # Raw scores for all pairs
        scores = torch.bmm(Q, K.transpose(1, 2))  # (batch, n, n)
        return scores  # masking applied in forward() of parent


class PairwiseAttentionWithLabelsHead(nn.Module):
    """k=2, general vocab: attention with label embeddings."""
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.label_embed = nn.Embedding(vocab_size, hidden_dim // 4)
        self.W_Q = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        self.W_K = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
    
    def forward(self, h, edge_index, labels=None):
        # Concatenate label embeddings with node embeddings
        # then compute pairwise scores
        ...


class PairOfPairsAttentionHead(nn.Module):
    """k=4: score pairs of edge-nodes for double swaps."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, h, edge_index):
        # Compute edge embeddings from endpoint node embeddings
        src, dst = edge_index
        e = self.edge_mlp(torch.cat([h[:, src], h[:, dst]], dim=-1))
        # Pairwise attention over edges
        Q = self.W_Q(e)
        K = self.W_K(e)
        scores = torch.bmm(Q, K.transpose(1, 2))
        return scores
```

### Training loop (problem-independent)

```python
# config_fm/train.py

def train_configuration_fm(model, config_space, target_sampler,
                           n_epochs, batch_size, lr, device,
                           seed=42, **kwargs):
    """Train configuration flow matching model.
    
    This is entirely problem-independent. The ConfigurationSpace
    handles all problem-specific logic.
    
    Args:
        model: ConfigurationRatePredictor
        config_space: ConfigurationSpace instance
        target_sampler: callable(rng, **kwargs) -> config
        n_epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        device: torch device
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    
    edge_index = config_space.position_graph_edges()
    edge_features = config_space.position_edge_features()
    
    # Convert to tensors
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_features_t = (torch.tensor(edge_features, dtype=torch.float32,
                                     device=device)
                       if edge_features is not None else None)
    
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for _ in range(max(1, 1000 // batch_size)):
            batch_data = []
            for _ in range(batch_size):
                # 1. Sample source and target
                config_0 = config_space.sample_source(rng)
                config_T = config_space.sample_target(rng, **kwargs)
                
                # 2. Sample flow time
                t = float(rng.uniform(0.0, 0.999))
                
                # 3. Sample intermediate configuration
                config_t, n_done, n_rem = config_space.sample_intermediate(
                    config_0, config_T, t, rng)
                
                # 4. Compute target rates (without 1/(1-t) factor)
                target_rates = config_space.compute_target_rates(
                    config_0, config_T, config_t, t)
                
                # 5. Compute features
                node_feat = config_space.node_features(config_t)
                global_feat = config_space.global_features(t=t, **kwargs)
                mask = config_space.transition_mask(config_t)
                
                batch_data.append((node_feat, global_feat,
                                   target_rates, mask))
            
            # Stack batch
            node_feats = torch.tensor(
                np.array([d[0] for d in batch_data]),
                dtype=torch.float32, device=device)
            global_feats = torch.tensor(
                np.array([d[1] for d in batch_data]),
                dtype=torch.float32, device=device)
            target_rates_t = torch.tensor(
                np.array([d[2] for d in batch_data]),
                dtype=torch.float32, device=device)
            masks = torch.tensor(
                np.array([d[3] for d in batch_data]),
                dtype=torch.float32, device=device)
            
            # Forward pass
            pred_rates = model(node_feats, edge_index_t,
                               edge_features_t, global_feats, masks)
            
            # Rate KL loss
            loss = rate_kl_loss(pred_rates, target_rates_t, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.6f}",
                  flush=True)
    
    return {'losses': losses}
```

### Inference (problem-independent)

```python
# config_fm/sample.py

def generate_samples(model, config_space, n_samples, n_steps=100,
                     device='cpu', seed=0, batch_size=512,
                     parallel=False, **kwargs):
    """Generate samples via learned configuration flow matching.
    
    Entirely problem-independent. The ConfigurationSpace handles
    transition application and validity.
    
    Args:
        model: trained ConfigurationRatePredictor
        config_space: ConfigurationSpace instance
        n_samples: number of samples to generate
        n_steps: number of integration steps
        device: torch device
        parallel: if True, apply multiple independent transitions per step
        **kwargs: passed to config_space.global_features()
    """
    model.eval()
    rng = np.random.default_rng(seed)
    dt = 1.0 / n_steps
    
    edge_index = config_space.position_graph_edges()
    edge_features = config_space.position_edge_features()
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_features_t = (torch.tensor(edge_features, dtype=torch.float32,
                                     device=device)
                       if edge_features is not None else None)
    
    all_samples = []
    
    for batch_start in range(0, n_samples, batch_size):
        B = min(batch_size, n_samples - batch_start)
        configs = np.array([config_space.sample_source(rng) for _ in range(B)])
        
        with torch.no_grad():
            for step in range(n_steps):
                t = step * dt
                
                # Compute features for all configs in batch
                node_feats = torch.tensor(
                    np.array([config_space.node_features(c) for c in configs]),
                    dtype=torch.float32, device=device)
                global_feats = torch.tensor(
                    np.array([config_space.global_features(t=t, **kwargs)
                              for _ in range(B)]),
                    dtype=torch.float32, device=device)
                masks = torch.tensor(
                    np.array([config_space.transition_mask(c)
                              for c in configs]),
                    dtype=torch.float32, device=device)
                
                # Predict rates
                rates = model(node_feats, edge_index_t, edge_features_t,
                              global_feats, masks)
                rates = rates / (1.0 - t + 1e-10)
                rates_np = rates.cpu().numpy()
                
                # Apply transitions
                for b in range(B):
                    rate_flat = rates_np[b].flatten()
                    total_rate = rate_flat.sum()
                    if total_rate <= 0:
                        continue
                    
                    if parallel:
                        # Sample multiple independent transitions
                        n_events = rng.poisson(total_rate * dt)
                        probs = rate_flat / total_rate
                        for _ in range(n_events):
                            idx = rng.choice(len(probs), p=probs)
                            new_config = config_space.apply_transition(
                                configs[b], idx)
                            if new_config is not None:
                                configs[b] = new_config
                                # Recompute probs if transitions changed
                                # (for correlated transitions)
                                break  # conservative: one per step
                    else:
                        # Single transition per step
                        n_events = rng.poisson(total_rate * dt)
                        if n_events > 0:
                            probs = rate_flat / total_rate
                            idx = rng.choice(len(probs), p=probs)
                            new_config = config_space.apply_transition(
                                configs[b], idx)
                            if new_config is not None:
                                configs[b] = new_config
        
        all_samples.append(configs)
    
    return np.concatenate(all_samples, axis=0)
```

### Rate KL loss (problem-independent)

```python
# config_fm/loss.py

def rate_kl_loss(pred_rates, target_rates, mask):
    """Rate KL divergence loss for configuration flow matching.
    
    D(r || r_theta) = sum [r log(r/r_theta) - r + r_theta]
    
    Summed over all valid transitions (where mask > 0).
    """
    eps = 1e-10
    # Only compute where target > 0 (geodesic-progressing transitions)
    active = (target_rates > eps) & (mask > 0)
    
    loss_active = (target_rates[active]
                   * torch.log(target_rates[active]
                               / (pred_rates[active] + eps))
                   - target_rates[active]
                   + pred_rates[active])
    
    # For non-progressing valid transitions: target=0, loss = pred_rate
    # (penalize predicting rate where there should be none)
    inactive = (~active) & (mask > 0)
    loss_inactive = pred_rates[inactive]
    
    return (loss_active.sum() + loss_inactive.sum()) / mask.sum()
```

## Johnson graph instance

```python
# config_fm/spaces/johnson.py

class JohnsonSpace(ConfigurationSpace):
    """Configuration space for J(n,k): binary strings with fixed
    Hamming weight k.
    
    Position graph: complete graph K_n (or with edge features from J).
    Vocabulary: {0, 1}
    Invariant: sum(labels) = k
    Transitions: swap one 1-position with one 0-position (k=2)
    """
    
    def __init__(self, n, k, J_coupling, h_field, beta_range=(0.5, 2.0)):
        self.n = n
        self.k = k
        self.J = J_coupling          # (n, n) Ising coupling matrix
        self.h = h_field              # (n,) external field
        self.beta_range = beta_range
        self._current_beta = 1.0      # set per sample
    
    @property
    def n_positions(self):
        return self.n
    
    @property
    def vocab_size(self):
        return 2
    
    @property
    def transition_order(self):
        return 2
    
    def position_graph_edges(self):
        """Complete graph K_n."""
        src, dst = [], []
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    src.append(i)
                    dst.append(j)
        return np.array([src, dst])
    
    def position_edge_features(self):
        """J[i,j] coupling strengths as edge features."""
        edges = self.position_graph_edges()
        feats = np.array([self.J[edges[0, e], edges[1, e]]
                          for e in range(edges.shape[1])],
                         dtype=np.float32)
        return feats[:, None]  # (n_edges, 1)
    
    def node_features(self, config):
        """[x_i, h_i] per node."""
        return np.stack([config, self.h], axis=-1).astype(np.float32)
    
    def global_features(self, t=0.0, beta=1.0, **kwargs):
        """[t, beta]."""
        return np.array([t, beta], dtype=np.float32)
    
    def transition_mask(self, config):
        """(n, n) mask: 1 where config[i]=1 and config[j]=0."""
        mask = config[:, None] * (1 - config[None, :])
        return mask.astype(np.float32)
    
    def apply_transition(self, config, transition_idx):
        """Apply swap. transition_idx indexes into flattened (n, n) array."""
        i = transition_idx // self.n
        j = transition_idx % self.n
        if config[i] == 1 and config[j] == 0:
            new_config = config.copy()
            new_config[i] = 0
            new_config[j] = 1
            return new_config
        return None  # invalid transition
    
    def geodesic_distance(self, config_a, config_b):
        S_plus = np.where((config_a == 1) & (config_b == 0))[0]
        return len(S_plus)
    
    def sample_intermediate(self, config_0, config_T, t, rng):
        S_plus = np.where((config_0 == 1) & (config_T == 0))[0]
        S_minus = np.where((config_0 == 0) & (config_T == 1))[0]
        d = len(S_plus)
        
        if d == 0:
            return config_0.copy(), 0, 0
        
        ell = rng.binomial(d, t)
        
        A = rng.choice(S_plus, size=ell, replace=False) if ell > 0 else []
        B = rng.choice(S_minus, size=ell, replace=False) if ell > 0 else []
        
        config_t = config_0.copy()
        config_t[A] = 0
        config_t[B] = 1
        
        return config_t, ell, d - ell
    
    def compute_target_rates(self, config_0, config_T, config_t, t):
        """Target rate = 1/(d - ell) for geodesic-progressing swaps, 0 otherwise.
        
        Geodesic-progressing: swap (i, j) where i is in S_plus_remaining
        (still 1 in config_t, should be 0 in config_T) and j is in
        S_minus_remaining (still 0 in config_t, should be 1 in config_T).
        
        Returns (n, n) array. Does NOT include 1/(1-t) factor.
        """
        S_plus_rem = np.where((config_t == 1) & (config_T == 0))[0]
        S_minus_rem = np.where((config_t == 0) & (config_T == 1))[0]
        d_rem = len(S_plus_rem)
        
        rates = np.zeros((self.n, self.n), dtype=np.float32)
        if d_rem > 0:
            for i in S_plus_rem:
                for j in S_minus_rem:
                    rates[i, j] = 1.0 / d_rem
        
        return rates
    
    def sample_source(self, rng):
        """Uniform random binary string with k ones."""
        config = np.zeros(self.n, dtype=np.float32)
        ones = rng.choice(self.n, size=self.k, replace=False)
        config[ones] = 1.0
        return config
    
    def sample_target(self, rng, beta=1.0, mcmc_pool=None, **kwargs):
        """Sample from Boltzmann distribution via MCMC pool."""
        if mcmc_pool is not None:
            idx = rng.integers(len(mcmc_pool))
            return mcmc_pool[idx].copy()
        else:
            # Fallback: run MCMC
            from johnson_fm.energy import mcmc_kawasaki, ising_energy
            energy_fn = lambda x: ising_energy(x, self.J, self.h)
            return mcmc_kawasaki(energy_fn, self.n, self.k, beta, 5000, rng)

    def enumerate_transitions(self, config):
        """For documentation/debugging. Returns structured transition info."""
        ones = np.where(config == 1)[0]
        zeros = np.where(config == 0)[0]
        S_minus_list = []  # nodes going 1->0
        S_plus_list = []   # nodes going 0->1
        for i in ones:
            for j in zeros:
                S_minus_list.append(np.array([i]))
                S_plus_list.append(np.array([j]))
        return {
            'S_minus': S_minus_list,
            'S_plus': S_plus_list,
            'n_transitions': len(S_minus_list),
        }
```

## DFM baseline instance

```python
# config_fm/spaces/dfm.py

class DFMSpace(ConfigurationSpace):
    """DFM on {0,1}^n (unconstrained binary strings).
    
    Position graph: complete graph K_n
    Vocabulary: {0, 1}
    Invariant: none
    Transitions: single bit flip (k=1)
    """
    
    def __init__(self, n, J_coupling, h_field):
        self.n = n
        self.J = J_coupling
        self.h = h_field
    
    @property
    def n_positions(self):
        return self.n
    
    @property
    def vocab_size(self):
        return 2
    
    @property
    def transition_order(self):
        return 1
    
    def position_graph_edges(self):
        """Complete graph K_n."""
        src, dst = [], []
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    src.append(i)
                    dst.append(j)
        return np.array([src, dst])
    
    def position_edge_features(self):
        edges = self.position_graph_edges()
        feats = np.array([self.J[edges[0, e], edges[1, e]]
                          for e in range(edges.shape[1])],
                         dtype=np.float32)
        return feats[:, None]
    
    def node_features(self, config):
        return np.stack([config, self.h], axis=-1).astype(np.float32)
    
    def global_features(self, t=0.0, beta=1.0, **kwargs):
        return np.array([t, beta], dtype=np.float32)
    
    def transition_mask(self, config):
        """(n,) mask: all positions can flip."""
        return np.ones(self.n, dtype=np.float32)
    
    def apply_transition(self, config, transition_idx):
        """Flip bit at position transition_idx."""
        new_config = config.copy()
        new_config[transition_idx] = 1.0 - new_config[transition_idx]
        return new_config
    
    def geodesic_distance(self, config_a, config_b):
        return int(np.sum(config_a != config_b))
    
    def sample_intermediate(self, config_0, config_T, t, rng):
        diff = np.where(config_0 != config_T)[0]
        d = len(diff)
        if d == 0:
            return config_0.copy(), 0, 0
        ell = rng.binomial(d, t)
        flipped = rng.choice(diff, size=ell, replace=False) if ell > 0 else []
        config_t = config_0.copy()
        config_t[flipped] = config_T[flipped]
        return config_t, ell, d - ell
    
    def compute_target_rates(self, config_0, config_T, config_t, t):
        """Per-position flip rate. 1/d_rem for positions that still
        differ from target, 0 otherwise."""
        diff_rem = np.where(config_t != config_T)[0]
        d_rem = len(diff_rem)
        rates = np.zeros(self.n, dtype=np.float32)
        if d_rem > 0:
            rates[diff_rem] = 1.0 / d_rem
        return rates
    
    def sample_source(self, rng):
        return rng.binomial(1, 0.5, size=self.n).astype(np.float32)
    
    def sample_target(self, rng, beta=1.0, mcmc_pool=None, **kwargs):
        if mcmc_pool is not None:
            idx = rng.integers(len(mcmc_pool))
            return mcmc_pool[idx].copy()
        else:
            raise ValueError("DFM needs precomputed target pool")
```

## Experiment script

```python
# experiments/ex20_johnson_graph.py

# This becomes a thin script that:
# 1. Creates JohnsonSpace and DFMSpace instances
# 2. Creates ConfigurationRatePredictor models
#    (one with transition_order=2 for FM, one with transition_order=1 for DFM)
# 3. Calls train_configuration_fm() for each
# 4. Calls generate_samples() for each
# 5. Runs evaluation (energy stats, correlations, TV, validity)
# 6. Generates plots
#
# The experiment-specific code is ONLY:
# - Ising model generation (J, h)
# - MCMC pool generation
# - Evaluation metrics
# - Plotting
#
# All flow matching logic is in config_fm/.
```

## Migration steps

1. Create `config_fm/` package with the abstract base class and
   problem-independent training/inference/loss code.

2. Implement `JohnsonSpace` in `config_fm/spaces/johnson.py`.

3. Implement `DFMSpace` in `config_fm/spaces/dfm.py`.

4. Implement `ConfigurationRatePredictor` with `SingleNodeHead` (k=1)
   and `PairwiseAttentionHead` (k=2).

5. Implement `FiLMGNNBackbone` — can adapt from existing
   `FiLMConditionalGNNRateMatrixPredictor` in `meta_fm/model.py`.

6. Refactor `ex20_johnson_graph.py` to use the new framework.
   Keep all evaluation and plotting code in the experiment script.

7. Verify: rerun Ex20 with the refactored code and confirm results
   match the previous implementation (same model, same training, same
   outputs — just different code organization).

8. After verification, delete `johnson_fm/` (the old ad-hoc package).

## Key design principles

- **ConfigurationSpace is a pure numpy class.** No torch dependency.
  It defines the combinatorial structure. Tensor conversion happens
  in the training loop and model.

- **The training loop never imports from a specific space.** It only
  calls ConfigurationSpace methods. Adding a new problem requires zero
  changes to training or inference code.

- **The model's scoring head is selected by transition_order at init time.**
  The forward pass is polymorphic — same call signature regardless of
  the head type.

- **Evaluation is problem-specific** and stays in the experiment script.
  The framework doesn't try to generalize metrics.
