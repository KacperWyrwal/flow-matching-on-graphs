# Spec: Device Acceleration + Experiment 7 (Generative Modeling)

## Part 1: Device Detection Refactor

### Problem

All current experiments hardcode `device='cpu'`. As we scale to larger graphs and
more training samples, GPU/MPS acceleration becomes important.

### Changes to `meta_fm/utils.py` (new file or add to existing utils)

```python
import torch

def get_device() -> torch.device:
    """
    Auto-detect the best available device.
    Priority: CUDA > MPS > CPU.
    
    Returns torch.device.
    Prints which device was selected (once).
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device
```

### Changes to `meta_fm/train.py`

Replace hardcoded `device='cpu'` with auto-detection:

```python
from meta_fm.utils import get_device

def train(model, dataset, n_epochs=500, batch_size=64, lr=1e-3, device=None):
    if device is None:
        device = get_device()
    model = model.to(device)
    # ... rest of training loop unchanged, but ensure:
    # - All tensors from dataset are moved to device in the training loop
    # - mu.to(device), t.to(device), R_target.to(device) before forward pass
```

### Changes to `meta_fm/sample.py`

```python
def sample_trajectory(model, mu_start, n_steps=200, device=None):
    if device is None:
        device = get_device()
    model = model.to(device)
    # ... tensors moved to device during integration
```

Same for `backward_trajectory` if it exists.

### Changes to `meta_fm/model.py`

The GNNRateMatrixPredictor stores edge_index as a buffer via register_buffer,
which means it automatically moves to the correct device with model.to(device).
No changes needed to the model itself.

One thing to verify: when constructing PyG Batch objects inside forward(), the
Data objects should inherit the device from the input tensors:

```python
def forward(self, mu, t):
    # mu and t are already on the correct device
    # Data objects should use the same device
    data_list = []
    for b in range(B):
        data_list.append(Data(x=node_features[b], edge_index=self.edge_index))
    # self.edge_index is a buffer, already on the right device
    # node_features[b] is derived from mu, already on the right device
    # So Batch.from_data_list will produce a batch on the correct device
```

### Changes to experiment scripts

Each experiment script should replace:
```python
device = 'cpu'
```
with:
```python
from meta_fm.utils import get_device
device = get_device()
```

Apply this change to: ex3_meta_level.py, ex4_generalization.py,
ex5_source_localization.py, and the new ex7_generative_modeling.py.

Experiments 1 and 2 are pure NumPy (graph-level exact solver), no changes needed.

### MPS Caveats

Some PyTorch operations may not be supported on MPS yet. If an operation fails
on MPS, fall back gracefully:

```python
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Test MPS with a small operation to ensure it works
        try:
            test = torch.zeros(1, device='mps')
            _ = test + 1
            device = torch.device('mps')
            print("Using Apple MPS")
        except Exception:
            device = torch.device('cpu')
            print("MPS available but failed, falling back to CPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device
```

---

## Part 2: Experiment 7 — Discrete Generative Modeling

### Motivation

The core promise of flow matching on graphs is generative modeling over discrete
structures. Experiments 3-4 showed the meta-level GNN can learn to transport one
family of distributions to another. This experiment demonstrates actual generative
modeling: learning to sample structured distributions on a graph from a simple
prior.

The task: given a dataset of distributions that exhibit community structure on a
graph, learn a flow that transports uniform noise to the data distribution. At
inference, start from uniform and integrate forward to generate new samples that
look like the training data.

### Graph

3x3 grid graph (9 nodes). Small enough for fast training and easy visualization,
large enough to have meaningful spatial structure.

Node layout:
```
0-(0,0)  1-(0,1)  2-(0,2)
3-(1,0)  4-(1,1)  5-(1,2)
6-(2,0)  7-(2,1)  8-(2,2)
```

### Data Distribution: Community-Structured Distributions

The target data distribution is a MIXTURE of 4 community types. Each type
concentrates mass on one corner region of the grid:

- **Community A (top-left)**: nodes {0, 1, 3}
- **Community B (top-right)**: nodes {1, 2, 5}
- **Community C (bottom-left)**: nodes {3, 6, 7}
- **Community D (bottom-right)**: nodes {5, 7, 8}

To generate a sample from community X:
1. Pick the 3 nodes belonging to community X.
2. Draw weights from Dirichlet(alpha=5) for those 3 nodes.
3. Assign small uniform noise (0.01) to the remaining 6 nodes.
4. Renormalize to sum to 1.

Generate 200 data samples total: 50 from each community, shuffled.

This creates a multimodal target distribution over the simplex — the generative
model must learn all 4 modes.

### Prior Distribution

200 near-uniform distributions on the 9-node graph. Generate each as: start
from (1/9, ..., 1/9), add Gaussian noise (std=0.02), clip, renormalize.

### Training

1. Build the 3x3 grid graph and GraphStructure.
2. Compute cost matrix c(i,j).
3. Generate 200 prior samples and 200 data samples.
4. Compute meta-cost matrix W(mu_prior_s, mu_data_t) for all pairs.
5. Solve meta-OT coupling between prior and data.
6. Pre-generate 10000 training samples (mu_tau, tau, R_target).
   More samples than previous experiments because the target distribution
   is more complex (4 modes).
7. Train GNNRateMatrixPredictor:
   - hidden_dim = 64
   - n_layers = 4
   - 500 epochs, Adam lr=1e-3
   - batch_size = 64
   If loss still decreasing at 500, extend to 1000.
8. Save checkpoint as checkpoints/meta_model_generative_gnn.pt.

### Evaluation

**Generate 100 new samples:**
1. Start from a fresh near-uniform distribution (not from training set).
2. Integrate forward: dp/dt = p @ R_theta(p, t) from t=0 to t=1.
3. Output at t=1 is a generated sample.

**Community classification:**
For each generated sample, assign to the community with highest total mass
in its corner region ({0,1,3}, {1,2,5}, {3,6,7}, {5,7,8}).
Report distribution over communities. Should be roughly 25% each.

**Sharpness (entropy):**
Compare entropy histograms of real data vs generated samples. Generated
entropy should be comparably low (mass concentrated, not spread).

**Diversity (mode coverage):**
All 4 communities should be represented with >10% share each.

### Plots (single figure, 2x3 grid)

- **Panel A**: Training loss curve.

- **Panel B**: Real data examples. Show 8 samples as 3x3 grid heatmaps
  (2 from each community), arranged in a 2x4 sub-grid. Title: "Real data".
  
- **Panel C**: Generated samples. Show 8 generated samples as 3x3 grid heatmaps,
  arranged in a 2x4 sub-grid. Title: "Generated". Should visually resemble
  Panel B — mass concentrated in corner regions.

- **Panel D**: Community distribution. Grouped bar chart comparing fraction
  of samples in each community (A, B, C, D) for real data vs generated.

- **Panel E**: Entropy histogram. Overlaid histograms of H(p) for real data
  and generated samples. Generated entropy should overlap with real entropy.

- **Panel F**: Trajectory visualization. For ONE generated sample, show the
  3x3 grid heatmap at t = 0, 0.25, 0.5, 0.75, 1.0 (5 snapshots in a row).
  Should show mass starting uniform and progressively concentrating into one
  corner region.

### Validation Checks (print to console)

```
=== Experiment 7: Generative Modeling Results ===
Training: initial loss = X, final loss = Y
Generated 100 samples:
  Community distribution: A=XX%, B=XX%, C=XX%, D=XX%
  Mode coverage: X/4 communities with >10% representation
  Mean entropy (real): X.XX
  Mean entropy (generated): X.XX
  Entropy ratio (generated/real): X.XX (want ~1.0)
```

### Expected Outcome

The model should learn to generate distributions concentrated in corner regions,
with all 4 communities represented. Starting from uniform noise, the learned flow
creates structured distributions qualitatively indistinguishable from training data.

This is a proof of concept for discrete generative modeling via flow matching on
graphs — the meta-level model generates new distributions on a graph by running a
learned flow from noise.

### Dependencies

No new dependencies.
