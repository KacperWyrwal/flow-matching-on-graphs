# Spec: Lattice Gas on 2D Grid (Ex20b)

## Overview

Binary lattice gas: place $k$ atoms on an $L \times L$ grid. Same
Johnson graph $J(L^2, k)$ as Ex20, same `JohnsonSpace`, same model.
The only changes:

1. Energy uses lattice neighbor interactions instead of random $J_{ij}$
2. Visualization renders samples as $L \times L$ colored grids
3. Evaluation includes spatial metrics (domain size, clustering)

This is a visualization and framing change, not a code change. It
demonstrates the same framework on a physically interpretable problem.

## Setup

**Lattice:** $L \times L$ periodic grid (torus). $N = L^2$ sites.

**Labels:** $\{0, 1\}$ — empty or occupied.

**Constraint:** Exactly $k$ sites occupied (fixed density $\rho = k/N$).

**Energy:** Nearest-neighbor attraction on the lattice:
$$E(x) = -J \sum_{\langle i,j \rangle} x_i x_j$$

where the sum is over lattice neighbor pairs (4 per site on square
lattice). This favors clustering of occupied sites.

**Coupling matrix:** $J_{ij} = J$ if $i,j$ are lattice neighbors,
$J_{ij} = 0$ otherwise. This is the input to the model via edge
features — sparse (4 neighbors per site) instead of dense (random).

## Sizes

**Small (development):** $L = 6$, $N = 36$, $k = 12$ (density 1/3).
Configuration space: $\binom{36}{12} \approx 1.3 \times 10^9$.

**Medium (paper):** $L = 8$, $N = 64$, $k = 16$ (density 1/4).
Configuration space: $\binom{64}{16} \approx 4.9 \times 10^{13}$.

Both are far too large for explicit enumeration but run at the same
speed as Ex20 since $n$ is comparable (36 or 64 vs Ex20's 16-24).

## Beta values

With $J = 1$:
- $\beta = 0.2$: high temperature, random gas
- $\beta = 0.5$: mild clustering
- $\beta = 1.0$: clear clustering, small domains
- $\beta = 2.0$: strong clustering, large compact domains

The 2D lattice gas with fixed density has a phase transition. At low
density ($\rho \ll 1$), the transition temperature depends on $\rho$.
The exact critical $\beta$ varies, but $\beta \in [0.5, 2.0]$ should
capture the interesting range.

## Implementation

### File: `experiments/ex20b_lattice_gas.py`

Reuses `JohnsonSpace` from `otfm/configuration/spaces/johnson.py`.
The only difference from Ex20 is how $J$ and $h$ are constructed:

```python
# Lattice neighbor coupling (instead of random J)
n = L * L
J = np.zeros((n, n))
for y in range(L):
    for x in range(L):
        i = y * L + x
        # Right neighbor (periodic)
        j = y * L + (x + 1) % L
        J[i, j] = J[j, i] = 1.0
        # Down neighbor (periodic)
        j = ((y + 1) % L) * L + x
        J[i, j] = J[j, i] = 1.0

# No external field
h = np.zeros(n)
```

Everything else — `JohnsonSpace`, `ConfigurationRatePredictor`,
`train_configuration_fm`, `generate_samples` — is identical to Ex20.

### MCMC

Same Kawasaki-style MCMC as Ex20 (any-pair swaps with Metropolis
acceptance), using the lattice energy function. The MCMC doesn't need
to be local — it just needs to sample the Boltzmann distribution.

### Model

Same as Ex20:
```python
ConfigurationRatePredictor(
    node_feature_dim=2,   # [x_i, h_i] = [x_i, 0]
    edge_feature_dim=1,   # J_ij (lattice neighbor = 1, else = 0)
    global_dim=2,         # [t, beta]
    hidden_dim=128,
    n_layers=4,
    transition_order=2,
)
```

The model receives the lattice structure through edge features.
Non-neighbor pairs have $J_{ij} = 0$ as edge feature, so the model
learns that swaps between distant sites don't directly affect the
energy.

Note: since $h = 0$ everywhere, node_feature_dim could be reduced
to 1 (just $x_i$). But keeping it at 2 maintains compatibility with
the Ex20 `JohnsonSpace` code which returns `[x_i, h_i]`.

## Visualization

### Hero figure: Sample gallery at different β

4 rows × 6 columns. Each row is a different $\beta$. Each sample is
an $L \times L$ grid rendered with a colormap:
- Occupied sites ($x_i = 1$): colored (e.g., dark blue)
- Empty sites ($x_i = 0$): white or light gray

This shows the phase transition: random scatter at $\beta = 0.2$,
mild clustering at $\beta = 0.5$, clear domains at $\beta = 1.0$,
large compact clusters at $\beta = 2.0$.

### Comparison figure: FM vs MCMC vs DFM at high β

3 rows (FM, DFM valid, MCMC-5000) × 6 columns (random samples).
Shows whether FM captures the same domain structure as the ground
truth.

### Trajectory figure: Generation sequence

One row showing a single FM generation at 6 time points
($t = 0, 0.2, 0.4, 0.6, 0.8, 1.0$). Shows atoms coalescing from
random placement into a compact cluster.

### Quantitative panels (same as Ex20)

- Energy bias vs $\beta$
- Correlation RMSE vs $\beta$
- DFM validity rate
- Domain size distribution (new): histogram of connected component
  sizes at high $\beta$, FM vs MCMC vs true

## Evaluation metrics

All Ex20 metrics plus:

**Domain size distribution:** Connected components of occupied sites
on the lattice (4-connectivity). At high β, true samples have a few
large domains. FM should match this distribution.

```python
from scipy.ndimage import label
def compute_domains(config, L):
    grid = config.reshape(L, L)
    labeled, n_domains = label(grid)
    sizes = [int((labeled == c).sum()) for c in range(1, n_domains + 1)]
    return sizes
```

**Spatial correlation function:** $C(r) = \langle x_i x_{i+r} \rangle$
as a function of Manhattan distance $r$. At high β, should show
positive correlation decaying over the domain size scale.

**Surface-to-volume ratio:** For the largest domain, ratio of boundary
sites (occupied with at least one empty neighbor) to total sites.
Compact domains have low ratio; fractal/diffuse domains have high
ratio. FM should match the true distribution.

## Expected results

**FM:** 100% validity (fixed Hamming weight by construction). Should
capture domain formation at high β, with energy and correlations
close to MCMC.

**DFM:** Low validity — same as Ex20 but more dramatic at larger $n$.
For $L=8$, $k=16$: $\binom{64}{16}/2^{64} \approx 2.7 \times 10^{-6}$.
Essentially zero valid samples. DFM is completely useless here.

**MCMC:** Ground truth (with sufficient chain length). Provides
reference energy, correlations, and domain statistics.

The comparison with DFM is particularly striking in this setting:
DFM produces essentially no valid samples, while FM produces
physically realistic lattice gas configurations with correct domain
structure — and both methods see the same energy function.

## Training parameters

```python
L = 8
n = 64
k = 16
betas = [0.2, 0.5, 1.0, 2.0]

n_epochs = 2000
batch_size = 256
hidden_dim = 128
n_layers = 4
lr = 5e-4
mcmc_pool_size = 10000
mcmc_chain_length = 5000
n_eval_samples = 2000
n_gen_steps = 100
```
