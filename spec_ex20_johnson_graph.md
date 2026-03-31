# Spec: Flow Matching on the Johnson Graph (Ex20)

## Overview

Demonstrate flow matching on a combinatorially large configuration graph
where the constraint (fixed Hamming weight) is enforced by construction.
Learn to sample from an energy-based distribution on binary strings with
exactly $k$ ones out of $n$ positions, using the Johnson graph $J(n,k)$
as the state space.

This is the first experiment operating on an exponentially large configuration
graph via sample-level conditional flow matching. The GNN operates on $n$
position nodes, not on the $\binom{n}{k}$ configuration nodes.

## Problem setup

### State space

Binary strings $x \in \{0,1\}^n$ with $\sum_i x_i = k$. These are the nodes
of the Johnson graph $J(n,k)$. Two strings are connected by an edge if they
differ by swapping one 1→0 and one 0→1 (preserving Hamming weight).

### Target distribution

Ising model with random couplings at fixed magnetization:

$$p_{\text{target}}(x) \propto \exp(-\beta E(x)), \quad E(x) = -\sum_{i<j} J_{ij} x_i x_j - \sum_i h_i x_i$$

where $J_{ij}$ are random couplings and $h_i$ are random external fields,
restricted to strings with $\sum_i x_i = k$.

Generate $J$ and $h$ once per experiment instance:
```python
rng = np.random.default_rng(seed)
J = rng.standard_normal((n, n))
J = (J + J.T) / 2  # symmetric
np.fill_diagonal(J, 0)
h = rng.standard_normal(n) * 0.5
```

The inverse temperature $\beta$ controls difficulty:
- $\beta = 0$: uniform distribution (trivial)
- $\beta$ small: weakly structured, easy to sample
- $\beta$ large: multimodal, hard to sample

### Source distribution

Uniform over $J(n,k)$: sample by choosing $k$ positions uniformly at random
to set to 1.

### Problem sizes

Start with two scales:
- **Small:** $n=16, k=8$. Configuration space $\binom{16}{8} = 12870$.
  Small enough to compute exact statistics for validation.
- **Medium:** $n=24, k=12$. Configuration space $\binom{24}{12} = 2704156$.
  Too large for exact enumeration, must use sample-based metrics.

For each size, use $\beta \in \{0.5, 1.0, 2.0\}$ to vary difficulty.

## Theory: conditional flow on $J(n,k)$

### Geodesic between $x_0$ and $x_T$

Given two configurations $x_0, x_T$ with the same Hamming weight:
```python
S_plus = {i : x_0[i]=1 and x_T[i]=0}   # positions to turn off
S_minus = {i : x_0[i]=0 and x_T[i]=1}  # positions to turn on
d = |S_plus| = |S_minus|                 # geodesic distance
```

Each geodesic step picks one $i \in S_+^{\text{rem}}$ and one $j \in S_-^{\text{rem}}$
and swaps them ($x_i: 1 \to 0$, $x_j: 0 \to 1$).

### Sampling intermediate states $x_t$

At flow time $t \in [0,1)$:
1. Sample $\ell \sim \text{Bin}(d, t)$ — number of completed swaps
2. Sample $A \subseteq S_+$ with $|A| = \ell$ uniformly — which $S_+$ positions are done
3. Sample $B \subseteq S_-$ with $|B| = \ell$ uniformly — which $S_-$ positions are done
4. Construct $x_t$: start from $x_0$, flip positions in $A$ from 1→0, flip
   positions in $B$ from 0→1

Note: the choices of $A$ and $B$ are independent. The intermediate state $x_t$
always has exactly $k$ ones.

### Conditional swap rate

At $x_t$ with $d - \ell$ remaining swaps, the conditional rate for each valid
swap pair $(i,j)$ with $i \in S_+^{\text{rem}}, j \in S_-^{\text{rem}}$ is:

$$R_t^{x_0 \to x_T}(i, j) = \frac{1}{(1-t)(d - \ell)}$$

All valid geodesic-progressing swaps have equal rate (Johnson graph symmetry).
Total outgoing rate is $(d-\ell)^2 \cdot \frac{1}{(1-t)(d-\ell)} = \frac{d-\ell}{1-t}$.

### Target for the model

Factoring out $1/(1-t)$, the model learns:

$$\tilde{r}(i, j) = \frac{1}{d - \ell}$$

for valid (geodesic-progressing) swaps, and 0 otherwise.

The model doesn't know $S_+, S_-$, $d$, or $\ell$ — it must infer from the
current configuration $x_t$ and the conditioning context which swaps are useful.

## Architecture

### Input

The GNN operates on $n$ nodes (positions), not on the configuration space.

Node features for position $i$:
- $x_t[i] \in \{0, 1\}$ — current value
- Observation context (depends on conditioning — see below)

Global features:
- $t$ — flow time (via FiLM conditioning)
- $\beta$ — inverse temperature (if training across multiple $\beta$)

### Output

For each ordered pair of positions $(i, j)$ where $x_t[i] = 1$ and $x_t[j] = 0$,
predict the swap rate $\tilde{r}(i, j)$.

Implementation: the GNN produces node embeddings $h_i \in \mathbb{R}^d$. For
each valid swap pair, compute:

```python
swap_rate_ij = MLP([h_i, h_j])  # scalar, non-negative (softplus output)
```

This is similar to the edge-level rate prediction in our existing framework,
but the "edges" are all pairs $(i,j)$ with $x_t[i]=1, x_t[j]=0$ — they
change dynamically with the configuration.

### GNN graph structure

What graph does the GNN pass messages on? Two natural choices:

**Option A: Complete graph on $n$ positions.** Every position communicates with
every other. Simple but $O(n^2)$ messages per layer.

**Option B: Coupling graph from $J$.** Positions $i$ and $j$ are connected if
they interact in the energy function (i.e., $J_{ij} \neq 0$). For random $J$
this is complete, but for structured $J$ (e.g., nearest-neighbor Ising on a
lattice) this would be sparse.

For the random coupling case, use a complete graph (Option A). For $n=16$ or
$n=24$, this is tractable.

### Model details

```python
FiLMConditionalGNNSwapPredictor(
    node_feature_dim=1,     # just x_t[i]
    global_dim=2,           # [t, beta]
    hidden_dim=128,
    n_layers=4,
)
```

Use FiLM conditioning for $t$ and $\beta$ as in our existing architecture.

## Training

### Dataset generation

No precomputed dataset — generate training pairs on the fly each epoch.

```python
for each training step:
    # 1. Sample target configuration from Boltzmann via MCMC
    x_T = mcmc_sample(energy_fn, n, k, beta, n_steps=1000)
    
    # 2. Sample source configuration uniformly
    x_0 = uniform_sample(n, k)
    
    # 3. Compute geodesic structure
    S_plus = where(x_0 == 1 and x_T == 0)
    S_minus = where(x_0 == 0 and x_T == 1)
    d = len(S_plus)
    
    # 4. Sample flow time and intermediate state
    t = rng.uniform(0, 0.999)
    ell = rng.binomial(d, t)
    A = rng.choice(S_plus, size=ell, replace=False)
    B = rng.choice(S_minus, size=ell, replace=False)
    x_t = x_0.copy()
    x_t[A] = 0
    x_t[B] = 1
    
    # 5. Compute target rates
    S_plus_rem = S_plus - A
    S_minus_rem = S_minus - B
    d_rem = d - ell
    # Target: rate = 1/d_rem for all pairs (i,j) with i in S_plus_rem, j in S_minus_rem
    # Target: rate = 0 for all other pairs
    
    # 6. Train model to predict these rates
    predicted_rates = model(x_t, t, beta)
    loss = rate_kl_loss(predicted_rates, target_rates)
```

### MCMC for target samples

Use Kawasaki dynamics (swap MCMC) on $J(n,k)$ with Metropolis-Hastings:

```python
def mcmc_sample(energy_fn, n, k, beta, n_steps, rng):
    x = uniform_sample(n, k, rng)
    for _ in range(n_steps):
        # Propose: swap a random 1-position with a random 0-position
        ones = np.where(x == 1)[0]
        zeros = np.where(x == 0)[0]
        i = rng.choice(ones)
        j = rng.choice(zeros)
        x_prop = x.copy()
        x_prop[i] = 0
        x_prop[j] = 1
        # Accept/reject
        dE = energy_fn(x_prop) - energy_fn(x)
        if dE < 0 or rng.uniform() < np.exp(-beta * dE):
            x = x_prop
    return x
```

Pre-generate a pool of MCMC samples (e.g., 10,000 per $\beta$ value) to draw
from during training. Run MCMC chains long enough to decorrelate (monitor
autocorrelation of energy).

### Loss

Rate KL loss over all valid swap pairs:

$$L = \sum_{(i,j): x_t[i]=1, x_t[j]=0} \left[ r_{ij} \log\frac{r_{ij}}{r_{ij}^\theta} - r_{ij} + r_{ij}^\theta \right]$$

where $r_{ij}$ is the target rate and $r_{ij}^\theta$ is the predicted rate.

### Training parameters

```python
n_epochs = 2000
lr = 5e-4
batch_size = 256          # number of (x_0, x_T) pairs per batch
mcmc_pool_size = 10000    # pre-generated target samples per beta
mcmc_chain_length = 5000  # steps per MCMC chain
```

## Baselines

### 1. DFM on $\{0,1\}^n$ (unconstrained)

Standard discrete flow matching treating each position independently.
Source: uniform over $\{0,1\}^n$ (no Hamming weight constraint).
Target: same Boltzmann distribution.

Each position has independent rates $0 \to 1$ and $1 \to 0$.
A neural network (same GNN architecture) predicts per-position flip rates.

At generation time, DFM produces strings with arbitrary Hamming weight.
Measure validity rate: fraction with exactly $k$ ones.

Implementation: use our existing framework on $K_2$ per position (product
graph), with the same energy-based target.

### 2. DFM + rejection

Same as DFM, but reject generated samples with wrong Hamming weight.
This produces valid samples but at cost of 1/(validity rate) overhead.
Report effective sample rate.

### 3. MCMC (Kawasaki dynamics)

Swap MCMC as described above, run for a fixed number of steps from a
uniform initialization. This is the classical baseline — valid by
construction, correct asymptotically, but slow near phase transitions.

Run for $T_{\text{mcmc}} \in \{100, 500, 1000, 5000\}$ steps and report
sample quality at each budget. This shows mixing time.

## Evaluation

### Small scale ($n=16, k=8$): exact validation

The configuration space (12870 states) is small enough to compute the
exact Boltzmann distribution. Evaluate:

**TV distance:** Between the empirical distribution of generated samples
and the exact target distribution. Generate 100,000 samples, compute
histogram over all 12870 states, compare to $p_{\text{target}}$.

**Log-likelihood:** Mean $\log p_{\text{target}}(x)$ over generated samples.
Higher is better. This measures whether the model finds high-probability
regions.

**Energy statistics:** Compare histograms of $E(x)$ for generated vs true
samples. Compute mean, variance, and KS test.

**Pairwise correlations:** Compare $\langle x_i x_j \rangle$ under generated
vs true distribution. Report RMSE of the correlation matrix.

**Validity (DFM only):** Fraction of samples with $\sum x_i = k$.

### Medium scale ($n=24, k=12$): sample-based metrics

Cannot compute exact distribution. Use sample-based metrics:

**Energy statistics:** As above.

**Pairwise correlations:** As above, compared against a long MCMC run
(ground truth).

**Effective sample size (ESS):** For MCMC baseline, compute ESS from
autocorrelation. For FM, all samples are independent by construction.

**Validity (DFM only):** As above.

### Metrics across $\beta$

Report all metrics at $\beta \in \{0.5, 1.0, 2.0\}$. The key comparison:
at high $\beta$, MCMC suffers from slow mixing (critical slowing down),
DFM wastes more samples on invalid configurations, and FM should maintain
quality.

## Output

### Console table

```
Method          beta   TV     Energy   Corr    Validity  Eff. rate
                       (↓)    RMSE(↓)  RMSE(↓) (↑)       (↑)
-------------------------------------------------------------------
FM (ours)       0.5    ...    ...      ...      100%      100%
FM (ours)       1.0    ...    ...      ...      100%      100%
FM (ours)       2.0    ...    ...      ...      100%      100%
DFM             0.5    ...    ...      ...      ...%      ...%
DFM             1.0    ...    ...      ...      ...%      ...%
DFM             2.0    ...    ...      ...      ...%      ...%
DFM+reject      0.5    ...    ...      ...      100%      ...%
DFM+reject      1.0    ...    ...      ...      100%      ...%
DFM+reject      2.0    ...    ...      ...      100%      ...%
MCMC-1000       0.5    ...    ...      ...      100%      ...
MCMC-1000       1.0    ...    ...      ...      100%      ...
MCMC-1000       2.0    ...    ...      ...      100%      ...
```

### Figure: `experiments/ex20_johnson_graph.png` (2x3 panels)

**Panel A: Energy histograms** at $\beta=2.0$ (hardest case).
Overlaid histograms for FM, DFM, DFM+reject, MCMC, and exact (if n=16).

**Panel B: Pairwise correlation scatter.**
True correlations $\langle x_i x_j \rangle$ vs model correlations for FM
and DFM. Points near diagonal = good.

**Panel C: TV vs $\beta$** (n=16 only).
Lines for FM, DFM+reject, and MCMC at various budgets.

**Panel D: Validity rate vs $n$.**
DFM validity at $k=n/2$ for $n \in \{12, 16, 20, 24, 28\}$. Shows
exponential decay. FM is always 100%.

**Panel E: Sample quality vs MCMC budget.**
TV or energy RMSE as a function of MCMC steps. Shows that FM achieves in
one shot what MCMC needs thousands of steps for.

**Panel F: Generated samples gallery.**
For $n=16$: show 10 generated samples as binary grids (4x4), colored by
$x_i$. Show FM, DFM (including invalid ones), and true samples side by side.

## Implementation

### File: `experiments/ex20_johnson_graph.py`

Standalone script. Does not depend on the existing `graph_ot_fm` or `meta_fm`
packages (those operate on explicit graph distributions). This experiment
uses sample-level CFM and needs new infrastructure:

### New modules needed:

**`johnson_fm/energy.py`:** Energy functions, MCMC sampling.
```python
def ising_energy(x, J, h):
    return -0.5 * x @ J @ x - h @ x

def mcmc_kawasaki(energy_fn, n, k, beta, n_steps, rng):
    ...
```

**`johnson_fm/model.py`:** GNN swap rate predictor.
```python
class SwapRatePredictor(nn.Module):
    """GNN on n position nodes, predicts rates for valid swaps."""
    def forward(self, x_t, t, beta, J, h):
        # x_t: (batch, n) binary
        # Returns: (batch, n, n) swap rates, masked to valid pairs
        ...
```

**`johnson_fm/flow.py`:** Conditional flow computations.
```python
def sample_intermediate(x_0, x_T, t, rng):
    """Sample x_t along geodesic on J(n,k)."""
    ...

def compute_target_rates(x_0, x_T, x_t, t):
    """Compute conditional swap rates at x_t."""
    ...
```

**`johnson_fm/dfm_baseline.py`:** DFM on {0,1}^n.
```python
class DFMBitFlipPredictor(nn.Module):
    """Per-position flip rate predictor (DFM baseline)."""
    ...
```

**`johnson_fm/train.py`:** Training loop with on-the-fly data generation.

### Key implementation detail: masking valid swaps

At configuration $x_t$, valid swaps are pairs $(i,j)$ with $x_t[i]=1, x_t[j]=0$.
The model outputs an $n \times n$ rate matrix, which is masked:

```python
valid_mask = x_t.unsqueeze(-1) * (1 - x_t).unsqueeze(-2)  # (batch, n, n)
predicted_rates = model_raw_output * valid_mask
```

This ensures zero rate for invalid swaps regardless of model output.

### Inference (generation)

```python
def generate_sample(model, n, k, beta, J, h, n_steps=100):
    x = uniform_sample(n, k)
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = step * dt
        rates = model(x, t, beta, J, h) / (1 - t)  # restore 1/(1-t)
        # Sample next swap from rates (tau-leaping or exact Gillespie)
        total_rate = rates.sum()
        if total_rate > 0:
            # Poisson number of events in dt
            n_events = rng.poisson(total_rate * dt)
            for _ in range(min(n_events, 1)):  # at most one swap per step
                probs = rates.flatten() / total_rate
                idx = rng.choice(len(probs), p=probs)
                i, j = idx // n, idx % n
                x[i], x[j] = 0, 1  # swap
    return x
```

Note: $x$ always has exactly $k$ ones because each swap preserves Hamming weight.
