# Spec: Kawasaki Ising on 2D Lattice (Ex21)

## Overview

Learn to sample from the Ising model on a 2D lattice with fixed
magnetization (Kawasaki dynamics). Configurations are binary grids —
visually interpretable as black/white pixel images. The experiment
demonstrates:

1. Constraint enforcement (fixed magnetization) by construction
2. Locality — swaps restricted to neighboring lattice sites
3. Visually striking phase transition from disorder to ordered domains
4. The unified ConfigurationSpace framework applied to a new problem

## Physics

### Ising model on a square lattice

$N = L \times L$ sites on a 2D square lattice with periodic boundary
conditions (torus). Each site has a spin $\sigma_i \in \{0, 1\}$ (equivalently
$\{-1, +1\}$ with $\sigma = (s+1)/2$).

Energy:
$$E(\sigma) = -J \sum_{\langle i,j \rangle} (2\sigma_i - 1)(2\sigma_j - 1)$$

where the sum is over nearest-neighbor pairs on the lattice, and $J > 0$
is the ferromagnetic coupling (uniform — all edges equal).

Target distribution:
$$p(\sigma) \propto \exp(-\beta E(\sigma)), \quad \sum_i \sigma_i = k$$

The constraint $\sum \sigma_i = k$ fixes the magnetization. At $k = N/2$
(zero total magnetization), the system is symmetric and the phase transition
is cleanest.

### Kawasaki dynamics

Transitions: swap spins at neighboring lattice sites. Pick edge $(i,j)$
where $\sigma_i = 1, \sigma_j = 0$ (or vice versa), exchange them. This
preserves $\sum \sigma_i$ exactly.

Key difference from Johnson: swaps are restricted to lattice neighbors,
not all pairs. This is physically motivated — particles can only hop to
adjacent empty sites. The GNN message passing graph IS the lattice.

### Phase transition

The 2D Ising model has a critical temperature $\beta_c = \frac{\ln(1+\sqrt{2})}{2J}
\approx 0.4407/J$. For $J=1$:

- $\beta < \beta_c$: disordered, random-looking configurations
- $\beta \approx \beta_c$: critical, fractal domain boundaries, long-range correlations
- $\beta > \beta_c$: ordered, large aligned domains separated by smooth walls

At fixed zero magnetization ($k = N/2$), the ordered phase has two large
domains (roughly half black, half white) with minimal boundary length.

## Configuration space

### KawasakiSpace (subclass of ConfigurationSpace)

```python
class KawasakiSpace(ConfigurationSpace):
    """Kawasaki dynamics on a 2D square lattice.
    
    Position graph: L x L grid with periodic boundary conditions
    Vocabulary: {0, 1}
    Invariant: sum(labels) = k (fixed magnetization)
    Transitions: swap neighboring sites with different labels (k=2)
    """
    
    def __init__(self, L, k=None, J_coupling=1.0):
        self.L = L
        self.N = L * L
        self.k = k if k is not None else self.N // 2
        self.J_coupling = J_coupling
    
    @property
    def n_positions(self):
        return self.N
    
    @property
    def vocab_size(self):
        return 2
    
    @property
    def transition_order(self):
        return 2
    
    def position_graph_edges(self):
        """2D periodic grid. Each site has 4 neighbors."""
        src, dst = [], []
        for y in range(self.L):
            for x in range(self.L):
                i = y * self.L + x
                # Right neighbor
                j = y * self.L + (x + 1) % self.L
                src.extend([i, j])
                dst.extend([j, i])
                # Down neighbor
                j = ((y + 1) % self.L) * self.L + x
                src.extend([i, j])
                dst.extend([j, i])
        return np.array([src, dst])
    
    def position_edge_features(self):
        """Uniform coupling J on all edges."""
        n_edges = self.position_graph_edges().shape[1]
        return np.full((n_edges, 1), self.J_coupling, dtype=np.float32)
    
    def node_features(self, config):
        """[sigma_i] per node. Could add positional encoding."""
        return config[:, None].astype(np.float32)
    
    def global_features(self, t=0.0, beta=1.0, **kwargs):
        return np.array([t, beta], dtype=np.float32)
    
    def transition_mask(self, config):
        """Sparse mask: only neighboring pairs with different labels.
        
        Returns (N, N) sparse-compatible array, but only lattice
        neighbors can be nonzero.
        """
        edges = self.position_graph_edges()
        mask = np.zeros((self.N, self.N), dtype=np.float32)
        for idx in range(edges.shape[1]):
            i, j = edges[0, idx], edges[1, idx]
            if config[i] == 1 and config[j] == 0:
                mask[i, j] = 1.0
        return mask
    
    def apply_transition(self, config, transition_idx):
        i = transition_idx // self.N
        j = transition_idx % self.N
        if config[i] == 1 and config[j] == 0:
            new_config = config.copy()
            new_config[i] = 0
            new_config[j] = 1
            return new_config
        return None
    
    def geodesic_distance(self, config_a, config_b):
        # Minimum number of Kawasaki swaps.
        # This is the optimal transport distance on the lattice:
        # match each 1-in-a-not-in-b with a 0-in-a-not-in-b
        # minimizing total lattice distance.
        # For now, use a simple upper bound or exact OT.
        S_plus = np.where((config_a == 1) & (config_b == 0))[0]
        S_minus = np.where((config_a == 0) & (config_b == 1))[0]
        if len(S_plus) == 0:
            return 0
        # Exact: solve assignment problem on lattice distances
        from scipy.optimize import linear_sum_assignment
        cost = np.zeros((len(S_plus), len(S_minus)))
        for ii, s in enumerate(S_plus):
            for jj, t in enumerate(S_minus):
                # Manhattan distance on torus
                sy, sx = s // self.L, s % self.L
                ty, tx = t // self.L, t % self.L
                dx = min(abs(sx - tx), self.L - abs(sx - tx))
                dy = min(abs(sy - ty), self.L - abs(sy - ty))
                cost[ii, jj] = dx + dy
        row_ind, col_ind = linear_sum_assignment(cost)
        return int(cost[row_ind, col_ind].sum())
    
    def sample_intermediate(self, config_0, config_T, t, rng):
        """Sample intermediate configuration along exact lattice geodesic.
        
        1. Solve optimal assignment between S_plus and S_minus positions
           on the torus (Hungarian algorithm, O(d^3), negligible cost).
        2. For each matched pair, compute the lattice path (Manhattan
           path on the torus).
        3. For each particle, sample how many steps completed along its
           path via Bin(path_length_i, t).
        4. Place each particle at the intermediate position along its path.
        
        The resulting configuration always has exactly k ones (the
        invariant is preserved) because each particle is at some
        position along its path, and paths don't collide (the optimal
        assignment ensures this generically).
        """
        S_plus = np.where((config_0 == 1) & (config_T == 0))[0]
        S_minus = np.where((config_0 == 0) & (config_T == 1))[0]
        d = len(S_plus)
        
        if d == 0:
            return config_0.copy(), 0, 0
        
        # Step 1: Optimal assignment on torus Manhattan distance
        from scipy.optimize import linear_sum_assignment
        cost = np.zeros((d, d))
        for ii, s in enumerate(S_plus):
            for jj, tgt in enumerate(S_minus):
                sy, sx = s // self.L, s % self.L
                ty, tx = tgt // self.L, tgt % self.L
                dx = min(abs(sx - tx), self.L - abs(sx - tx))
                dy = min(abs(sy - ty), self.L - abs(sy - ty))
                cost[ii, jj] = dx + dy
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Step 2-3: For each matched pair, compute path and sample
        # intermediate position
        config_t = config_0.copy()
        total_completed = 0
        total_remaining = 0
        
        # First, clear all S_plus positions (particles will be placed
        # at intermediate positions)
        config_t[S_plus] = 0
        
        # Track occupied positions to handle collisions
        occupied = set(np.where(config_t == 1)[0].tolist())
        
        for ii, jj in zip(row_ind, col_ind):
            src = S_plus[ii]
            tgt = S_minus[jj]
            path_len = int(cost[ii, jj])
            
            if path_len == 0:
                # Already at target
                config_t[tgt] = 1
                occupied.add(tgt)
                total_completed += 0
                continue
            
            # Sample how many steps completed along this path
            steps_done = rng.binomial(path_len, t)
            
            if steps_done >= path_len:
                # Particle reached target
                config_t[tgt] = 1
                occupied.add(tgt)
                total_completed += path_len
            else:
                # Place particle at intermediate position along
                # Manhattan path on torus
                sy, sx = src // self.L, src % self.L
                ty, tx = tgt // self.L, tgt % self.L
                
                # Compute direction (shortest path on torus)
                dx_raw = tx - sx
                dy_raw = ty - sy
                # Wrap for torus
                if abs(dx_raw) > self.L // 2:
                    dx_raw = dx_raw - self.L * np.sign(dx_raw)
                if abs(dy_raw) > self.L // 2:
                    dy_raw = dy_raw - self.L * np.sign(dy_raw)
                
                dx_sign = int(np.sign(dx_raw)) if dx_raw != 0 else 0
                dy_sign = int(np.sign(dy_raw)) if dy_raw != 0 else 0
                abs_dx = abs(int(dx_raw))
                abs_dy = abs(int(dy_raw))
                
                # Take steps_done steps: first horizontal, then vertical
                # (arbitrary but consistent canonical ordering)
                cx, cy = sx, sy
                steps_left = steps_done
                
                # Horizontal steps
                h_steps = min(steps_left, abs_dx)
                cx = (cx + h_steps * dx_sign) % self.L
                steps_left -= h_steps
                
                # Vertical steps
                v_steps = min(steps_left, abs_dy)
                cy = (cy + v_steps * dy_sign) % self.L
                
                pos = cy * self.L + cx
                
                # Handle collision: if position occupied, try
                # nearby positions (fallback, should be rare with
                # optimal assignment)
                if pos in occupied:
                    # Fallback: place at source (no progress)
                    pos = src
                
                config_t[pos] = 1
                occupied.add(pos)
                total_completed += steps_done
                total_remaining += path_len - steps_done
        
        # Also place particles that don't move (in both source and target)
        # These are already handled by config_t = config_0.copy() above
        # since we only cleared S_plus positions.
        
        return config_t, total_completed, total_remaining
    
    def compute_target_rates(self, config_0, config_T, config_t, t):
        S_plus_rem = np.where((config_t == 1) & (config_T == 0))[0]
        S_minus_rem = np.where((config_t == 0) & (config_T == 1))[0]
        d_rem = len(S_plus_rem)
        
        rates = np.zeros((self.N, self.N), dtype=np.float32)
        if d_rem > 0:
            # Only neighbor pairs that are geodesic-progressing
            edges = self.position_graph_edges()
            for idx in range(edges.shape[1]):
                i, j = edges[0, idx], edges[1, idx]
                if i in S_plus_rem and j in S_minus_rem:
                    rates[i, j] = 1.0 / d_rem
        
        return rates
    
    def sample_source(self, rng):
        config = np.zeros(self.N, dtype=np.float32)
        ones = rng.choice(self.N, size=self.k, replace=False)
        config[ones] = 1.0
        return config
    
    def sample_target(self, rng, beta=1.0, mcmc_pool=None, **kwargs):
        if mcmc_pool is not None:
            idx = rng.integers(len(mcmc_pool))
            return mcmc_pool[idx].copy()
        else:
            from config_fm.spaces.kawasaki_mcmc import kawasaki_mcmc
            return kawasaki_mcmc(self, beta, 10000, rng)
```

### Geodesics on the lattice

The Kawasaki geodesic requires solving an optimal transport problem on the
lattice: match particles in $S_+$ to target positions in $S_-$ minimizing
total Manhattan distance on the torus. This is a $d \times d$ assignment
problem solved by the Hungarian algorithm in $O(d^3)$.

For $L=20$ with $d \approx 100-150$ mismatched positions, this costs
$O(150^3) \approx 3 \times 10^6$ operations — negligible compared to a
GNN forward pass (sub-millisecond). So we use exact lattice geodesics from
the start.

Given the optimal assignment, each particle has a specific lattice path to
its target. The intermediate configuration at flow time $t$ is computed by
advancing each particle independently: sample $\text{Bin}(\text{path\_length}_i, t)$
steps along its path. The canonical path ordering is horizontal-first,
then vertical (arbitrary but consistent).

One subtlety: two particles' paths might collide (both trying to occupy the
same intermediate position). With optimal assignment this is rare, and the
implementation falls back to placing the colliding particle at its source
position (no progress). A more sophisticated implementation could use
alternative path orderings or perturb assignments to avoid collisions.

## Problem sizes

**Small (for development):** L=10, N=100, k=50. Manageable, fast training.

**Medium (for paper):** L=20, N=400, k=200. Visually striking, shows
clear domain structure at high β.

**Large (stretch goal):** L=32, N=1024, k=512. Tests scaling.

## Beta values

Use $J = 1.0$, so $\beta_c \approx 0.4407$.

- $\beta = 0.2$ (high temperature, disordered)
- $\beta = 0.44$ (near critical, interesting correlations)
- $\beta = 0.8$ (low temperature, large domains)
- $\beta = 1.5$ (deep in ordered phase, very large domains)

## MCMC for target samples

Standard Kawasaki MCMC on the lattice:

```python
def kawasaki_mcmc(space, beta, n_steps, rng):
    config = space.sample_source(rng)
    edges = space.position_graph_edges()
    n_edges = edges.shape[1]
    
    for _ in range(n_steps):
        # Pick random edge
        e = rng.integers(n_edges)
        i, j = edges[0, e], edges[1, e]
        
        # Only propose if different labels
        if config[i] == config[j]:
            continue
        
        # Compute energy change
        dE = compute_kawasaki_dE(config, i, j, space)
        
        # Metropolis accept/reject
        if dE < 0 or rng.uniform() < np.exp(-beta * dE):
            config[i], config[j] = config[j], config[i]
    
    return config

def compute_kawasaki_dE(config, i, j, space):
    """Energy change from swapping sites i and j.
    
    Only need to recompute interactions involving i and j.
    O(degree) computation, not O(N).
    """
    L = space.L
    J = space.J_coupling
    
    si = 2 * config[i] - 1  # convert to +/- 1
    sj = 2 * config[j] - 1
    
    # Neighbors of i (excluding j) and neighbors of j (excluding i)
    nbrs_i = get_neighbors(i, L)
    nbrs_j = get_neighbors(j, L)
    
    dE = 0.0
    for nb in nbrs_i:
        if nb != j:
            s_nb = 2 * config[nb] - 1
            dE += -J * (sj - si) * s_nb  # i changes from si to sj
    for nb in nbrs_j:
        if nb != i:
            s_nb = 2 * config[nb] - 1
            dE += -J * (si - sj) * s_nb  # j changes from sj to si
    
    return dE
```

MCMC chain length: at least 10 * N sweeps (one sweep = N proposed swaps)
for equilibration. Near critical temperature, much longer chains needed
(critical slowing down). Monitor energy autocorrelation.

Pool size: 10,000 samples per beta value.

## Baselines

### 1. DFM on {0,1}^N (unconstrained)

Standard DFM treating each site independently. Position graph is still the
lattice (for message passing), but transitions are independent per-site
flips. No magnetization constraint.

Use `DFMSpace` from the framework with the lattice as position graph.

Metrics: validity (fraction with sum = k), same as Ex20.

### 2. DFM + rejection

Filter DFM samples to those with correct magnetization.

### 3. MCMC at various budgets

Kawasaki MCMC at 100, 500, 1000, 5000, 10000 sweeps (one sweep = N
proposed moves). Report quality at each budget.

## Evaluation metrics

### Visual metrics (the main selling point)

**Sample gallery:** 4 rows (FM, DFM, MCMC, True) × 8 columns (random
samples), displayed as L×L binary grids. Black = 1, white = 0. This is
the hero figure.

**Phase transition montage:** One row per β value, 4 samples per row.
Shows the visual progression from disorder to large domains.

**Domain size distribution:** Histogram of connected component sizes
in generated vs true samples. At high β, true samples have a few large
domains. If FM captures this, the histograms match.

**Structure factor / Fourier spectrum:** Compute 2D FFT of configurations,
average power spectrum over samples. This captures spatial correlations at
all length scales. At criticality, the power spectrum should show power-law
decay — a stringent test of whether FM captures critical fluctuations.

### Quantitative metrics

**Energy statistics:** mean, std, bias, KS test — same as Ex20.

**Pairwise correlation RMSE:** same as Ex20. On a lattice, the correlation
function $C(r) = \langle \sigma_i \sigma_{i+r} \rangle - \langle \sigma_i
\rangle^2$ as a function of distance $r$ is particularly informative.

**Correlation length:** Fit exponential decay to $C(r)$ at high temperature.
At criticality, $C(r)$ should decay as a power law. This tests whether FM
captures the correct correlation structure.

**Validity:** fraction with correct magnetization (100% for FM and MCMC,
variable for DFM).

**TV distance:** Only feasible for L ≤ 4 or so (too few states). Skip for
larger lattices.

## Output

### Main figure: `experiments/ex21_kawasaki_ising.png` (2×3 panels)

**Panel A: Sample gallery at high β.**
4×8 grid: rows = FM, DFM (valid only), MCMC-10000, True. Columns = 8
random samples. Each sample shown as L×L binary image. This is the hero
panel — visually demonstrates that FM produces realistic domain structures.

**Panel B: Phase transition montage.**
4 rows (β = 0.2, 0.44, 0.8, 1.5) × 4 columns (FM samples). Shows the
full range from disorder to order. Add a column of true samples for
comparison.

**Panel C: Energy vs β.**
Lines for FM, DFM+reject, MCMC-1000, MCMC-10000, and exact (if available).
Shows energy matching across the phase transition.

**Panel D: Correlation function C(r).**
Log-linear plot of spatial correlations vs distance for FM, MCMC, and
exact at β = 0.8. This tests whether FM captures the correct correlation
length.

**Panel E: Domain size distribution.**
Histogram of connected component sizes at β = 0.8 for FM vs true samples.

**Panel F: Validity and efficiency.**
Bar chart showing DFM validity rate at different lattice sizes L.
FM is always 100%.

### Console output

Same format as Ex20: method × beta table with energy bias, KS, correlation
RMSE, validity.

## Implementation

### File: `config_fm/spaces/kawasaki.py`

The `KawasakiSpace` class as defined above.

### File: `config_fm/spaces/kawasaki_mcmc.py`

MCMC sampling utilities for generating target pools.

### File: `experiments/ex21_kawasaki_ising.py`

Thin experiment script:
1. Create `KawasakiSpace(L=20, k=200, J_coupling=1.0)`
2. Generate MCMC pools for each beta
3. Create `ConfigurationRatePredictor(transition_order=2, ...)`
4. Train via `train_configuration_fm()`
5. Generate samples via `generate_samples()`
6. Also train/evaluate DFM baseline
7. Run MCMC baselines at various budgets
8. Evaluate and plot

### Training parameters

```python
L = 20                    # lattice size
N = 400                   # total sites
k = 200                   # fixed magnetization (zero net)
J_coupling = 1.0          # ferromagnetic coupling
betas = [0.2, 0.44, 0.8, 1.5]

n_epochs = 2000
batch_size = 128          # smaller batch — configs are larger
hidden_dim = 128
n_layers = 6              # deeper for larger receptive field on lattice
lr = 5e-4
mcmc_pool_size = 10000
mcmc_chain_length = 20000  # 50 sweeps of N=400
n_eval_samples = 2000
n_gen_steps = 200         # more steps — lattice geodesics are longer
```

### Note on n_layers

For L=20, the lattice diameter is 20 (Manhattan distance corner to corner
on a torus is 10+10=20). With 6 GNN layers, the receptive field is 6 hops
— not enough to cover the full lattice. This means the model can only learn
local correlations directly; long-range correlations must emerge from the
flow dynamics (composing many local moves). This is physically natural —
Kawasaki dynamics IS local, and long-range order emerges from many local
swaps.

If this proves insufficient, options include:
- Increase n_layers to 8-10
- Add skip connections or attention layers for long-range communication
- Use a hierarchical / multi-scale GNN
