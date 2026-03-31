# Diagnostic Task: Investigate Ex14b Performance Issues

## Context

Experiment 14b (synthetic EEG on real cortical mesh) shows the learned model
(TV ~0.68) performing worse than LASSO (TV ~0.58). On the cube experiments
(Ex13), the learned model dramatically beat LASSO. We suspect the diffusion
time range is too small, meaning the model doesn't get to leverage graph
structure.

## What to Check

### 1. Diffusion amount comparison

Load the cortical graph data from `ex14_eeg_data.npz` and compare the
effective diffusion to the cube experiments.

```python
# Load cortical data
data = np.load('ex14_eeg_data.npz', allow_pickle=True)
R_cortical = data['R']
A = data['A']

# Load/create cube data for comparison
from graph_ot_fm import make_cube_graph
R_cube = make_cube_graph(5)

# For both graphs:
# a) Compute eigenvalues of R (spectral gap = smallest nonzero |eigenvalue|)
# b) Create a single-peak source, diffuse at several tau values
# c) Compute TV between source and diffused distribution
# d) Print: at what tau does TV(source, diffused) = 0.3? 0.5?

# For cube: check tau in [0.5, 2.0] (our training range)
# For cortical: check tau in [0.05, 0.2] (current training range)
# Also check tau in [0.5, 2.0] on the cortical graph

# The key question: at tau=0.1 on the cortical graph, how much does
# the distribution actually change? If TV(source, diffused) < 0.1,
# the diffusion is negligible and LASSO can just invert the linear map.
```

### 2. Spectral properties

```python
# For both graphs, compute and print:
# - Number of nodes, edges
# - Eigenvalues of R (sorted by magnitude)
# - Spectral gap (smallest nonzero eigenvalue magnitude)
# - Diameter (max shortest path distance)
# - Mean degree

# The spectral gap determines how fast diffusion mixes.
# If the cortical graph has a much smaller spectral gap than the cube,
# the same tau gives much less diffusion.
```

### 3. Leadfield condition number

```python
# Check the leadfield matrix properties:
# - Shape (should be 64 x 100)
# - Rank
# - Condition number
# - Singular value distribution (plot)
# - Column norms (are some parcels much harder to observe than others?)

# A well-conditioned leadfield means LASSO can invert it easily.
# A poorly-conditioned one means the inverse problem is harder (favors us).
```

### 4. Visualize diffusion at different tau values

```python
# Pick 3 parcels: one superficial (gyral), one deep (sulcal), one medial
# For each:
#   Create a single-peak source at that parcel
#   Diffuse at tau = 0.05, 0.1, 0.2, 0.5, 1.0, 2.0
#   Plot TV(source, diffused) vs tau
#   Also plot the EEG pattern (A @ diffused) and how it changes with tau
#   
# Save as: ex14_diffusion_diagnostics.png
```

### 5. Baseline performance vs tau

```python
# Generate 50 test cases at each of several tau values:
# tau = 0.05, 0.1, 0.2, 0.5, 1.0, 2.0
# Run LASSO, MNE, sLORETA on each
# Plot baseline TV vs tau
#
# Key question: at what tau does LASSO start to struggle?
# That's the regime where our model should have an advantage.
#
# Save as: ex14_baseline_vs_tau.png
```

### 6. Recommend new tau range

```python
# Based on the above analysis, print a recommendation:
# "On the cube, tau=1.0 gives TV(source, diffused) = X.XX"
# "On the cortical graph, equivalent diffusion requires tau = Y.YY"
# "Recommended training range for cortical graph: [A, B]"
# "At this range, LASSO TV = X.XX (weaker, giving room for learned model)"
```

## Output

Print all diagnostics to console. Save two figures:
1. `ex14_diffusion_diagnostics.png` — diffusion amount comparison
2. `ex14_baseline_vs_tau.png` — baseline performance vs diffusion time

## Files needed

- `ex14_eeg_data.npz` (from Ex14a)
- `graph_ot_fm` package (for make_cube_graph, etc.)
- Standard scipy, numpy, matplotlib
