# Diagnostic Task: Check Directed Graph Support

## Question

Does our graph infrastructure (GraphStructure, GeodesicCache,
compute_cost_matrix, compute_ot_coupling) correctly handle asymmetric
rate matrices where R_ab != R_ba?

## What to Check

### 1. GraphStructure: shortest path computation

Look at how `GraphStructure.__init__` computes distances.

```python
from graph_ot_fm import GraphStructure
import numpy as np

# Create a simple asymmetric graph: 3 nodes in a line
# a -> b has weight 1, b -> a has weight 0.1
# b -> c has weight 1, c -> b has weight 0.1
R = np.array([
    [-1.0, 1.0, 0.0],
    [0.1, -1.1, 1.0],
    [0.0, 0.1, -0.1],
])

gs = GraphStructure(R)
print("Distances:")
print(gs.distances)
# Expected for directed: d(0,2) = 2, d(2,0) = 2 (both have paths)
# But the WEIGHTS differ: 0->1->2 uses strong edges,
# 2->1->0 uses weak edges
# If distances are computed from adjacency (ignoring weights),
# d(a,b) = d(b,a) = shortest hop count, which IS symmetric even
# for directed graphs (since both directions exist).
# But if distances are computed from WEIGHTED shortest paths,
# they could differ.

# Check: does it use the adjacency (R > 0) or the weights?
print("\nGeodesic counts (R^d)_aj:")
print("N(0, target=2):", gs.geodesic_counts[0, 2] if hasattr(gs, 'geodesic_counts') else 'N/A')
print("N(2, target=0):", gs.geodesic_counts[2, 0] if hasattr(gs, 'geodesic_counts') else 'N/A')
# These SHOULD differ for asymmetric R because R^2 is asymmetric
```

### 2. Check what happens with R^d for asymmetric R

```python
# R^1 for our asymmetric matrix
R_off = R.copy()
np.fill_diagonal(R_off, 0)
print("\nR_off (off-diagonal):")
print(R_off)

# R^2
R2 = R_off @ R_off
print("\nR^2:")
print(R2)
# R^2[0,2] = R[0,1]*R[1,2] = 1.0*1.0 = 1.0
# R^2[2,0] = R[2,1]*R[1,0] = 0.1*0.1 = 0.01
# These are different! The geodesic count from 0->2 (1.0) is much
# larger than from 2->0 (0.01). This SHOULD affect the conditional
# rate matrices.
```

### 3. Check GraphStructure source code

Read the actual source code of GraphStructure to see how it computes:
- distances (does it use directed or undirected shortest paths?)
- geodesic counts (does it use R^d or adj^d?)
- neighbors closer to target (does it respect direction?)

```bash
# Find and read the GraphStructure class
grep -r "class GraphStructure" graph_ot_fm/
# Then read the file
```

### 4. Check compute_cost_matrix

```python
from graph_ot_fm import compute_cost_matrix
cost = compute_cost_matrix(gs)
print("\nCost matrix:")
print(cost)
# Is cost[0,2] == cost[2,0]? If so, asymmetry is lost.
```

### 5. Check OT coupling

```python
from graph_ot_fm.ot_solver import compute_ot_coupling

mu_0 = np.array([0.8, 0.1, 0.1])
mu_1 = np.array([0.1, 0.1, 0.8])

coupling = compute_ot_coupling(mu_0, mu_1, cost)
print("\nOT coupling (0->2 transport):")
print(coupling)

# Reverse direction
coupling_rev = compute_ot_coupling(mu_1, mu_0, cost)
print("\nOT coupling (2->0 transport):")
print(coupling_rev)
# If cost is symmetric, these should have the same total cost
# but possibly different structure. If cost is asymmetric,
# the costs should differ.
```

### 6. Check conditional rate matrix computation

```python
from graph_ot_fm.flow import marginal_rate_matrix_fast
from graph_ot_fm import GeodesicCache

geo_cache = GeodesicCache(gs)
geo_cache.precompute_for_coupling(coupling)

R_cond = marginal_rate_matrix_fast(geo_cache, coupling, 0.5)
print("\nMarginal rate matrix at t=0.5 (forward transport 0->2):")
print(R_cond)
# Is this rate matrix asymmetric? Does it reflect the directed
# graph structure?
```

### 7. Test with a clearly directed graph

```python
# Ring graph with one-way traffic
# 0 -> 1 -> 2 -> 0 (strong)
# 0 <- 1 <- 2 <- 0 (weak)
N = 4
R_ring = np.zeros((N, N))
for i in range(N):
    R_ring[i, (i+1) % N] = 2.0   # forward: strong
    R_ring[i, (i-1) % N] = 0.1   # backward: weak
np.fill_diagonal(R_ring, -R_ring.sum(axis=1))

gs_ring = GraphStructure(R_ring)
print("\n\nDirected ring graph:")
print("Distances:", gs_ring.distances)
cost_ring = compute_cost_matrix(gs_ring)
print("Cost matrix:")
print(cost_ring)
# Key question: is cost(0, 1) == cost(1, 0)?
# On this directed ring, going 0->1 follows the strong direction (1 hop)
# Going 1->0 can go backward (1 hop, weak) or forward (3 hops, strong)
# The costs SHOULD differ if direction is handled correctly.
```

## Output

Print all results to console. Summarize findings:
- Does GraphStructure use directed or undirected shortest paths?
- Does compute_cost_matrix produce asymmetric costs for asymmetric R?
- Are geodesic counts computed using the asymmetric R matrix?
- Does the flow matching training signal encode the asymmetry?

## Recommendation

Based on findings, recommend whether:
(a) The code already handles directed graphs correctly
(b) Specific functions need modification to handle direction
(c) The issue is fundamental and requires significant refactoring
