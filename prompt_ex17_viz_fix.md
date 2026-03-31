# Fix: Ex17 Visualization and Path Baselines

## 1. Add path baselines to Panel D

Currently Panel D shows:
- FM: path TV at t=0, 0.25, 0.5, 0.75, 1.0 (curve with confidence band)
- DirectGNN: single horizontal line at endpoint TV
- Interp: single horizontal line at endpoint TV=0.0

Add two path baselines computed via linear interpolation in distribution space:

### Linear interp toward true target
```python
def linear_path_toward_target(mu_0, mu_1_true, t):
    """Naive straight-line interpolation: (1-t)*mu_0 + t*mu_1."""
    return (1 - t) * mu_0 + t * mu_1_true
```
This is given the correct endpoint but takes the wrong path.
At each t, compute TV from the exact OT interpolation.
Plot as a curve (e.g., green dashed line labeled "Linear (true target)").

### Linear interp toward DirectGNN prediction
```python
def linear_path_toward_prediction(mu_0, mu_1_pred, t):
    """Naive straight-line toward GNN-predicted endpoint."""
    return (1 - t) * mu_0 + t * mu_1_pred
```
Wrong path AND wrong endpoint.
Plot as a curve (e.g., orange dashed line labeled "Linear (DirectGNN)").

### Updated Panel D should show 3 curves:
- FM (solid blue): learned OT flow path
- Linear toward true target (dashed green): correct endpoint, wrong path
- Linear toward DirectGNN (dashed orange): wrong endpoint, wrong path

The key insight this reveals: even with the correct endpoint, linear
interpolation in distribution space does NOT match OT interpolation
on the graph. Our model learns the graph-aware transport path.

On a barbell graph, at t=0.5:
- OT interpolation: mass queues at the bottleneck
- Linear interpolation: mass is split 50/50 between source and target
  regions, with artificial mass appearing in the bridge
These look very different.

## 2. Fix transport gallery labeling

The current gallery has unclear row labels. Update to:

### Row labels (left side, large font)
For each test graph, show 3 rows clearly labeled:
- **"Exact OT"** — ground truth OT interpolation
- **"FM (Learned)"** — our model's trajectory
- **"DirectGNN"** — DirectGNN has no path, so show:
  Row of 5 panels: source at t=0, then the DirectGNN prediction
  repeated for t=0.25, 0.5, 0.75, 1.0 (since it only has one output)
  OR better: show the linear interpolation toward DirectGNN's
  predicted endpoint at each t.

### Column labels (top, for each time point)
- t=0.00, t=0.25, t=0.50, t=0.75, t=1.00

### Add separator lines
Add a thin horizontal line between each group of 3 rows (between
different test graphs) to make it clear which rows belong together.

### Add graph name label
On the left margin, label each group with the graph name
(e.g., "grid_5x6 (OOD-topo)", "rgg_100 (OOD-size)").

## 3. Increase node size

Current nodes are too small to see distribution values clearly.
Increase the node marker size:

```python
# Current (too small)
ax.scatter(pos[:, 0], pos[:, 1], c=mu, s=30, ...)

# Updated (larger, more visible)
ax.scatter(pos[:, 0], pos[:, 1], c=mu, s=80, cmap='hot_r',
           edgecolors='gray', linewidths=0.3, vmin=0, vmax=vmax,
           zorder=5)
```

For larger graphs (N>60), scale down slightly:
```python
node_size = max(30, min(80, 3000 / N))
```

## 4. Add edge drawing to gallery

Currently the gallery shows only nodes. Add faint edges to show
graph structure:

```python
# Draw edges faintly
for src, dst in zip(edge_index[0], edge_index[1]):
    ax.plot([pos[src, 0], pos[dst, 0]],
            [pos[src, 1], pos[dst, 1]],
            color='lightgray', lw=0.3, zorder=1)
```

This helps the viewer understand the topology and see how mass
flows along graph edges.

## 5. Fix blank panels at top of gallery

The current gallery has blank space at the top. Ensure the first
test graph's rows start at the top of the figure with proper
spacing.

## 6. Add colorbar

Add a shared colorbar to the gallery (or per-row colorbars) so
the viewer can read absolute distribution values, not just
relative colors.

## Summary of changes

1. Add linear interpolation path baselines to Panel D
2. Clear row labels in transport gallery (Exact OT / FM / DirectGNN)
3. Add graph name + split label per group
4. Add separator lines between test graphs
5. Increase node marker size (scale with graph size)
6. Add faint edge drawing
7. Fix blank panels
8. Add colorbar
