# Experiments 4 & 5: Generalization Test and Source Localization

## Experiment 4: `ex4_generalization.py` — Meta-Level Generalization

### Motivation

Experiment 3 trained and tested on distributions drawn from the same generating
process. For real applications, the network must generalize to distributions it
has never seen. This experiment tests whether the meta-level model has learned
the structure of the rate matrix field on the simplex, or just memorized specific
transport plans from training.

### Setup

- Same cycle graph with N=6 nodes, unweighted
- **Training source distributions (50 samples):** peaked distributions centered
  at nodes 0, 1, 2 only. Generate each by: pick one of {0, 1, 2}, place mass
  0.6 there, distribute 0.4 uniformly among the other 5 nodes, add Gaussian
  noise (std=0.03), clip to non-negative, renormalize.
- **Training target distributions (50 samples):** approximately uniform
  distributions (same as Experiment 3 — start from (1/6,...,1/6), add Gaussian
  noise std=0.05, clip, renormalize).
- **Test source distributions (30 samples):** peaked distributions centered at
  nodes 3, 4, 5 only. Same generation procedure as training sources but using
  the OTHER half of the cycle. These nodes were NEVER seen as peak locations
  during training.
- **Test target distributions (30 samples):** approximately uniform, generated
  identically to training targets (these are fine to overlap since the target
  class is the same).

### Procedure

1. Build the graph, compute cost matrix.
2. Generate training and test distribution sets as described above.
3. Build training dataset:
   a. Compute meta-cost matrix W(mu_s, nu_t) for all training source-target pairs.
   b. Solve meta-OT coupling on the training set.
   c. Pre-generate 5000 training samples (mu_tau, tau, R_target).
4. Train RateMatrixPredictor for 500 epochs (same hyperparameters as Experiment 3).
5. Evaluate on BOTH training-like and test distributions:
   a. For 5 training-like sources (peaked at nodes 0, 1, 2): run learned flow
      forward, compute exact flow, compare.
   b. For 5 test sources (peaked at nodes 3, 4, 5): run learned flow forward,
      compute exact flow, compare.
   
   For the exact flow on test pairs: pick the closest training target for each
   test source, compute the exact graph-level OT coupling and rate matrices for
   that pair, and use as ground truth.

### Plots (single figure, 2x2 grid of panels)

- **Panel A**: Training loss curve (same as Ex3).
- **Panel B**: Entropy H(p_t) along trajectories for 3 training-regime test cases
  (peaked at nodes 0, 1, 2). Solid = learned, dashed = exact. These should
  match well (sanity check).
- **Panel C**: Entropy H(p_t) along trajectories for 3 out-of-distribution test
  cases (peaked at nodes 3, 4, 5). Solid = learned, dashed = exact. The key
  question: do these still match?
- **Panel D**: Bar chart comparing TV distance to target at t=1 for all 6 test
  cases (3 in-distribution, 3 out-of-distribution), grouped and color-coded.
  This is the main result: if the OOD bars are comparable to the in-distribution
  bars, the model generalizes.

### Validation checks (print to console)

- Mean TV to target for in-distribution test cases
- Mean TV to target for out-of-distribution test cases
- Ratio of the two (ideally close to 1.0)
- For each test case: print source peak node, final TV to target, final entropy

### Expected outcome

On a cycle graph with N=6, the symmetry is high — nodes 0,1,2 and 3,4,5 are
structurally identical up to relabeling. So the network SHOULD generalize well
if it has learned the rate matrix as a function of the distribution shape rather
than memorizing node identities. If it fails, this points to the network encoding
positional information too rigidly, and suggests we need graph-equivariant
architectures.

---

## Experiment 5: `ex5_source_localization.py` — Inverse Problem via Backward Flow

### Motivation

The meta-level model learns a rate matrix field u_theta(mu, t) that can be
integrated forward (source -> target) or backward (target -> source). Source
localization runs the learned flow backward from an observed distribution to
recover the initial condition. This requires NO additional training — just
reverse-time integration of the already-trained model.

### Setup

- Cycle graph with N=6 nodes, unweighted
- Use the trained model from Experiment 3 (which learned to transport peaked
  distributions to approximately uniform ones)
- Ground truth source distributions: 10 distributions peaked at various nodes
  (same generating process as Ex3 training sources)
- For each ground truth source mu_0:
  1. Compute the exact forward flow to get the target mu_1
  2. Add observation noise: mu_1_noisy = mu_1 + N(0, sigma^2), clip, renormalize
  3. Run the learned flow BACKWARD from mu_1_noisy to recover mu_0_recovered
  4. Compare mu_0_recovered to the true mu_0

### Backward integration

```python
def backward_trajectory(model, mu_end, n_steps=200, device='cpu'):
    """
    Integrate the learned flow backward from t=1 to t=0.
    
    Uses Euler method in reverse:
        mu_{k-1} = mu_k - dt * mu_k @ R_theta(mu_k, t_k)
    
    Start at t = 0.999 (avoid singularity at t=1).
    Step backward with dt = 0.999 / n_steps.
    
    At each step:
        1. Evaluate R = model(mu_current, t_current)
        2. mu_next = mu_current - dt * mu_current @ R
        3. Clip to non-negative, renormalize
        4. t_next = t_current - dt
    
    Returns:
        times: np.ndarray (n_steps,) from ~1 to ~0
        trajectory: np.ndarray (n_steps, N) distributions along backward path
    """
```

### Noise levels

Run the experiment at three noise levels:
- sigma = 0.0 (no noise — pure test of backward integration)
- sigma = 0.02 (mild noise)
- sigma = 0.05 (moderate noise)

### Evaluation metrics

For each recovered source mu_0_recovered vs true mu_0:
- **TV distance**: total variation between recovered and true source
- **Peak recovery**: does argmax(mu_0_recovered) == argmax(mu_0)? (binary)
- **KL divergence**: KL(mu_0 || mu_0_recovered), where defined

### Plots (single figure, 2x2 grid of panels)

- **Panel A**: Example backward trajectories. For ONE test case at sigma=0,
  show bar plots of the distribution at t = 1.0, 0.75, 0.5, 0.25, 0.0
  (5 snapshots arranged left to right). The distribution should start
  near-uniform and progressively concentrate toward the source node.

- **Panel B**: Overlay of recovered source vs true source for 3 test cases
  at sigma=0. For each test case, show a grouped bar chart with two bars per
  node (true vs recovered). These should match closely.

- **Panel C**: TV distance between recovered and true source as a function of
  noise level. Box plot or scatter plot with 10 points per noise level (one per
  test case), three groups (sigma = 0, 0.02, 0.05) on the x-axis. Shows
  degradation with noise.

- **Panel D**: Peak recovery accuracy (fraction of 10 test cases where the
  correct peak node is identified) as a function of noise level. Bar chart
  with 3 bars. Even at moderate noise, we expect high accuracy since the
  framework has strong structural prior.

### Validation checks (print to console)

- For each noise level: mean TV, std TV, peak recovery accuracy
- For sigma=0: all TV distances should be small (< 0.1), peak recovery should
  be 100%
- Print the recovered vs true peak node for each test case

### Expected outcome

At zero noise, backward integration should recover the source almost exactly
(limited only by Euler discretization error and the meta-level model's
approximation quality — both small based on Ex3 results).

At moderate noise, the peak node should still be correctly identified in most
cases because the flow's structural prior (optimal transport on the graph)
provides strong regularization — it "knows" that sources should be peaked
distributions, so it naturally concentrates mass during backward integration
even from noisy observations.

### Important implementation note

The backward integration reuses the SAME trained model from Experiment 3.
Do NOT retrain. The point is that source localization is free — it's just
running the existing model in reverse. Load the model checkpoint from Ex3
(save it at the end of ex3_meta_level.py if not already saved).

To support this, ex3_meta_level.py should be modified to save:
- The trained model: torch.save(model.state_dict(), 'checkpoints/meta_model.pt')
- The graph structure: pickle or np.save the rate matrix
- The test distributions used in Ex3 (for consistency)

Add a 'checkpoints/' directory to the repo structure.

---

## Updated Repository Structure

Add these files:
```
graph-ot-fm/
├── ...
├── experiments/
│   ├── ...
│   ├── ex4_generalization.py
│   ├── ex5_source_localization.py
│   └── plotting.py           # Add backward_trajectory helper here or in meta_fm/sample.py
├── checkpoints/               # Created by ex3, used by ex5
│   └── .gitkeep
└── ...
```

## Dependencies

No new dependencies needed. Everything uses the existing stack (numpy, scipy,
torch, matplotlib, pot).
