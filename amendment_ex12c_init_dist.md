# Amendment: Experiment 12c — Configurable Initial Distribution

## CLI Arguments

```python
parser.add_argument('--init-dist', type=str, default='dirichlet',
                    choices=['dirichlet', 'logistic-normal'],
                    help='Prior distribution over the simplex')
parser.add_argument('--dirichlet-alpha', type=float, default=10.0,
                    help='Dirichlet concentration parameter (only for dirichlet)')
parser.add_argument('--logistic-normal-beta', type=float, default=1.0,
                    help='Scale of Laplacian covariance (only for logistic-normal)')
```

## Implementation

```python
def sample_init_distribution(N, rng, method='dirichlet', alpha=10.0,
                              beta=1.0, L_inv=None):
    """
    Sample a starting distribution from the chosen prior.
    
    Args:
        N: number of nodes
        rng: numpy random generator
        method: 'dirichlet' or 'logistic-normal'
        alpha: Dirichlet concentration (higher = closer to uniform)
        beta: logistic-normal scale (higher = more diverse)
        L_inv: precomputed pseudo-inverse of graph Laplacian (N, N),
               required for logistic-normal
    
    Returns: np.ndarray (N,) on the simplex
    """
    if method == 'dirichlet':
        return rng.dirichlet(np.full(N, alpha))
    
    elif method == 'logistic-normal':
        # Sample z ~ N(0, beta * L_inv) in R^N
        # Then softmax to get a point on the simplex
        #
        # L_inv is the pseudo-inverse of the graph Laplacian.
        # We use it as covariance so that nearby nodes on the graph
        # get correlated perturbations — spatially smooth starts.
        z = rng.multivariate_normal(np.zeros(N), beta * L_inv)
        # Softmax: shift for numerical stability
        z = z - z.max()
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()
    
    else:
        raise ValueError(f"Unknown init distribution: {method}")


def precompute_laplacian_pseudoinverse(R):
    """
    Compute the pseudo-inverse of the graph Laplacian L = -R.
    Used as covariance for the logistic-normal prior.
    
    The Laplacian has a zero eigenvalue (constant eigenvector).
    The pseudo-inverse inverts all other eigenvalues and maps the
    zero eigenvalue to zero.
    """
    L = -R.copy()
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    # Pseudo-inverse: invert nonzero eigenvalues
    pinv_eigenvalues = np.zeros_like(eigenvalues)
    threshold = 1e-10
    for i, lam in enumerate(eigenvalues):
        if abs(lam) > threshold:
            pinv_eigenvalues[i] = 1.0 / lam
    L_inv = eigenvectors @ np.diag(pinv_eigenvalues) @ eigenvectors.T
    return L_inv
```

## Usage in Dataset

```python
# Precompute L_inv once if using logistic-normal
L_inv = None
if args.init_dist == 'logistic-normal':
    L_inv = precompute_laplacian_pseudoinverse(R)

# In the dataset construction loop:
for _ in range(n_starts_per_pair):
    mu_start = sample_init_distribution(
        N, rng,
        method=args.init_dist,
        alpha=args.dirichlet_alpha,
        beta=args.logistic_normal_beta,
        L_inv=L_inv,
    )
    pi = compute_ot_coupling(mu_start, mu_source, cost)
    # ... rest as before
```

## Usage at Inference

```python
# Same function at inference — consistency between train and test
for k in range(K):
    mu_start = sample_init_distribution(
        N, rng,
        method=args.init_dist,
        alpha=args.dirichlet_alpha,
        beta=args.logistic_normal_beta,
        L_inv=L_inv,
    )
    _, traj = sample_trajectory_flexible(
        model, mu_start, context, edge_index,
        n_steps=n_steps, device=device)
    samples.append(traj[-1])
```

## Checkpoint Naming

Include the prior choice in the checkpoint filename:

```python
init_str = f'{args.init_dist}_a{args.dirichlet_alpha}' if args.init_dist == 'dirichlet' \
    else f'{args.init_dist}_b{args.logistic_normal_beta}'
ckpt_path = os.path.join(checkpoint_dir,
    f'meta_model_ex12c_{init_str}_{args.n_epochs}ep.pt')
```

## Comparison Runs

```bash
# Dirichlet with different concentrations
python ex12c_cube_posterior.py --init-dist dirichlet --dirichlet-alpha 1.0
python ex12c_cube_posterior.py --init-dist dirichlet --dirichlet-alpha 10.0
python ex12c_cube_posterior.py --init-dist dirichlet --dirichlet-alpha 50.0

# Logistic-normal with different scales
python ex12c_cube_posterior.py --init-dist logistic-normal --logistic-normal-beta 0.1
python ex12c_cube_posterior.py --init-dist logistic-normal --logistic-normal-beta 1.0
python ex12c_cube_posterior.py --init-dist logistic-normal --logistic-normal-beta 5.0
```

## What to Look For

**Dirichlet(1,...,1):** Most diverse starts. Risk: too diverse, model can't
converge them all. Posterior may be too broad. But if it works, the
uncertainty estimates are most meaningful.

**Dirichlet(10,...,10):** Mild perturbations of uniform. Safe default. Less
diverse posteriors but more accurate reconstruction per sample.

**Dirichlet(50,...,50):** Near-deterministic (almost uniform). Minimal
posterior diversity. Approaches the 12b point estimate. Useful as a
sanity check — should reproduce 12b results.

**Logistic-normal(beta=0.1):** Mild, spatially smooth perturbations.
Adjacent nodes get correlated starting values. The posterior diversity
should be spatially coherent — nearby nodes vary together, not independently.
This is the most principled choice for graph-structured problems.

**Logistic-normal(beta=1.0):** More diverse but still spatially smooth.
Good middle ground.

**Logistic-normal(beta=5.0):** Very diverse, but still smoother than
Dirichlet(1,...,1) due to the Laplacian correlation structure.

The logistic-normal is expected to produce more interpretable posterior
diversity because the starting points have spatial structure. Two samples
that start with mass concentrated in different regions of the cube will
flow to different reconstructions — and this geographic diversity maps
directly to uncertainty about source location.
