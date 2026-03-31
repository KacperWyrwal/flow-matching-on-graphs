"""
Device detection, EMA, and shared metric utilities.

Combines meta_fm/utils.py and graph_ot_fm/utils.py.
"""

import numpy as np
import torch


def get_device(auto: bool = True) -> torch.device:
    """
    Return a torch.device.

    Args:
        auto: if True (default), probe for CUDA then MPS and use the first available.
              if False, always return CPU.

    Prints which device was selected.
    """
    if auto:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                _ = torch.zeros(1, device='mps') + 1
                device = torch.device('mps')
                print("Using Apple MPS")
            except Exception:
                device = torch.device('cpu')
                print("MPS available but failed, falling back to CPU")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """Call after each optimizer.step()."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay)

    def apply(self, model):
        """Swap model weights with EMA weights (for evaluation)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original weights (after evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q), handling zeros gracefully."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # Only sum where p > 0
    mask = p > 1e-15
    q_safe = np.where(mask, np.maximum(q, 1e-15), 1.0)
    return float(np.sum(p[mask] * np.log(p[mask] / q_safe[mask])))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """TV distance = 0.5 * sum |p - q|."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(0.5 * np.sum(np.abs(p - q)))


# ── Graph constructor utilities (from graph_ot_fm/utils.py) ───────────────────

def make_cycle_graph(n: int, weighted: bool = False) -> np.ndarray:
    """
    Create rate matrix for cycle graph on n nodes.
    Node i connects to (i-1) % n and (i+1) % n.
    If weighted: R[i, (i+1)%n] = random weight > 0 (symmetric).
    If not weighted: all edge rates = 1.
    Returns: (N, N) rate matrix with diagonal = -sum of off-diag row.
    """
    R = np.zeros((n, n))
    rng = np.random.default_rng(42)

    for i in range(n):
        left = (i - 1) % n
        right = (i + 1) % n

        if weighted:
            # Generate symmetric weights
            w_right = rng.uniform(0.5, 2.0)
            R[i, right] = w_right
        else:
            R[i, right] = 1.0
            R[i, left] = 1.0

    if weighted:
        # Make symmetric: average the two directions
        R = (R + R.T) / 2.0
        # Ensure nonzero where edges exist
        for i in range(n):
            right = (i + 1) % n
            if R[i, right] == 0:
                R[i, right] = 0.5
                R[right, i] = 0.5

    # Set diagonal
    np.fill_diagonal(R, 0.0)
    np.fill_diagonal(R, -R.sum(axis=1))
    return R


def make_grid_graph(rows: int, cols: int, weighted: bool = False) -> np.ndarray:
    """
    Create rate matrix for 2D grid graph (rows x cols).
    Node (r,c) has index r*cols + c.
    Edges to 4-neighbors (up, down, left, right) within bounds.
    Returns: (N, N) rate matrix with diagonal = -sum of off-diag row.
    """
    N = rows * cols
    R = np.zeros((N, N))
    rng = np.random.default_rng(42)

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            neighbors = []
            if r > 0:
                neighbors.append((r - 1) * cols + c)
            if r < rows - 1:
                neighbors.append((r + 1) * cols + c)
            if c > 0:
                neighbors.append(r * cols + c - 1)
            if c < cols - 1:
                neighbors.append(r * cols + c + 1)

            for nbr in neighbors:
                if weighted:
                    w = rng.uniform(0.5, 2.0)
                    R[idx, nbr] = w
                else:
                    R[idx, nbr] = 1.0

    if weighted:
        # Make symmetric
        R = (R + R.T) / 2.0

    np.fill_diagonal(R, 0.0)
    np.fill_diagonal(R, -R.sum(axis=1))
    return R


def make_path_graph(n: int) -> np.ndarray:
    """Linear chain: 0-1-2-...(n-1). All edge rates = 1."""
    R = np.zeros((n, n))
    for i in range(n - 1):
        R[i, i + 1] = 1.0
        R[i + 1, i] = 1.0
    np.fill_diagonal(R, -R.sum(axis=1))
    return R


def make_star_graph(n: int) -> np.ndarray:
    """Central node 0 connected to nodes 1..n-1. All edge rates = 1."""
    R = np.zeros((n, n))
    for i in range(1, n):
        R[0, i] = 1.0
        R[i, 0] = 1.0
    np.fill_diagonal(R, -R.sum(axis=1))
    return R


def make_complete_bipartite_graph(n1: int, n2: int) -> np.ndarray:
    """K_{n1,n2}: every node in set A (0..n1-1) connected to every node in set B (n1..n1+n2-1)."""
    n = n1 + n2
    R = np.zeros((n, n))
    for i in range(n1):
        for j in range(n1, n):
            R[i, j] = 1.0
            R[j, i] = 1.0
    np.fill_diagonal(R, -R.sum(axis=1))
    return R


def make_barbell_graph(clique_size: int, path_length: int) -> np.ndarray:
    """
    Two complete graphs of clique_size connected by path_length intermediate nodes.
    Total nodes: 2 * clique_size + path_length.
    Node layout:
        Clique 1: 0 .. clique_size-1
        Path:     clique_size .. clique_size+path_length-1
        Clique 2: clique_size+path_length .. 2*clique_size+path_length-1
    The last node of clique 1 connects to the first path node, and the last
    path node connects to the first node of clique 2.
    """
    n = 2 * clique_size + path_length
    R = np.zeros((n, n))

    # First clique
    for i in range(clique_size):
        for j in range(clique_size):
            if i != j:
                R[i, j] = 1.0

    # Second clique
    c2 = clique_size + path_length
    for i in range(clique_size):
        for j in range(clique_size):
            if i != j:
                R[c2 + i, c2 + j] = 1.0

    # Bridge
    if path_length == 0:
        R[clique_size - 1, c2] = 1.0
        R[c2, clique_size - 1] = 1.0
    else:
        # clique1 exit -> path[0]
        R[clique_size - 1, clique_size] = 1.0
        R[clique_size, clique_size - 1] = 1.0
        # path edges
        for i in range(clique_size, clique_size + path_length - 1):
            R[i, i + 1] = 1.0
            R[i + 1, i] = 1.0
        # path[-1] -> clique2 entry
        R[clique_size + path_length - 1, c2] = 1.0
        R[c2, clique_size + path_length - 1] = 1.0

    np.fill_diagonal(R, 0.0)
    np.fill_diagonal(R, -R.sum(axis=1))
    return R


def make_cube_graph(size: int = 5) -> np.ndarray:
    """
    3D grid graph of size x size x size.
    Node (x,y,z) -> index x*size*size + y*size + z.
    Edges to 6-neighbors within bounds.
    Returns: (N, N) rate matrix with N = size^3.
    """
    N = size ** 3
    R = np.zeros((N, N))
    for x in range(size):
        for y in range(size):
            for z in range(size):
                i = x * size * size + y * size + z
                for dx, dy, dz in [
                    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
                ]:
                    nx_, ny_, nz_ = x + dx, y + dy, z + dz
                    if 0 <= nx_ < size and 0 <= ny_ < size and 0 <= nz_ < size:
                        j = nx_ * size * size + ny_ * size + nz_
                        R[i, j] = 1.0
    np.fill_diagonal(R, -R.sum(axis=1))
    return R


def cube_boundary_mask(size: int = 5) -> np.ndarray:
    """
    Returns binary mask: 1 for boundary nodes, 0 for interior.
    Boundary = any coordinate is 0 or (size-1).
    Interior = all coordinates in {1, ..., size-2}.
    """
    N = size ** 3
    mask = np.zeros(N)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                i = x * size * size + y * size + z
                if (x == 0 or x == size - 1 or
                        y == 0 or y == size - 1 or
                        z == 0 or z == size - 1):
                    mask[i] = 1.0
    return mask


def cube_node_depth(size: int = 5) -> np.ndarray:
    """
    Returns depth of each node: minimum distance to any boundary face.
    Boundary nodes have depth 0, next layer depth 1, center depth 2.
    For size=5: depths are 0, 1, or 2.
    """
    N = size ** 3
    depth = np.zeros(N)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                i = x * size * size + y * size + z
                d = min(x, size - 1 - x, y, size - 1 - y, z, size - 1 - z)
                depth[i] = d
    return depth


def make_petersen_graph() -> np.ndarray:
    """
    The Petersen graph (10 nodes, 15 edges, all degree 3).
    Outer pentagon: 0-1-2-3-4-0
    Inner pentagram: 5-7-9-6-8-5
    Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    """
    n = 10
    R = np.zeros((n, n))
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),   # outer pentagon
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),    # inner pentagram
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),    # spokes
    ]
    for i, j in edges:
        R[i, j] = 1.0
        R[j, i] = 1.0
    np.fill_diagonal(R, -R.sum(axis=1))
    return R
