"""
Experiment 16: Heat Equation Flow Matching on Mesh Geometries.

Trains a FiLM-conditional GNN flow matching model on heat equation simulations
across 8 mesh geometries and tests topology generalization on 4 unseen geometries.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import expm
from scipy.spatial import Delaunay
from matplotlib.path import Path

from graph_ot_fm import (
    GraphStructure,
    GeodesicCache,
)
from graph_ot_fm.ot_solver import compute_ot_coupling
from graph_ot_fm.flow import marginal_distribution_fast, marginal_rate_matrix_fast
from graph_ot_fm import total_variation

from meta_fm import (
    FiLMConditionalGNNRateMatrixPredictor,
    DirectGNNPredictor,
    train_film_conditional,
    train_direct_gnn,
    get_device,
)
from meta_fm.model import rate_matrix_to_edge_index, RateMessagePassing
from meta_fm.sample import sample_posterior_film


# ── Constants ─────────────────────────────────────────────────────────────────

TRAIN_GEOS = ['square', 'circle', 'rectangle_2_1', 'rectangle_1_2',
              'triangle', 'pentagon', 'hexagon', 'ellipse']
TEST_GEOS  = ['L_shape', 'annulus', 'star', 'trapezoid']
IC_TYPES   = ['single_peak', 'multi_peak', 'gradient', 'smooth_random']


# ── Mesh Generation ───────────────────────────────────────────────────────────

def _boundary_points_square(n_boundary):
    """Generate boundary points for [0,1]^2."""
    n_per_side = max(n_boundary // 4, 3)
    pts = []
    t = np.linspace(0, 1, n_per_side, endpoint=False)
    pts.extend([(x, 0.0) for x in t])
    pts.extend([(1.0, y) for y in t])
    pts.extend([(1.0 - x, 1.0) for x in t])
    pts.extend([(0.0, 1.0 - y) for y in t])
    return np.array(pts)


def _boundary_points_circle(n_boundary, cx=0.5, cy=0.5, r=0.5):
    angles = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    pts = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1)
    return pts


def _boundary_points_polygon(vertices, n_boundary):
    """Distribute boundary points along a polygon perimeter."""
    verts = np.array(vertices)
    n_verts = len(verts)
    # Compute edge lengths for proportional distribution
    edges = [np.linalg.norm(verts[(i+1) % n_verts] - verts[i]) for i in range(n_verts)]
    total = sum(edges)
    pts = []
    for i in range(n_verts):
        n_edge = max(int(round(n_boundary * edges[i] / total)), 2)
        p0 = verts[i]
        p1 = verts[(i+1) % n_verts]
        for k in range(n_edge):
            t = k / n_edge
            pts.append(p0 + t * (p1 - p0))
    # Deduplicate approximately
    return np.array(pts[:n_boundary] if len(pts) >= n_boundary else pts)


def _inside_fn_for_shape(shape):
    """Return a function that tests if (x, y) is inside the shape."""
    if shape == 'square':
        return lambda x, y: (0 <= x <= 1) and (0 <= y <= 1)
    elif shape == 'circle':
        return lambda x, y: (x - 0.5)**2 + (y - 0.5)**2 <= 0.5**2
    elif shape == 'rectangle_2_1':
        return lambda x, y: (0 <= x <= 2) and (0 <= y <= 1)
    elif shape == 'rectangle_1_2':
        return lambda x, y: (0 <= x <= 1) and (0 <= y <= 2)
    elif shape == 'triangle':
        # Equilateral triangle (0,0),(1,0),(0.5, sqrt(3)/2)
        h = np.sqrt(3) / 2
        def _inside_tri(x, y):
            if y < 0 or y > h:
                return False
            # Check using barycentric
            v0 = np.array([0.5, h]) - np.array([0.0, 0.0])
            v1 = np.array([1.0, 0.0]) - np.array([0.0, 0.0])
            v2 = np.array([x, y]) - np.array([0.0, 0.0])
            dot00 = v0 @ v0; dot01 = v0 @ v1; dot02 = v0 @ v2
            dot11 = v1 @ v1; dot12 = v1 @ v2
            inv = 1.0 / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * inv
            v = (dot00 * dot12 - dot01 * dot02) * inv
            return (u >= 0) and (v >= 0) and (u + v <= 1)
        return _inside_tri
    elif shape == 'pentagon':
        n = 5
        angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, n, endpoint=False)
        verts = [(0.5 + 0.45*np.cos(a), 0.5 + 0.45*np.sin(a)) for a in angles]
        path = Path(verts + [verts[0]])
        return lambda x, y: path.contains_point((x, y))
    elif shape == 'hexagon':
        n = 6
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        verts = [(0.5 + 0.45*np.cos(a), 0.5 + 0.45*np.sin(a)) for a in angles]
        path = Path(verts + [verts[0]])
        return lambda x, y: path.contains_point((x, y))
    elif shape == 'ellipse':
        return lambda x, y: ((x - 0.5)/0.45)**2 + ((y - 0.5)/0.28)**2 <= 1.0
    elif shape == 'L_shape':
        verts = [(0,0),(1,0),(1,0.5),(0.5,0.5),(0.5,1),(0,1),(0,0)]
        path = Path(verts)
        return lambda x, y: path.contains_point((x, y))
    elif shape == 'annulus':
        return lambda x, y: (0.20**2 <= (x-0.5)**2 + (y-0.5)**2 <= 0.48**2)
    elif shape == 'star':
        n_points = 5
        outer_r = 0.40
        inner_r = 0.18
        cx, cy = 0.5, 0.5
        angles_outer = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, n_points, endpoint=False)
        angles_inner = angles_outer + np.pi / n_points
        verts = []
        for i in range(n_points):
            verts.append((cx + outer_r*np.cos(angles_outer[i]),
                          cy + outer_r*np.sin(angles_outer[i])))
            verts.append((cx + inner_r*np.cos(angles_inner[i]),
                          cy + inner_r*np.sin(angles_inner[i])))
        verts.append(verts[0])
        path = Path(verts)
        return lambda x, y: path.contains_point((x, y))
    elif shape == 'trapezoid':
        verts = [(0,0),(1,0),(0.75,0.6),(0.25,0.6),(0,0)]
        path = Path(verts)
        return lambda x, y: path.contains_point((x, y))
    else:
        raise ValueError(f"Unknown shape: {shape}")


def _get_bounds(shape):
    """Return (xmin, xmax, ymin, ymax) sampling bounds for shape."""
    if shape == 'rectangle_2_1':
        return 0, 2, 0, 1
    elif shape == 'rectangle_1_2':
        return 0, 1, 0, 2
    else:
        return 0, 1, 0, 1


def _boundary_pts_for_shape(shape, n_boundary):
    """Get boundary points for a shape."""
    if shape == 'square':
        return _boundary_points_square(n_boundary)
    elif shape == 'circle':
        return _boundary_points_circle(n_boundary)
    elif shape == 'rectangle_2_1':
        pts = []
        n = max(n_boundary // 6, 3)
        t = np.linspace(0, 1, n, endpoint=False)
        t2 = np.linspace(0, 2, 2*n, endpoint=False)
        pts.extend([(x, 0.0) for x in t2])
        pts.extend([(2.0, y) for y in t])
        pts.extend([(2.0 - x, 1.0) for x in t2])
        pts.extend([(0.0, 1.0 - y) for y in t])
        return np.array(pts[:n_boundary])
    elif shape == 'rectangle_1_2':
        pts = []
        n = max(n_boundary // 6, 3)
        t = np.linspace(0, 1, n, endpoint=False)
        t2 = np.linspace(0, 2, 2*n, endpoint=False)
        pts.extend([(x, 0.0) for x in t])
        pts.extend([(1.0, y) for y in t2])
        pts.extend([(1.0 - x, 2.0) for x in t])
        pts.extend([(0.0, 2.0 - y) for y in t2])
        return np.array(pts[:n_boundary])
    elif shape == 'triangle':
        h = np.sqrt(3) / 2
        verts = [(0, 0), (1, 0), (0.5, h)]
        return _boundary_points_polygon(verts, n_boundary)
    elif shape == 'pentagon':
        angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, 5, endpoint=False)
        verts = [(0.5 + 0.45*np.cos(a), 0.5 + 0.45*np.sin(a)) for a in angles]
        return _boundary_points_polygon(verts, n_boundary)
    elif shape == 'hexagon':
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        verts = [(0.5 + 0.45*np.cos(a), 0.5 + 0.45*np.sin(a)) for a in angles]
        return _boundary_points_polygon(verts, n_boundary)
    elif shape == 'ellipse':
        t = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
        pts = np.stack([0.5 + 0.45*np.cos(t), 0.5 + 0.28*np.sin(t)], axis=1)
        return pts
    elif shape == 'L_shape':
        verts = [(0,0),(1,0),(1,0.5),(0.5,0.5),(0.5,1),(0,1)]
        return _boundary_points_polygon(verts, n_boundary)
    elif shape == 'annulus':
        # Inner and outer boundary
        n_outer = int(n_boundary * 0.67)
        n_inner = n_boundary - n_outer
        outer = _boundary_points_circle(n_outer, 0.5, 0.5, 0.48)
        inner = _boundary_points_circle(n_inner, 0.5, 0.5, 0.20)
        return np.vstack([outer, inner])
    elif shape == 'star':
        n_points = 5
        outer_r = 0.40
        inner_r = 0.18
        cx, cy = 0.5, 0.5
        angles_outer = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, n_points, endpoint=False)
        angles_inner = angles_outer + np.pi / n_points
        verts = []
        for i in range(n_points):
            verts.append((cx + outer_r*np.cos(angles_outer[i]),
                          cy + outer_r*np.sin(angles_outer[i])))
            verts.append((cx + inner_r*np.cos(angles_inner[i]),
                          cy + inner_r*np.sin(angles_inner[i])))
        return _boundary_points_polygon(verts, n_boundary)
    elif shape == 'trapezoid':
        verts = [(0,0),(1,0),(0.75,0.6),(0.25,0.6)]
        return _boundary_points_polygon(verts, n_boundary)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def generate_mesh(shape, n_points=60, seed=42):
    """
    Generate a triangulated mesh for a given shape.

    Returns: (points, triangles, boundary_mask)
        points: (N, 2) array of node coordinates
        triangles: (T, 3) array of triangle vertex indices
        boundary_mask: (N,) bool array, True for boundary nodes
    """
    rng = np.random.default_rng(seed)
    inside_fn = _inside_fn_for_shape(shape)

    n_boundary = max(int(n_points * 0.35), 12)
    boundary_pts = _boundary_pts_for_shape(shape, n_boundary)
    n_boundary_actual = len(boundary_pts)

    n_interior = n_points - n_boundary_actual
    xmin, xmax, ymin, ymax = _get_bounds(shape)
    interior_pts = []
    max_tries = n_interior * 100
    tries = 0
    while len(interior_pts) < n_interior and tries < max_tries:
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        if inside_fn(x, y):
            interior_pts.append([x, y])
        tries += 1

    if len(interior_pts) < n_interior:
        # Fallback: use a grid inside the shape
        xs = np.linspace(xmin + 0.05, xmax - 0.05, 20)
        ys = np.linspace(ymin + 0.05, ymax - 0.05, 20)
        for x in xs:
            for y in ys:
                if inside_fn(x, y) and len(interior_pts) < n_interior:
                    interior_pts.append([x, y])

    all_pts = np.vstack([boundary_pts, np.array(interior_pts)])
    N = len(all_pts)
    boundary_mask = np.zeros(N, dtype=bool)
    boundary_mask[:n_boundary_actual] = True

    tri = Delaunay(all_pts)
    triangles = tri.simplices

    # Filter triangles for non-convex shapes by checking centroid is inside
    non_convex = {'L_shape', 'annulus', 'star'}
    if shape in non_convex:
        centroids = all_pts[triangles].mean(axis=1)
        keep = np.array([inside_fn(c[0], c[1]) for c in centroids])
        triangles = triangles[keep]

    return all_pts, triangles, boundary_mask


def mesh_to_graph(points, triangles):
    """
    Build rate matrix from triangulation.

    Edge weight = 1/distance for triangle edges.
    Diagonal = -row sums.

    Returns: R (N, N) rate matrix
    """
    N = len(points)
    R = np.zeros((N, N))

    for tri in triangles:
        for i in range(3):
            for j in range(3):
                if i != j:
                    a, b = tri[i], tri[j]
                    d = np.linalg.norm(points[a] - points[b])
                    if d > 1e-10:
                        R[a, b] = max(R[a, b], 1.0 / d)

    # Set diagonal
    np.fill_diagonal(R, 0.0)
    np.fill_diagonal(R, -R.sum(axis=1))
    return R


def _is_connected(R):
    """Check if the graph defined by R is connected."""
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    adj = (R > 0).astype(float)
    np.fill_diagonal(adj, 0)
    n_comp, _ = connected_components(csr_matrix(adj), directed=False)
    return n_comp == 1


# ── Initial Conditions ────────────────────────────────────────────────────────

def generate_initial_condition(N, points, rng, ic_type='random'):
    """
    Generate an initial probability distribution on N nodes.

    ic_type: 'single_peak', 'multi_peak', 'gradient', 'smooth_random', 'random'
    """
    if ic_type == 'random':
        ic_type = rng.choice(IC_TYPES)

    if ic_type == 'single_peak':
        mu = np.ones(N) * 0.01
        peak = rng.integers(N)
        mu[peak] += 1.0
        mu /= mu.sum()

    elif ic_type == 'multi_peak':
        n_peaks = rng.integers(2, 4)
        mu = np.zeros(N)
        peaks = rng.choice(N, size=n_peaks, replace=False)
        weights = rng.dirichlet(np.ones(n_peaks))
        for p, w in zip(peaks, weights):
            mu[p] = w
        mu += 1e-3
        mu /= mu.sum()

    elif ic_type == 'gradient':
        direction = rng.standard_normal(2)
        direction /= np.linalg.norm(direction) + 1e-10
        proj = points @ direction
        proj = proj - proj.min()
        proj = proj / (proj.max() + 1e-10)
        mu = proj + 0.05
        mu /= mu.sum()

    elif ic_type == 'smooth_random':
        n_bumps = rng.integers(2, 6)
        mu = np.zeros(N)
        for _ in range(n_bumps):
            center = rng.uniform(points.min(0), points.max(0))
            sigma = rng.uniform(0.05, 0.25)
            diffs = points - center[None, :]
            vals = np.exp(-0.5 * (diffs**2).sum(1) / sigma**2)
            mu += vals * rng.uniform(0.5, 2.0)
        mu += 1e-3
        mu /= mu.sum()

    else:
        raise ValueError(f"Unknown ic_type: {ic_type}")

    return mu.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class HeatMeshDataset(torch.utils.data.Dataset):
    """
    Dataset for heat equation flow matching on mesh geometries.

    Generates (mu_tau, tau, node_ctx, global_ctx, R_target, edge_index, N) tuples.
    """

    def __init__(self, geometries, n_meshes_per_geo=10, n_traj_per_mesh=5,
                 dt_range=(0.01, 0.1), n_samples=10000, n_points=60, seed=42):
        self.samples = []
        self.all_pairs = []  # for evaluation: list of dicts

        rng = np.random.default_rng(seed)
        all_items = []

        for geo in geometries:
            for mesh_idx in range(n_meshes_per_geo):
                mesh_seed = int(rng.integers(0, 100000))
                try:
                    points, triangles, boundary_mask = generate_mesh(geo, n_points, seed=mesh_seed)
                except Exception as e:
                    print(f"  Mesh gen failed for {geo} mesh {mesh_idx}: {e}", flush=True)
                    continue

                R = mesh_to_graph(points, triangles)

                if not _is_connected(R):
                    print(f"  Skipping disconnected graph: {geo} mesh {mesh_idx}", flush=True)
                    continue

                N = len(points)
                try:
                    graph_struct = GraphStructure(R)
                    geo_cache = GeodesicCache(graph_struct)
                    edge_index = rate_matrix_to_edge_index(R)
                except Exception as e:
                    print(f"  Graph struct failed for {geo} mesh {mesh_idx}: {e}", flush=True)
                    continue

                bnd = boundary_mask.astype(np.float32)  # (N,)

                for traj_idx in range(n_traj_per_mesh):
                    ic_type = IC_TYPES[int(rng.integers(len(IC_TYPES)))]
                    mu_init = generate_initial_condition(N, points, rng, ic_type)
                    dt_val = float(rng.uniform(dt_range[0], dt_range[1]))

                    try:
                        P = expm(dt_val * R)
                        mu_next = mu_init @ P
                        mu_next = np.clip(mu_next, 0, None)
                        mu_next /= mu_next.sum() + 1e-15

                        coupling = compute_ot_coupling(mu_init, mu_next, graph_struct=graph_struct)
                        geo_cache.precompute_for_coupling(coupling)
                    except Exception as e:
                        print(f"  OT failed for {geo} traj {traj_idx}: {e}", flush=True)
                        continue

                    # Store eval pair
                    self.all_pairs.append({
                        'geo': geo,
                        'N': N,
                        'points': points,
                        'triangles': triangles,
                        'boundary': bnd,
                        'R': R,
                        'edge_index': edge_index,
                        'mu_source': mu_init,
                        'mu_target': mu_next,
                        'dt': dt_val,
                        'graph_struct': graph_struct,
                        'geo_cache': geo_cache,
                        'coupling': coupling,
                    })

                    # Sample n_traj_per_mesh worth of training tuples from this trajectory
                    n_per_traj = max(1, n_samples // (len(geometries) * n_meshes_per_geo * n_traj_per_mesh))
                    for _ in range(n_per_traj):
                        tau = float(rng.uniform(0.0, 0.999))
                        try:
                            mu_tau = marginal_distribution_fast(geo_cache, coupling, tau)
                            R_target = marginal_rate_matrix_fast(geo_cache, coupling, tau)
                            # Model predicts u_tilde = (1-tau)*R
                            u_tilde = R_target * (1.0 - tau)
                        except Exception:
                            continue

                        node_ctx = np.stack([mu_init, bnd], axis=1).astype(np.float32)  # (N, 2)
                        global_ctx = np.array([dt_val], dtype=np.float32)  # (1,)

                        all_items.append((
                            torch.tensor(mu_tau, dtype=torch.float32),
                            torch.tensor([tau], dtype=torch.float32),
                            torch.tensor(node_ctx, dtype=torch.float32),
                            torch.tensor(global_ctx, dtype=torch.float32),
                            torch.tensor(u_tilde, dtype=torch.float32),
                            edge_index,
                            N,
                        ))

        # Shuffle and trim to n_samples
        idx = np.random.default_rng(seed + 1).permutation(len(all_items))
        n_actual = min(n_samples, len(all_items))
        self.samples = [all_items[i] for i in idx[:n_actual]]
        print(f"  Dataset: {len(self.samples)} training samples, {len(self.all_pairs)} eval pairs", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Baselines ─────────────────────────────────────────────────────────────────

class MeshGraphNetBaseline(nn.Module):
    """
    Simple GNN baseline that directly predicts next-step distribution.

    Input: [mu_i, is_boundary_i, dt_broadcast] → 3 dims per node
    Output: softmax distribution
    """

    def __init__(self, hidden_dim=64, n_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(3, hidden_dim)
        self.mp_layers = nn.ModuleList([
            RateMessagePassing(in_dim=hidden_dim, hidden_dim=hidden_dim)
            for _ in range(n_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feats, edge_index):
        """
        node_feats: (N, 3) — [mu, is_boundary, dt_broadcast]
        edge_index:  (2, E)
        Returns: (N,) softmax distribution
        """
        h = self.input_proj(node_feats)
        for mp in self.mp_layers:
            h = mp(h, edge_index)
        logits = self.readout(h).squeeze(-1)
        return torch.softmax(logits, dim=0)

    def predict(self, mu, dt_val, boundary, edge_index):
        """
        mu: (N,) tensor, dt_val: float, boundary: (N,) tensor, edge_index: (2, E) tensor
        Returns: (N,) numpy array
        """
        self.eval()
        with torch.no_grad():
            dt_broadcast = torch.full_like(mu, dt_val)
            node_feats = torch.stack([mu, boundary, dt_broadcast], dim=1)
            pred = self.forward(node_feats, edge_index)
        return pred.cpu().numpy()


def train_meshgraphnet(mgn, all_pairs, n_epochs=300, lr=5e-4, device=None):
    """Train MeshGraphNet baseline on all_pairs."""
    if device is None:
        device = get_device()
    mgn = mgn.to(device)
    optimizer = torch.optim.Adam(mgn.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        # Shuffle pairs each epoch
        idx = np.random.permutation(len(all_pairs))
        for i in idx:
            pair = all_pairs[i]
            mu_src = torch.tensor(pair['mu_source'], dtype=torch.float32, device=device)
            mu_tgt = torch.tensor(pair['mu_target'], dtype=torch.float32, device=device)
            bnd = torch.tensor(pair['boundary'], dtype=torch.float32, device=device)
            ei = pair['edge_index'].to(device)
            dt_val = pair['dt']

            dt_broadcast = torch.full_like(mu_src, dt_val)
            node_feats = torch.stack([mu_src, bnd, dt_broadcast], dim=1)
            mu_pred = mgn(node_feats, ei)

            loss = F.kl_div(torch.log(mu_pred + 1e-8), mu_tgt, reduction='sum')

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mgn.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(all_pairs), 1)
        losses.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            print(f"  MGN Epoch {epoch+1}/{n_epochs} | KL Loss: {avg_loss:.6f}", flush=True)

    return losses


def build_direct_gnn_pairs(all_pairs):
    """Convert all_pairs to list of (context_np, mu_target_np, edge_index) tuples."""
    pairs = []
    for p in all_pairs:
        N = p['N']
        mu_src = p['mu_source']
        bnd = p['boundary']
        dt_val = p['dt']
        dt_broadcast = np.full(N, dt_val, dtype=np.float32)
        context = np.stack([mu_src, bnd, dt_broadcast], axis=1)  # (N, 3)
        pairs.append((context, p['mu_target'], p['edge_index']))
    return pairs


# ── Test Case Generation ──────────────────────────────────────────────────────

def generate_test_cases(geometries, n_meshes=5, n_cases_per_mesh=5,
                        dt_range=(0.01, 0.1), n_points=60, seed=9000):
    """
    Generate test cases for evaluation.

    Returns list of dicts: {geo, N, points, triangles, boundary, R, edge_index,
                             mu_source, mu_target, dt, ic_type}
    """
    rng = np.random.default_rng(seed)
    test_cases = []

    for geo_idx, geo in enumerate(geometries):
        for mesh_idx in range(n_meshes):
            mesh_seed = seed + (geo_idx * n_meshes + mesh_idx) * 1000
            try:
                points, triangles, boundary_mask = generate_mesh(geo, n_points, seed=mesh_seed)
            except Exception as e:
                print(f"  Test mesh gen failed for {geo} mesh {mesh_idx}: {e}", flush=True)
                continue

            R = mesh_to_graph(points, triangles)
            if not _is_connected(R):
                continue

            N = len(points)
            edge_index = rate_matrix_to_edge_index(R)
            bnd = boundary_mask.astype(np.float32)

            for case_idx in range(n_cases_per_mesh):
                ic_type = IC_TYPES[int(rng.integers(len(IC_TYPES)))]
                mu_init = generate_initial_condition(N, points, rng, ic_type)
                dt_val = float(rng.uniform(dt_range[0], dt_range[1]))

                try:
                    P = expm(dt_val * R)
                    mu_next = mu_init @ P
                    mu_next = np.clip(mu_next, 0, None)
                    mu_next /= mu_next.sum() + 1e-15
                except Exception:
                    continue

                test_cases.append({
                    'geo': geo,
                    'N': N,
                    'points': points,
                    'triangles': triangles,
                    'boundary': bnd,
                    'R': R,
                    'edge_index': edge_index,
                    'mu_source': mu_init,
                    'mu_target': mu_next,
                    'dt': dt_val,
                    'ic_type': ic_type,
                })

    print(f"  Generated {len(test_cases)} test cases for {geometries}", flush=True)
    return test_cases


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_one_step(model, mgn, direct_gnn, test_cases, device, n_ode_steps=50):
    """
    Evaluate one-step prediction for all methods on test_cases.

    Returns dict with arrays of TV values per method.
    """
    model.eval()
    mgn.eval()
    direct_gnn.eval()

    tv_learned = []
    tv_mgn = []
    tv_direct = []
    tv_laplacian = []

    for tc in test_cases:
        N = tc['N']
        mu_src = tc['mu_source']
        mu_tgt = tc['mu_target']
        R = tc['R']
        dt = tc['dt']
        bnd = tc['boundary']
        ei = tc['edge_index']

        node_ctx = np.stack([mu_src, bnd], axis=1).astype(np.float32)  # (N, 2)
        global_ctx = np.array([dt], dtype=np.float32)  # (1,)

        # Learned model
        try:
            mu_pred_learned = sample_posterior_film(
                model, mu_src[None], node_ctx, global_ctx, ei,
                n_steps=n_ode_steps, device=device
            )[0]
            tv_learned.append(0.5 * np.abs(mu_pred_learned - mu_tgt).sum())
        except Exception as e:
            tv_learned.append(float('nan'))

        # MeshGraphNet
        try:
            mu_t = torch.tensor(mu_src, dtype=torch.float32, device=device)
            dt_t = dt
            bnd_t = torch.tensor(bnd, dtype=torch.float32, device=device)
            ei_d = ei.to(device)
            mu_pred_mgn = mgn.predict(mu_t, dt_t, bnd_t, ei_d)
            tv_mgn.append(0.5 * np.abs(mu_pred_mgn - mu_tgt).sum())
        except Exception:
            tv_mgn.append(float('nan'))

        # DirectGNN
        try:
            dt_broadcast = np.full(N, dt, dtype=np.float32)
            context = np.stack([mu_src, bnd, dt_broadcast], axis=1)
            ctx_t = torch.tensor(context, dtype=torch.float32, device=device)
            ei_d = ei.to(device)
            with torch.no_grad():
                mu_pred_direct = direct_gnn(ctx_t, ei_d).cpu().numpy()
            tv_direct.append(0.5 * np.abs(mu_pred_direct - mu_tgt).sum())
        except Exception:
            tv_direct.append(float('nan'))

        # Laplacian baseline: mu_next ≈ mu @ (I + dt*R), clip, normalize
        try:
            I = np.eye(N)
            mu_pred_lap = mu_src @ (I + dt * R)
            mu_pred_lap = np.clip(mu_pred_lap, 0, None)
            s = mu_pred_lap.sum()
            if s > 1e-15:
                mu_pred_lap /= s
            tv_laplacian.append(0.5 * np.abs(mu_pred_lap - mu_tgt).sum())
        except Exception:
            tv_laplacian.append(float('nan'))

    return {
        'learned': np.array(tv_learned),
        'mgn': np.array(tv_mgn),
        'direct_gnn': np.array(tv_direct),
        'laplacian': np.array(tv_laplacian),
    }


def eval_rollout(model, mgn, direct_gnn, test_case, K=20, device='cpu', n_ode_steps=30):
    """
    Evaluate long-horizon rollout for all methods on a single test_case.

    Returns dict of TV trajectory arrays, each of length K+1.
    """
    model.eval()
    mgn.eval()
    direct_gnn.eval()

    N = test_case['N']
    mu_init = test_case['mu_source']
    R = test_case['R']
    dt = test_case['dt']
    bnd = test_case['boundary']
    ei = test_case['edge_index']

    # Compute exact trajectory
    P = expm(dt * R)
    exact_traj = [mu_init.copy()]
    mu_exact = mu_init.copy()
    for _ in range(K):
        mu_exact = mu_exact @ P
        mu_exact = np.clip(mu_exact, 0, None)
        mu_exact /= mu_exact.sum() + 1e-15
        exact_traj.append(mu_exact.copy())

    # Helper: TV from exact
    def _tv(a, b):
        return 0.5 * np.abs(a - b).sum()

    # Learned model rollout
    learned_tv = [0.0]
    mu_current_learned = mu_init.copy()
    for step in range(K):
        node_ctx = np.stack([mu_current_learned, bnd], axis=1).astype(np.float32)
        global_ctx = np.array([dt], dtype=np.float32)
        try:
            mu_next_learned = sample_posterior_film(
                model, mu_current_learned[None], node_ctx, global_ctx, ei,
                n_steps=n_ode_steps, device=device
            )[0]
            mu_current_learned = mu_next_learned
        except Exception:
            pass
        learned_tv.append(_tv(mu_current_learned, exact_traj[step + 1]))

    # MGN rollout
    mgn_tv = [0.0]
    mu_current_mgn = mu_init.copy()
    for step in range(K):
        mu_t = torch.tensor(mu_current_mgn, dtype=torch.float32, device=device)
        bnd_t = torch.tensor(bnd, dtype=torch.float32, device=device)
        ei_d = ei.to(device)
        mu_next_mgn = mgn.predict(mu_t, dt, bnd_t, ei_d)
        mu_current_mgn = mu_next_mgn
        mgn_tv.append(_tv(mu_current_mgn, exact_traj[step + 1]))

    # DirectGNN rollout
    direct_tv = [0.0]
    mu_current_direct = mu_init.copy()
    for step in range(K):
        dt_broadcast = np.full(N, dt, dtype=np.float32)
        context = np.stack([mu_current_direct, bnd, dt_broadcast], axis=1)
        ctx_t = torch.tensor(context, dtype=torch.float32, device=device)
        ei_d = ei.to(device)
        with torch.no_grad():
            mu_next_direct = direct_gnn(ctx_t, ei_d).cpu().numpy()
        mu_current_direct = mu_next_direct
        direct_tv.append(_tv(mu_current_direct, exact_traj[step + 1]))

    # Laplacian rollout
    lap_tv = [0.0]
    I = np.eye(N)
    mu_current_lap = mu_init.copy()
    for step in range(K):
        mu_next_lap = mu_current_lap @ (I + dt * R)
        mu_next_lap = np.clip(mu_next_lap, 0, None)
        s = mu_next_lap.sum()
        if s > 1e-15:
            mu_next_lap /= s
        mu_current_lap = mu_next_lap
        lap_tv.append(_tv(mu_current_lap, exact_traj[step + 1]))

    return {
        'learned': np.array(learned_tv),
        'mgn': np.array(mgn_tv),
        'direct_gnn': np.array(direct_tv),
        'laplacian': np.array(lap_tv),
    }


# ── Results Printing ──────────────────────────────────────────────────────────

def print_results(id_results, ood_results, rollout_results):
    """Print experiment results in formatted table."""
    methods = ['learned', 'mgn', 'direct_gnn', 'laplacian']
    method_names = ['Learned (FM)', 'MeshGraphNet', 'DirectGNN', 'Laplacian']

    print("\n" + "="*60, flush=True)
    print("  EX16 Heat Mesh Results", flush=True)
    print("="*60, flush=True)
    print(f"\n{'Method':<20} {'ID TV':>10} {'OOD TV':>10} {'Gap':>10}", flush=True)
    print("-"*52, flush=True)

    for method, name in zip(methods, method_names):
        id_tv = id_results[method]
        ood_tv = ood_results[method]
        id_mean = float(np.nanmean(id_tv))
        ood_mean = float(np.nanmean(ood_tv))
        gap = ood_mean - id_mean
        print(f"  {name:<18} {id_mean:>10.4f} {ood_mean:>10.4f} {gap:>10.4f}", flush=True)

    print("\nLong-horizon Rollout (final TV at K steps):", flush=True)
    print("-"*52, flush=True)
    for method, name in zip(methods, method_names):
        final_tv = float(rollout_results[method][-1])
        print(f"  {name:<18} {final_tv:>10.4f}", flush=True)
    print("="*60 + "\n", flush=True)


# ── Plotting ──────────────────────────────────────────────────────────────────

def compute_rate_matrix_errors(film_model, eval_pairs, device,
                                taus=None, n_pairs=30):
    """
    For each tau, compute mean ||R_pred - R_true||_F / N² across pairs.
    R_true = (1-tau) * marginal_rate_matrix_fast(cache, pi, tau).
    R_pred = film_model.forward_batch(...).

    Returns: taus (array), means (array), stds (array)
    """
    if taus is None:
        taus = np.linspace(0.05, 0.95, 19)

    film_model.eval()
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(eval_pairs), size=min(n_pairs, len(eval_pairs)), replace=False)

    all_errors = []
    with torch.no_grad():
        for pidx in idxs:
            pair = eval_pairs[int(pidx)]
            N = pair['N']
            cache = pair['geo_cache']
            pi    = pair['coupling']
            bnd   = pair['boundary']
            dt    = float(pair['dt'])
            mu_init = pair['mu_source']
            node_ctx  = np.stack([mu_init, bnd], axis=1).astype(np.float32)  # (N, 2)
            global_ctx = np.array([dt], dtype=np.float32)                    # (1,)
            errs = []
            for tau in taus:
                tau_f = float(tau)
                mu_tau = marginal_distribution_fast(cache, pi, tau_f)
                R_true = (1.0 - tau_f) * marginal_rate_matrix_fast(cache, pi, tau_f)

                mu_t   = torch.tensor(mu_tau[None],     dtype=torch.float32, device=device)
                tau_t  = torch.tensor([[tau_f]],        dtype=torch.float32, device=device)
                nctx_t = torch.tensor(node_ctx[None],   dtype=torch.float32, device=device)
                gctx_t = torch.tensor(global_ctx[None], dtype=torch.float32, device=device)
                ei     = pair['edge_index'].to(device)

                R_pred = film_model.forward_batch(mu_t, tau_t, nctx_t, gctx_t, ei)[0]
                R_pred_np = R_pred.cpu().numpy()

                err = np.linalg.norm(R_pred_np - R_true, 'fro') / (N ** 2)
                errs.append(err)
            all_errors.append(errs)

    all_errors = np.array(all_errors)   # (n_pairs, n_taus)
    return np.array(taus), all_errors.mean(axis=0), all_errors.std(axis=0)


def plot_prediction_comparison(film_model, mgn, direct_gnn, test_case, device, save_path):
    """
    4-panel figure: Ground Truth / FM / MeshGraphNet / DirectGNN predictions.
    """
    tc       = test_case
    points   = tc['points']
    triangles = tc['triangles']
    boundary = tc['boundary'].astype(float)
    dt       = float(tc['dt'])
    mu_src   = tc['mu_source']
    mu_true  = tc['mu_target']
    ei       = tc['edge_index'].to(device)

    # FM prediction
    node_ctx   = np.stack([mu_src, boundary], axis=1).astype(np.float32)
    global_ctx = np.array([dt], dtype=np.float32)
    with torch.no_grad():
        mu_fm = sample_posterior_film(
            film_model, mu_src[None], node_ctx, global_ctx,
            tc['edge_index'], n_steps=50, device=device)[0]

    # MeshGraphNet
    with torch.no_grad():
        mu_t   = torch.tensor(mu_src,   dtype=torch.float32, device=device)
        bnd_t  = torch.tensor(boundary, dtype=torch.float32, device=device)
        mu_mgn = mgn.predict(mu_t, dt, bnd_t, ei)

    # DirectGNN
    with torch.no_grad():
        N = len(mu_src)
        ctx = np.stack([mu_src, boundary, np.full(N, dt)], axis=1).astype(np.float32)
        ctx_t  = torch.tensor(ctx, dtype=torch.float32, device=device)
        mu_dgnn = direct_gnn(ctx_t, ei).cpu().numpy()

    preds  = [mu_true, mu_fm, mu_mgn, mu_dgnn]
    titles = ['Ground Truth', 'FM (Learned)', 'MeshGraphNet', 'DirectGNN']
    vmin   = min(p.min() for p in preds)
    vmax   = max(p.max() for p in preds)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.suptitle(f'Prediction comparison — {tc["geo"]}  (dt={dt:.3f})', fontsize=10)
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    im = None
    for ax, pred, title in zip(axes, preds, titles):
        im = ax.tripcolor(triang, pred, shading='gouraud',
                          vmin=vmin, vmax=vmax, cmap='hot_r')
        ax.triplot(triang, color='grey', lw=0.25, alpha=0.35)
        ax.set_aspect('equal')
        ax.axis('off')
        tv = 0.5 * float(np.abs(pred - mu_true).sum())
        label = title if title == 'Ground Truth' else f'{title}\nTV={tv:.4f}'
        ax.set_title(label, fontsize=8)
    if im is not None:
        plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved prediction comparison: {save_path}", flush=True)


def plot_main_figure(train_losses, id_results, ood_results, rollout_results,
                     train_test_meshes, rate_errors, save_path):
    """
    Plot 2x3 figure summarizing experiment results.

    train_test_meshes: dict with 'train' and 'test' entries, each list of
        (geo_name, points, triangles) tuples
    rate_errors: (taus, means, stds) arrays from compute_rate_matrix_errors
    """
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    methods = ['learned', 'mgn', 'direct_gnn', 'laplacian']
    method_names = ['FM (Learned)', 'MeshGraphNet', 'DirectGNN', 'Laplacian']
    method_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # ── Panel A: Training loss ─────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.semilogy(train_losses, color='steelblue', lw=1.5)
    ax_a.set_xlabel('Epoch')
    ax_a.set_ylabel('Loss')
    ax_a.set_title('A  Training Loss')
    ax_a.grid(True, alpha=0.3)

    # ── Panel B: Mesh visualization (2x4 grid) ─────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_visible(False)
    inner_gs = gs[0, 1].subgridspec(2, 4, hspace=0.05, wspace=0.05)

    train_show = ['square', 'circle', 'triangle', 'hexagon']
    test_show = ['L_shape', 'annulus', 'star', 'trapezoid']
    all_show = train_show + test_show

    for col_idx, geo in enumerate(train_show):
        ax = fig.add_subplot(inner_gs[0, col_idx])
        if geo in train_test_meshes['train']:
            pts, tris = train_test_meshes['train'][geo]
            ax.triplot(pts[:, 0], pts[:, 1], tris, lw=0.3, color='navy')
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(geo.replace('_', '\n'), fontsize=5, pad=1)

    for col_idx, geo in enumerate(test_show):
        ax = fig.add_subplot(inner_gs[1, col_idx])
        if geo in train_test_meshes['test']:
            pts, tris = train_test_meshes['test'][geo]
            ax.triplot(pts[:, 0], pts[:, 1], tris, lw=0.3, color='darkred')
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(geo.replace('_', '\n'), fontsize=5, pad=1)

    # Add label
    fig.text(0.38, 0.92, 'B  Mesh Geometries (top: train, bottom: test)',
             fontsize=9, ha='center')

    # ── Panel C: Grouped bar chart ─────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    x = np.arange(len(methods))
    width = 0.35
    id_means = [float(np.nanmean(id_results[m])) for m in methods]
    ood_means = [float(np.nanmean(ood_results[m])) for m in methods]
    bars1 = ax_c.bar(x - width/2, id_means, width, label='In-dist', color=method_colors, alpha=0.85)
    bars2 = ax_c.bar(x + width/2, ood_means, width, label='OOD',
                     color=method_colors, alpha=0.45, hatch='//')
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([n.replace(' (Learned)', '').replace('GraphNet', 'GNet')
                          for n in method_names], fontsize=7, rotation=15)
    ax_c.set_ylabel('Total Variation')
    ax_c.set_title('C  TV: In-dist vs OOD')
    ax_c.legend(fontsize=7)
    ax_c.grid(True, axis='y', alpha=0.3)

    # ── Panel D: Rate matrix prediction error vs flow time τ ───────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    taus_arr, means_arr, stds_arr = rate_errors
    ax_d.plot(taus_arr, means_arr, color='steelblue', lw=1.8, label='FM')
    ax_d.fill_between(taus_arr, means_arr - stds_arr, means_arr + stds_arr,
                      alpha=0.25, color='steelblue')
    ax_d.set_xlabel('Flow time τ')
    ax_d.set_ylabel('‖R_pred − R_true‖_F / N²')
    ax_d.set_title('D  Rate Matrix Error vs τ')
    ax_d.grid(True, alpha=0.3)
    ax_d.legend(fontsize=7)

    # ── Panel E: Long-horizon rollout ─────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    K = len(rollout_results['learned']) - 1
    steps = np.arange(K + 1)
    for method, name, color in zip(methods, method_names, method_colors):
        ax_e.plot(steps, rollout_results[method], label=name, color=color, lw=1.5)
    ax_e.set_xlabel('Rollout Step')
    ax_e.set_ylabel('Total Variation')
    ax_e.set_title('E  Long-horizon Rollout')
    ax_e.legend(fontsize=7)
    ax_e.grid(True, alpha=0.3)

    # ── Panel F: Generalization gap per test geo ───────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])

    # We need per-geo results — pass them in ood_results if they are dicts
    # ood_results may contain per-geo keys; fall back to aggregated if not
    if isinstance(ood_results.get('learned'), dict):
        test_geos = list(ood_results['learned'].keys())
        gaps = {}
        for method, color in zip(methods, method_colors):
            id_mean = float(np.nanmean(id_results[method]))
            geo_gaps = [float(np.nanmean(ood_results[method].get(g, [float('nan')]))) - id_mean
                        for g in test_geos]
            gaps[method] = geo_gaps

        x = np.arange(len(test_geos))
        w = 0.2
        for i, (method, color) in enumerate(zip(methods, method_colors)):
            ax_f.bar(x + (i - 1.5) * w, gaps[method], w, label=method_names[i], color=color, alpha=0.8)
        ax_f.set_xticks(x)
        ax_f.set_xticklabels(test_geos, rotation=20, fontsize=7)
        ax_f.set_ylabel('OOD − ID TV')
    else:
        # Simple version: per-method generalization gap
        id_means = [float(np.nanmean(id_results[m])) for m in methods]
        ood_means = [float(np.nanmean(ood_results[m])) for m in methods]
        gaps = [o - i for o, i in zip(ood_means, id_means)]
        ax_f.bar(range(len(methods)), gaps, color=method_colors, alpha=0.8)
        ax_f.set_xticks(range(len(methods)))
        ax_f.set_xticklabels([n.replace(' (Learned)', '') for n in method_names],
                             rotation=20, fontsize=7)
        ax_f.set_ylabel('OOD − ID TV')

    ax_f.axhline(0, color='black', lw=0.8, ls='--')
    ax_f.set_title('F  Generalization Gap')
    ax_f.grid(True, axis='y', alpha=0.3)
    ax_f.legend(fontsize=6)

    fig.suptitle('Ex16: Heat Equation Flow Matching on Mesh Geometries', fontsize=12, y=0.98)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figure: {save_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Ex16: Heat Mesh Flow Matching')
    parser.add_argument('--n-points', type=int, default=60)
    parser.add_argument('--n-meshes-per-geo', type=int, default=10)
    parser.add_argument('--n-traj-per-mesh', type=int, default=5)
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--n-epochs', type=int, default=1000)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-test-meshes', type=int, default=5)
    parser.add_argument('--n-test-cases', type=int, default=5)
    parser.add_argument('--rollout-steps', type=int, default=20)
    args = parser.parse_args()

    device = get_device()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    meta_ckpt = os.path.join(
        ckpt_dir,
        f'meta_model_ex16_heat_{args.n_epochs}ep_h{args.hidden_dim}_l{args.n_layers}.pt'
    )
    mgn_ckpt = os.path.join(ckpt_dir, f'mgn_ex16_{args.n_epochs}ep.pt')
    direct_gnn_ckpt = os.path.join(ckpt_dir, f'direct_gnn_ex16_{args.n_epochs}ep.pt')

    # ── Dataset ────────────────────────────────────────────────────────────────
    print("\nBuilding training dataset...", flush=True)
    dataset = HeatMeshDataset(
        geometries=TRAIN_GEOS,
        n_meshes_per_geo=args.n_meshes_per_geo,
        n_traj_per_mesh=args.n_traj_per_mesh,
        n_samples=args.n_samples,
        n_points=args.n_points,
        seed=args.seed,
    )

    # ── FiLM model ─────────────────────────────────────────────────────────────
    # node_context_dim=2 (mu_init, boundary), global_dim=1 (dt)
    film_model = FiLMConditionalGNNRateMatrixPredictor(
        node_context_dim=2,
        global_dim=1,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    )

    train_losses = []
    if os.path.exists(meta_ckpt):
        print(f"\nLoading FiLM model from {meta_ckpt}", flush=True)
        state = torch.load(meta_ckpt, map_location=device)
        film_model.load_state_dict(state['model'])
        train_losses = state.get('losses', [])
    else:
        print(f"\nTraining FiLM model ({args.n_epochs} epochs)...", flush=True)
        result = train_film_conditional(
            film_model, dataset,
            n_epochs=args.n_epochs,
            batch_size=256,
            lr=args.lr,
            device=device,
            loss_weighting='uniform',
            loss_type='rate_kl',
        )
        train_losses = result['losses']
        torch.save({'model': film_model.state_dict(), 'losses': train_losses}, meta_ckpt)
        print(f"  Saved FiLM model to {meta_ckpt}", flush=True)

    film_model = film_model.to(device)

    # ── MeshGraphNet baseline ──────────────────────────────────────────────────
    mgn = MeshGraphNetBaseline(hidden_dim=64, n_layers=3)

    if os.path.exists(mgn_ckpt):
        print(f"\nLoading MeshGraphNet from {mgn_ckpt}", flush=True)
        mgn.load_state_dict(torch.load(mgn_ckpt, map_location=device))
    else:
        print(f"\nTraining MeshGraphNet ({args.n_epochs} epochs)...", flush=True)
        mgn_losses = train_meshgraphnet(
            mgn, dataset.all_pairs,
            n_epochs=args.n_epochs,
            lr=5e-4,
            device=device,
        )
        torch.save(mgn.state_dict(), mgn_ckpt)
        print(f"  Saved MeshGraphNet to {mgn_ckpt}", flush=True)

    mgn = mgn.to(device)

    # ── DirectGNN baseline ─────────────────────────────────────────────────────
    direct_gnn = DirectGNNPredictor(context_dim=3, hidden_dim=64, n_layers=4)

    if os.path.exists(direct_gnn_ckpt):
        print(f"\nLoading DirectGNN from {direct_gnn_ckpt}", flush=True)
        direct_gnn.load_state_dict(torch.load(direct_gnn_ckpt, map_location=device))
    else:
        print(f"\nTraining DirectGNN ({args.n_epochs} epochs)...", flush=True)
        dgnn_pairs = build_direct_gnn_pairs(dataset.all_pairs)
        result_dgnn = train_direct_gnn(
            direct_gnn, dgnn_pairs,
            n_epochs=args.n_epochs,
            lr=5e-4,
            device=device,
            seed=args.seed,
        )
        torch.save(direct_gnn.state_dict(), direct_gnn_ckpt)
        print(f"  Saved DirectGNN to {direct_gnn_ckpt}", flush=True)

    direct_gnn = direct_gnn.to(device)

    # ── Generate test cases ────────────────────────────────────────────────────
    print("\nGenerating in-distribution test cases...", flush=True)
    id_test_cases = generate_test_cases(
        TRAIN_GEOS,
        n_meshes=args.n_test_meshes,
        n_cases_per_mesh=args.n_test_cases,
        n_points=args.n_points,
        seed=9000,
    )

    print("Generating OOD test cases...", flush=True)
    ood_test_cases = generate_test_cases(
        TEST_GEOS,
        n_meshes=args.n_test_meshes,
        n_cases_per_mesh=args.n_test_cases,
        n_points=args.n_points,
        seed=9001,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\nEvaluating in-distribution...", flush=True)
    id_results = eval_one_step(
        film_model, mgn, direct_gnn, id_test_cases, device, n_ode_steps=50
    )

    print("Evaluating OOD...", flush=True)
    ood_results = eval_one_step(
        film_model, mgn, direct_gnn, ood_test_cases, device, n_ode_steps=50
    )

    # Rollout on first OOD test case with enough nodes
    rollout_case = None
    for tc in ood_test_cases:
        if tc['geo'] == 'L_shape':
            rollout_case = tc
            break
    if rollout_case is None and ood_test_cases:
        rollout_case = ood_test_cases[0]

    rollout_results = {'learned': np.zeros(args.rollout_steps + 1),
                       'mgn': np.zeros(args.rollout_steps + 1),
                       'direct_gnn': np.zeros(args.rollout_steps + 1),
                       'laplacian': np.zeros(args.rollout_steps + 1)}

    if rollout_case is not None:
        print(f"\nEvaluating rollout on {rollout_case['geo']}...", flush=True)
        rollout_results = eval_rollout(
            film_model, mgn, direct_gnn, rollout_case,
            K=args.rollout_steps, device=device, n_ode_steps=30
        )

    # ── Print results ──────────────────────────────────────────────────────────
    print_results(id_results, ood_results, rollout_results)

    # ── Collect meshes for visualization ──────────────────────────────────────
    train_show = ['square', 'circle', 'triangle', 'hexagon']
    test_show = ['L_shape', 'annulus', 'star', 'trapezoid']
    train_test_meshes = {'train': {}, 'test': {}}

    for geo in train_show:
        try:
            pts, tris, _ = generate_mesh(geo, args.n_points, seed=args.seed)
            train_test_meshes['train'][geo] = (pts, tris)
        except Exception:
            pass

    for geo in test_show:
        try:
            pts, tris, _ = generate_mesh(geo, args.n_points, seed=args.seed + 1)
            train_test_meshes['test'][geo] = (pts, tris)
        except Exception:
            pass

    # ── Rate matrix error vs τ ─────────────────────────────────────────────────
    print("\nComputing rate matrix prediction errors...", flush=True)
    rate_errors = compute_rate_matrix_errors(film_model, dataset.all_pairs, device)

    # ── Plot main figure ───────────────────────────────────────────────────────
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'ex16_heat_mesh.png'
    )
    print("\nGenerating figure...", flush=True)
    plot_main_figure(
        train_losses=train_losses,
        id_results=id_results,
        ood_results=ood_results,
        rollout_results=rollout_results,
        train_test_meshes=train_test_meshes,
        rate_errors=rate_errors,
        save_path=save_path,
    )

    # ── Prediction comparison figure ───────────────────────────────────────────
    comparison_case = None
    for tc in ood_test_cases:
        if tc['geo'] == 'L_shape':
            comparison_case = tc
            break
    if comparison_case is None and ood_test_cases:
        comparison_case = ood_test_cases[0]

    if comparison_case is not None:
        comp_save = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'ex16_prediction_comparison.png'
        )
        print("\nGenerating prediction comparison figure...", flush=True)
        plot_prediction_comparison(
            film_model, mgn, direct_gnn, comparison_case, device, comp_save
        )

    print("Done.", flush=True)


if __name__ == '__main__':
    main()
