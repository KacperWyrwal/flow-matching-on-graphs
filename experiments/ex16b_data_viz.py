"""
Visualisation of the Ex16b advection-diffusion training data.
Run: uv run experiments/ex16b_data_viz.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from ex16_heat_mesh import (
    generate_mesh, generate_initial_condition,
    TRAIN_GEOS, TEST_GEOS,
)
from ex16b_advdiff import (
    make_velocity_field, build_advection_diffusion_rate_matrix, simulate_exact,
    VELOCITY_FIELDS,
)


# ── Figure 1: All geometries with vortex field ────────────────────────────────

def figure_geo_variety(save_path, n_points=60, seed=42):
    """
    12 rows × 3 cols.
    Row per geometry (8 train + 4 test).
    Col 0: mesh + velocity arrows
    Col 1: initial condition
    Col 2: after advdiff at T=0.1
    """
    all_geos = TRAIN_GEOS + TEST_GEOS   # 12 total

    # Fixed vortex field for all geometries
    v_field = make_velocity_field('vortex', {'strength': 1.5, 'cx': 0.5, 'cy': 0.5})
    T_viz   = 0.1
    D       = 1.0
    alpha   = 1.0

    rng = np.random.default_rng(seed)

    n_rows = len(all_geos)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3.2 * n_rows))

    for row_idx, geo in enumerate(all_geos):
        geo_seed = seed + row_idx * 17
        try:
            points, triangles, boundary_mask = generate_mesh(geo, n_points, seed=geo_seed)
        except Exception as e:
            print(f"  Mesh gen failed for {geo}: {e}", flush=True)
            continue

        N = len(points)

        try:
            R = build_advection_diffusion_rate_matrix(
                points, triangles, v_field, D=D, alpha=alpha)
        except Exception as e:
            print(f"  R build failed for {geo}: {e}", flush=True)
            continue

        # Initial condition
        ic_rng = np.random.default_rng(seed + row_idx * 100)
        mu_init = generate_initial_condition(N, points, ic_rng, ic_type='single_peak')

        try:
            mu_final = simulate_exact(mu_init, R, T_viz)
        except Exception as e:
            print(f"  Sim failed for {geo}: {e}", flush=True)
            mu_final = mu_init.copy()

        # Velocity at nodes (normalized for display)
        v_nodes = np.array([v_field(points[a]) for a in range(N)], dtype=np.float32)
        v_max   = np.abs(v_nodes).max() + 1e-10
        v_norm  = v_nodes / v_max

        triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
        is_test = (geo in TEST_GEOS)
        mesh_color = 'darkred' if is_test else 'navy'

        # Arrow scale relative to domain size
        scale = (points[:, 0].max() - points[:, 0].min()) * 0.08

        # Col 0: mesh + arrows
        ax0 = axes[row_idx, 0]
        ax0.triplot(triang, lw=0.3, color=mesh_color, alpha=0.5)
        ax0.quiver(points[:, 0], points[:, 1],
                   v_norm[:, 0] * scale, v_norm[:, 1] * scale,
                   color='crimson', alpha=0.8, scale=1.0, scale_units='xy',
                   width=0.004)
        ax0.set_aspect('equal')
        ax0.axis('off')
        tag = '(test)' if is_test else ''
        ax0.set_title(f'{geo} {tag}', fontsize=7)

        # Col 1: initial condition
        ax1 = axes[row_idx, 1]
        vmax1 = mu_init.max() + 1e-10
        im1 = ax1.tripcolor(triang, mu_init, shading='gouraud',
                             vmin=0, vmax=vmax1, cmap='hot_r')
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title('Initial', fontsize=7)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Col 2: after advdiff
        ax2 = axes[row_idx, 2]
        vmax2 = max(mu_init.max(), mu_final.max()) + 1e-10
        im2 = ax2.tripcolor(triang, mu_final, shading='gouraud',
                             vmin=0, vmax=vmax2, cmap='hot_r')
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title(f'After T={T_viz}', fontsize=7)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle('Ex16b: Geometry Variety — vortex velocity field', fontsize=11, y=1.002)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}", flush=True)


# ── Figure 2: Velocity field variety on square ────────────────────────────────

def figure_field_variety(save_path, n_points=60, seed=42):
    """
    3 rows × 5 cols.
    Fixed geometry: square.
    Cols: uniform, vortex, source, sink, shear.
    Row 0: velocity arrows on mesh
    Row 1: initial condition (single_peak, same for all)
    Row 2: result after T=0.1
    """
    geo   = 'square'
    T_viz = 0.1
    D     = 1.0
    alpha = 1.0

    points, triangles, boundary_mask = generate_mesh(geo, n_points, seed=seed)
    N = len(points)
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    scale  = (points[:, 0].max() - points[:, 0].min()) * 0.08

    # Same IC for all columns
    ic_rng = np.random.default_rng(seed)
    mu_init = generate_initial_condition(N, points, ic_rng, ic_type='single_peak')

    # Default params per field type
    field_params_map = {
        'uniform': {'theta': np.pi / 6},
        'vortex':  {'strength': 1.5, 'cx': 0.5, 'cy': 0.5},
        'source':  {'strength': 1.5, 'cx': 0.5, 'cy': 0.5},
        'sink':    {'strength': 1.5, 'cx': 0.5, 'cy': 0.5},
        'shear':   {'strength': 2.0, 'cy': 0.5},
    }

    fig, axes = plt.subplots(3, 5, figsize=(15, 8))

    for col_idx, field_type in enumerate(VELOCITY_FIELDS):
        params  = field_params_map[field_type]
        v_field = make_velocity_field(field_type, params)

        v_nodes = np.array([v_field(points[a]) for a in range(N)], dtype=np.float32)
        v_max   = np.abs(v_nodes).max() + 1e-10
        v_norm  = v_nodes / v_max

        try:
            R       = build_advection_diffusion_rate_matrix(
                points, triangles, v_field, D=D, alpha=alpha)
            mu_final = simulate_exact(mu_init, R, T_viz)
        except Exception as e:
            print(f"  Failed for {field_type}: {e}", flush=True)
            mu_final = mu_init.copy()

        # Row 0: velocity arrows on mesh
        ax0 = axes[0, col_idx]
        ax0.triplot(triang, lw=0.3, color='navy', alpha=0.4)
        ax0.quiver(points[:, 0], points[:, 1],
                   v_norm[:, 0] * scale, v_norm[:, 1] * scale,
                   color='crimson', alpha=0.8, scale=1.0, scale_units='xy',
                   width=0.004)
        ax0.set_aspect('equal')
        ax0.axis('off')
        ax0.set_title(field_type, fontsize=8)

        # Row 1: initial condition
        ax1 = axes[1, col_idx]
        vmax_ic = mu_init.max() + 1e-10
        im1 = ax1.tripcolor(triang, mu_init, shading='gouraud',
                            vmin=0, vmax=vmax_ic, cmap='hot_r')
        ax1.set_aspect('equal')
        ax1.axis('off')
        if col_idx == 0:
            ax1.set_title('Initial', fontsize=7)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Row 2: after advdiff
        ax2 = axes[2, col_idx]
        vmax2 = max(mu_init.max(), mu_final.max()) + 1e-10
        im2 = ax2.tripcolor(triang, mu_final, shading='gouraud',
                            vmin=0, vmax=vmax2, cmap='hot_r')
        ax2.set_aspect('equal')
        ax2.axis('off')
        if col_idx == 0:
            ax2.set_title(f'After T={T_viz}', fontsize=7)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Row labels
    axes[0, 0].set_ylabel('Velocity', fontsize=8)
    axes[1, 0].set_ylabel('Initial', fontsize=8)
    axes[2, 0].set_ylabel(f'After T={T_viz}', fontsize=8)

    fig.suptitle('Ex16b: Velocity Field Variety — square geometry', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}", flush=True)


# ── Figure 3: Temporal evolution on L_shape ───────────────────────────────────

def figure_evolution(save_path, n_points=60, seed=42):
    """
    1 row × 7 cols.
    Geometry: L_shape, field: vortex.
    t = 0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12
    Each panel: tripcolor + faint quiver arrows.
    """
    geo        = 'L_shape'
    field_type = 'vortex'
    v_params   = {'strength': 1.5, 'cx': 0.35, 'cy': 0.35}
    v_field    = make_velocity_field(field_type, v_params)
    D          = 1.0
    alpha      = 1.0
    t_vals     = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]

    try:
        points, triangles, boundary_mask = generate_mesh(geo, n_points, seed=seed)
    except Exception as e:
        print(f"  Mesh gen failed for {geo}: {e}", flush=True)
        return

    N = len(points)
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    scale  = (points[:, 0].max() - points[:, 0].min()) * 0.07

    try:
        R = build_advection_diffusion_rate_matrix(
            points, triangles, v_field, D=D, alpha=alpha)
    except Exception as e:
        print(f"  R build failed: {e}", flush=True)
        return

    # Initial condition
    ic_rng  = np.random.default_rng(seed)
    mu_init = generate_initial_condition(N, points, ic_rng, ic_type='single_peak')

    # Velocity for quiver (same throughout)
    v_nodes = np.array([v_field(points[a]) for a in range(N)], dtype=np.float32)
    v_max   = np.abs(v_nodes).max() + 1e-10
    v_norm  = v_nodes / v_max

    # Compute distributions at each t
    snapshots = []
    for t in t_vals:
        if t == 0.0:
            snapshots.append(mu_init.copy())
        else:
            try:
                mu_t = simulate_exact(mu_init, R, t)
                snapshots.append(mu_t)
            except Exception as e:
                print(f"  Sim failed at t={t}: {e}", flush=True)
                snapshots.append(mu_init.copy())

    vmax_all = max(s.max() for s in snapshots) + 1e-10

    fig, axes = plt.subplots(1, len(t_vals), figsize=(2.8 * len(t_vals), 3.5))

    for col_idx, (t, mu_t) in enumerate(zip(t_vals, snapshots)):
        ax = axes[col_idx]
        im = ax.tripcolor(triang, mu_t, shading='gouraud',
                          vmin=0, vmax=vmax_all, cmap='hot_r')
        ax.quiver(points[:, 0], points[:, 1],
                  v_norm[:, 0] * scale, v_norm[:, 1] * scale,
                  color='cyan', alpha=0.35, scale=1.0, scale_units='xy',
                  width=0.003)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f't = {t:.2f}', fontsize=8)

    fig.suptitle(f'Ex16b: Temporal Evolution — {geo} / {field_type}', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}", flush=True)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    seed    = 42

    print("Generating Figure 1: geometry variety...", flush=True)
    figure_geo_variety(
        save_path=os.path.join(out_dir, 'ex16b_data_viz.png'),
        n_points=60, seed=seed,
    )

    print("Generating Figure 2: velocity field variety...", flush=True)
    figure_field_variety(
        save_path=os.path.join(out_dir, 'ex16b_field_variety.png'),
        n_points=60, seed=seed,
    )

    print("Generating Figure 3: temporal evolution...", flush=True)
    figure_evolution(
        save_path=os.path.join(out_dir, 'ex16b_evolution.png'),
        n_points=60, seed=seed,
    )

    print("All figures saved.", flush=True)


if __name__ == '__main__':
    main()
