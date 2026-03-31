"""
Visualisation of the Ex16 heat-mesh training data.

For each geometry (8 train + 4 test) shows:
  - Mesh triangulation
  - One example: mu_source (initial condition)
  - mu_target after dt = heat propagation via expm(dt*R)

Run:
    uv run experiments/ex16_data_viz.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.linalg import expm

# Re-use mesh generation and IC functions from the main script
from ex16_heat_mesh import (
    generate_mesh, mesh_to_graph,
    generate_initial_condition,
    TRAIN_GEOS, TEST_GEOS, IC_TYPES,
)

SEED       = 42
N_POINTS   = 60
DT         = 0.05
IC_EXAMPLE = ['single_peak', 'multi_peak', 'gradient', 'smooth_random',
               'single_peak', 'multi_peak', 'gradient', 'smooth_random',
               'single_peak', 'multi_peak', 'gradient', 'smooth_random']

ALL_GEOS = TRAIN_GEOS + TEST_GEOS   # 12 total


def draw_mesh(ax, points, triangles, values, title, vmin=None, vmax=None, cmap='hot_r'):
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    tc = ax.tripcolor(triang, values, cmap=cmap,
                      vmin=vmin or 0, vmax=vmax or values.max(),
                      shading='gouraud')
    ax.triplot(triang, color='grey', lw=0.3, alpha=0.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=7, pad=2)
    return tc


def main():
    rng = np.random.default_rng(SEED)

    # ── Figure 1: mesh overview (3 columns: mesh / source / target) ───────────
    n_geos = len(ALL_GEOS)
    fig, axes = plt.subplots(n_geos, 3, figsize=(8, n_geos * 1.7))
    fig.suptitle('Ex16 training data: mesh / initial condition / after heat (dt=0.05)',
                 fontsize=9, y=1.002)

    col_labels = ['Mesh', 'μ(0)  initial condition', 'μ(dt)  after heat']
    for c, lbl in enumerate(col_labels):
        axes[0, c].set_title(lbl, fontsize=8, pad=4)

    for row, geo in enumerate(ALL_GEOS):
        tag = 'TEST' if geo in TEST_GEOS else 'train'
        points, triangles, boundary = generate_mesh(geo, N_POINTS, seed=SEED)
        R = mesh_to_graph(points, triangles)
        N = len(points)

        ic_type = IC_EXAMPLE[row % len(IC_EXAMPLE)]
        mu0 = generate_initial_condition(N, points, rng, ic_type)

        P = expm(DT * R)
        mu1 = mu0 @ P
        mu1 = np.clip(mu1, 1e-10, None)
        mu1 /= mu1.sum()

        vmax = max(mu0.max(), mu1.max())

        ax0, ax1, ax2 = axes[row, 0], axes[row, 1], axes[row, 2]

        # Panel: mesh structure (colour = boundary)
        bnd_vals = boundary.astype(float)
        draw_mesh(ax0, points, triangles, bnd_vals,
                  f'{geo} [{tag}]\nN={N}', cmap='Blues')

        # Panel: initial condition
        draw_mesh(ax1, points, triangles, mu0,
                  f'{ic_type}', vmin=0, vmax=vmax)

        # Panel: after heat
        tc = draw_mesh(ax2, points, triangles, mu1,
                       f'dt={DT}', vmin=0, vmax=vmax)

    plt.tight_layout(pad=0.4)
    out = os.path.join(os.path.dirname(__file__), 'ex16_data_viz.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')

    # ── Figure 2: IC variety — show all 4 IC types on the same mesh ──────────
    geo = 'square'
    points, triangles, boundary = generate_mesh(geo, N_POINTS, seed=SEED)
    R = mesh_to_graph(points, triangles)
    N = len(points)

    fig2, axes2 = plt.subplots(2, 4, figsize=(10, 4.5))
    fig2.suptitle('IC variety and heat propagation on square mesh', fontsize=9)

    for col, ic in enumerate(IC_TYPES):
        rng2 = np.random.default_rng(SEED + col * 7)
        mu0 = generate_initial_condition(N, points, rng2, ic)
        P   = expm(DT * R)
        mu1 = mu0 @ P
        mu1 = np.clip(mu1, 1e-10, None)
        mu1 /= mu1.sum()
        vmax = max(mu0.max(), mu1.max())

        draw_mesh(axes2[0, col], points, triangles, mu0,
                  f'μ(0): {ic}', vmin=0, vmax=vmax)
        draw_mesh(axes2[1, col], points, triangles, mu1,
                  f'μ({DT})', vmin=0, vmax=vmax)

    plt.tight_layout(pad=0.4)
    out2 = os.path.join(os.path.dirname(__file__), 'ex16_ic_variety.png')
    plt.savefig(out2, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {out2}')

    # ── Figure 3: multi-step heat evolution on L_shape ────────────────────────
    geo = 'L_shape'
    points, triangles, boundary = generate_mesh(geo, N_POINTS, seed=SEED)
    R = mesh_to_graph(points, triangles)
    N = len(points)

    rng3 = np.random.default_rng(SEED + 99)
    mu = generate_initial_condition(N, points, rng3, 'single_peak')

    dt_step = 0.02
    P = expm(dt_step * R)
    n_steps = 6
    traj = [mu.copy()]
    for _ in range(n_steps):
        mu = mu @ P
        mu = np.clip(mu, 1e-10, None)
        mu /= mu.sum()
        traj.append(mu.copy())

    vmax = traj[0].max()
    fig3, axes3 = plt.subplots(1, n_steps + 1, figsize=(13, 2.2))
    fig3.suptitle(f'Heat diffusion on L_shape (dt={dt_step} per step)', fontsize=9)
    for k, (ax, snap) in enumerate(zip(axes3, traj)):
        draw_mesh(ax, points, triangles, snap,
                  f't={k * dt_step:.2f}', vmin=0, vmax=vmax)
    plt.tight_layout(pad=0.3)
    out3 = os.path.join(os.path.dirname(__file__), 'ex16_heat_evolution.png')
    plt.savefig(out3, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved {out3}')


if __name__ == '__main__':
    main()
