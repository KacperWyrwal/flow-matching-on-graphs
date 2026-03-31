"""
Shared visualization utilities for experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_distribution_bars(ax, dist: np.ndarray, title: str = ''):
    """
    Bar plot of a probability distribution over nodes.

    Args:
        ax: matplotlib Axes
        dist: (N,) array of probabilities
        title: plot title
    """
    N = len(dist)
    ax.bar(range(N), dist, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(0, max(dist.max() * 1.2, 0.1))
    ax.set_xlabel('Node')
    ax.set_ylabel('Probability')
    ax.set_title(title)
    ax.set_xticks(range(N))


def plot_distribution_grid(ax, dist: np.ndarray, rows: int, cols: int, title: str = ''):
    """
    Grid visualization of a probability distribution for a grid graph.

    Args:
        ax: matplotlib Axes
        dist: (N,) array of probabilities, N = rows * cols
        rows: number of rows
        cols: number of columns
        title: plot title
    """
    assert len(dist) == rows * cols, "Distribution size must match rows * cols"
    grid = dist.reshape(rows, cols)

    im = ax.imshow(grid, cmap='Blues', vmin=0, vmax=dist.max() + 1e-10, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Add text annotations
    for r in range(rows):
        for c in range(cols):
            val = grid[r, c]
            color = 'white' if val > 0.5 * dist.max() else 'black'
            ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    return im


def plot_heatmap(ax, data: np.ndarray, xlabel: str = '', ylabel: str = '', title: str = ''):
    """
    Heatmap visualization.

    Args:
        ax: matplotlib Axes
        data: 2D array
        xlabel, ylabel: axis labels
        title: plot title
    """
    im = ax.imshow(data, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return im
