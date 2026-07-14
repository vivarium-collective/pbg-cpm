"""Static lattice renderers reproducing the two figure styles in Glazier &
Graner (1993):

* ``render_types``   -- cells filled by type (dark=black, light=light-grey,
  medium=white) with thin black cell-outlines. Matches Figs 7, 12, 18, 20, 22,
  25-28.
* ``render_walls``   -- white fill, black cell walls only. Matches the
  cell-boundary / annealing panels Figs 3, 4.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .types import MEDIUM, DARK, LIGHT

# fill colours (RGB 0..1)
_FILL = {
    MEDIUM: (1.0, 1.0, 1.0),
    DARK: (0.12, 0.12, 0.12),
    LIGHT: (0.80, 0.80, 0.80),
}


def _wall_mask(owner):
    """Boolean mask of sites on a cell wall (owner differs from right/down
    neighbour) — von-Neumann faces, enough to draw 1-px outlines."""
    m = np.zeros(owner.shape, dtype=bool)
    m[:, :-1] |= owner[:, :-1] != owner[:, 1:]
    m[:, 1:] |= owner[:, :-1] != owner[:, 1:]
    m[:-1, :] |= owner[:-1, :] != owner[1:, :]
    m[1:, :] |= owner[:-1, :] != owner[1:, :]
    return m


def center_on_aggregate(owner, type_grid, pad=14):
    """Crop a window centred on the non-medium centre-of-mass so a drifting
    free-floating aggregate stays framed (the paper centres its displays).
    Returns (owner_c, type_c) cropped to the aggregate bounding box + pad."""
    ys, xs = np.where(owner != MEDIUM)
    if xs.size == 0:
        return owner, type_grid
    y0, y1 = max(0, ys.min() - pad), min(owner.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(owner.shape[1], xs.max() + pad + 1)
    return owner[y0:y1, x0:x1], type_grid[y0:y1, x0:x1]


def _rgb_from_types(type_grid):
    rgb = np.ones((*type_grid.shape, 3), dtype=float)
    for t, col in _FILL.items():
        rgb[type_grid == t] = col
    return rgb


def render_types(owner, type_grid, path, title=None, walls=True,
                 wall_color=(0, 0, 0), dpi=150, size=4.0):
    """Fill cells by type, optionally overlay black cell walls; save to `path`."""
    rgb = _rgb_from_types(type_grid)
    if walls:
        rgb[_wall_mask(owner)] = wall_color
    _save(rgb, path, title, dpi, size)


def render_walls(owner, path, title=None, dpi=150, size=4.0):
    """White fill, black cell walls only (cell-boundary view)."""
    rgb = np.ones((*owner.shape, 3), dtype=float)
    rgb[_wall_mask(owner)] = (0, 0, 0)
    _save(rgb, path, title, dpi, size)


def _save(rgb, path, title, dpi, size):
    ny, nx = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(size, size * ny / nx))
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
