"""Which empty space is outside, and which voxels anyone can see."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from cubist.core.grid import VoxelGrid

SIX = ndimage.generate_binary_structure(3, 1)


def outside_air(filled: np.ndarray) -> np.ndarray:
    """Empty cells connected (through faces) to the border of an array padded by 1."""
    labels, _ = ndimage.label(~filled, structure=SIX)
    return labels == labels[0, 0, 0]


def sealed(filled: np.ndarray) -> np.ndarray:
    """``filled`` with enclosed cavities filled in (input padded by 1)."""
    return ~outside_air(filled)


def hidden_mask(grid: VoxelGrid) -> np.ndarray:
    """Filled voxels with no face open to outside air (enclosed cavities don't count)."""
    filled = np.pad(grid.data > 0, 1)
    outside = outside_air(filled)
    visible = np.zeros_like(filled)
    for axis in range(3):
        for shift in (1, -1):
            visible |= filled & np.roll(outside, shift, axis)
    return (filled & ~visible)[1:-1, 1:-1, 1:-1]
