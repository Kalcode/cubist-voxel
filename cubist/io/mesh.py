"""Greedy meshing: exposed voxel faces merged into same-color rectangles.

For each face direction and slice, runs of same-color exposed faces are found per row
with numpy, then identical runs in consecutive rows merge into one quad. Interior faces
never appear, and every exposed face is covered by exactly one quad.

With ``seal=True`` enclosed cavities count as solid, so only faces open to outside air are
meshed. With ``ao=True`` each face also gets per-corner ambient occlusion (0 = most occluded,
3 = open). Fully open faces still merge; faces with any occlusion stay one quad each so the
corner values interpolate correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cubist.core.grid import VoxelGrid
from cubist.core.topology import sealed


@dataclass(frozen=True)
class Quad:
    axis: int  # normal axis: 0 x, 1 y, 2 z
    sign: int  # +1 faces the positive direction, -1 the negative
    layer: int  # world coordinate of the face plane along ``axis``
    u0: int  # world extent along the first other axis
    u1: int
    v0: int  # world extent along the second other axis
    v1: int
    color: int
    # Corner AO (u0v0, u1v0, u0v1, u1v1), 0..3; None when meshed without AO or fully open.
    ao: tuple[int, int, int, int] | None = None


def _runs(rows: np.ndarray) -> list[tuple[int, int, int, int]]:
    """(row, start, end, color) for runs of equal nonzero values along axis 1, row-major."""
    padded = np.pad(rows, ((0, 0), (1, 1)))
    r_idx, c_idx = np.nonzero(padded[:, 1:] != padded[:, :-1])
    # Consecutive change points in the same row bound a run.
    same = r_idx[:-1] == r_idx[1:]
    r, start, end = r_idx[:-1][same], c_idx[:-1][same], c_idx[1:][same]
    color = rows[r, start]
    keep = color != 0
    return list(
        zip(
            r[keep].tolist(),
            start[keep].tolist(),
            end[keep].tolist(),
            color[keep].tolist(),
            strict=True,
        )
    )


def _corner_ao(filled: np.ndarray, k: int, sign: int) -> np.ndarray:
    """(4, nu, nv) corner AO for faces in slice k of an (axis, u, v)-ordered padded array."""
    out_layer = filled[k + sign]
    nu, nv = out_layer.shape
    corners = []
    for cv in (0, 1):
        for cu in (0, 1):
            su, sv = 2 * cu - 1, 2 * cv - 1
            side1 = np.roll(out_layer, -su, axis=0)
            side2 = np.roll(out_layer, -sv, axis=1)
            corner = np.roll(np.roll(out_layer, -su, axis=0), -sv, axis=1)
            open_count = 3 - side1.astype(int) - side2 - corner
            corners.append(np.where(side1 & side2, 0, open_count))
    return np.stack(corners)


def greedy_mesh(grid: VoxelGrid, ao: bool = False, seal: bool = False) -> list[Quad]:
    g = grid.cropped()
    if g.count() == 0:
        return []
    data = np.pad(g.data, 1)
    origin = np.array(g.origin) - 1
    filled = sealed(data > 0) if seal else data > 0
    quads: list[Quad] = []
    for axis in range(3):
        u_axis, v_axis = [i for i in range(3) if i != axis]
        moved_filled = np.moveaxis(filled, (axis, u_axis, v_axis), (0, 1, 2))
        for sign in (1, -1):
            shifted = np.roll(filled, -sign, axis)  # neighbor in the face direction
            faces = np.where(filled & ~shifted, data, 0)
            faces = np.moveaxis(faces, (axis, u_axis, v_axis), (0, 1, 2))
            for k in np.nonzero(faces.any(axis=(1, 2)))[0]:
                layer = int(origin[axis] + k + (1 if sign > 0 else 0))
                mergeable = faces[k]
                if ao:
                    corners = _corner_ao(moved_filled, k, sign)
                    occluded = (mergeable > 0) & np.any(corners < 3, axis=0)
                    for u, v in np.argwhere(occluded):
                        quads.append(
                            Quad(
                                axis,
                                sign,
                                layer,
                                int(origin[u_axis] + u),
                                int(origin[u_axis] + u + 1),
                                int(origin[v_axis] + v),
                                int(origin[v_axis] + v + 1),
                                int(mergeable[u, v]),
                                tuple(int(c) for c in corners[:, u, v]),
                            )
                        )
                    mergeable = np.where(occluded, 0, mergeable)
                # (start, end, color) -> [first row, last row] of the quad growing downward
                active: dict[tuple[int, int, int], list[int]] = {}
                for r, start, end, color in _runs(mergeable):
                    key = (start, end, color)
                    span = active.get(key)
                    if span is not None and span[1] == r - 1:
                        span[1] = r
                        continue
                    if span is not None:
                        quads.append(_quad(axis, sign, layer, span, key, origin, u_axis, v_axis))
                    active[key] = [r, r]
                for key, span in active.items():
                    quads.append(_quad(axis, sign, layer, span, key, origin, u_axis, v_axis))
    return quads


def _quad(axis, sign, layer, span, key, origin, u_axis, v_axis) -> Quad:
    (row0, row1), (start, end, color) = span, key
    return Quad(
        axis=axis,
        sign=sign,
        layer=layer,
        u0=int(origin[u_axis] + row0),
        u1=int(origin[u_axis] + row1 + 1),
        v0=int(origin[v_axis] + start),
        v1=int(origin[v_axis] + end),
        color=color,
    )


def triangle_count(grid: VoxelGrid) -> int:
    return 2 * len(greedy_mesh(grid))
