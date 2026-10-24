"""Posed rendering: assets with moving parts, drawn as rotated blocks the way an engine shows
them, never re-voxelized into a staircase.

Each part is greedy-meshed once, with ambient occlusion computed within the part, so shading
stays put as a part moves. A frame projects every face through its part's matrix and rasterizes
the resulting parallelograms at pixel centers with a z-buffer, using the static renderer's
projections, lights, and outlines, so a rest frame looks like ``raster.iso`` / ``raster.ortho``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from cubist.core.model import BuiltAsset
from cubist.io.mesh import greedy_mesh
from cubist.motion.pose import posed_bounds
from cubist.render.raster import (
    _VIEWS,
    AO_LEVELS,
    DEPTH_CUE,
    EDGE_STEP,
    ISO_EDGE_STEP,
    Region,
    ViewRender,
    _finish,
    _iso_dims,
    _ticks,
)

_CORNERS = ((0, 0), (1, 0), (0, 1), (1, 1))  # (u, v) per corner, as Quad.ao is ordered


class _Mesh:
    """One part's faces as arrays: corners (n, 4, 3), normals (n, 3), colors (n,), ao (n, 4)."""

    def __init__(self, grid):
        quads = greedy_mesh(grid, ao=True, seal=True)
        n = len(quads)
        self.corners = np.zeros((n, 4, 3))
        self.normals = np.zeros((n, 3))
        self.colors = np.zeros(n, dtype=np.int64)
        self.ao = np.ones((n, 4))
        for i, q in enumerate(quads):
            u_axis, v_axis = [a for a in range(3) if a != q.axis]
            for c, (cu, cv) in enumerate(_CORNERS):
                self.corners[i, c, q.axis] = q.layer
                self.corners[i, c, u_axis] = q.u1 if cu else q.u0
                self.corners[i, c, v_axis] = q.v1 if cv else q.v0
            self.normals[i, q.axis] = q.sign
            self.colors[i] = q.color
            if q.ao is not None:
                self.ao[i] = AO_LEVELS[list(q.ao)]


def _iso_light(n: np.ndarray) -> np.ndarray:
    up = np.where(n[:, 1] > 0, 1.0, 0.62)
    return 0.68 * n[:, 0] ** 2 + 0.84 * n[:, 2] ** 2 + up * n[:, 1] ** 2


def _ortho_light(n: np.ndarray) -> np.ndarray:
    up = np.where(n[:, 1] > 0, 1.0, 0.62)
    return 0.86 * (n[:, 0] ** 2 + n[:, 2] ** 2) + up * n[:, 1] ** 2


def _next_pow2(v: np.ndarray) -> np.ndarray:
    return 1 << np.ceil(np.log2(np.maximum(v, 1))).astype(np.int64)


class PosedScene:
    def __init__(self, asset: BuiltAsset, table: np.ndarray, glow: np.ndarray | None = None):
        self.asset = asset
        self.table = np.asarray(table)
        self.glow = glow
        self.meshes = {p.name: _Mesh(p.grid) for p in asset.parts}

    def bounds(self, matrices: dict[str, np.ndarray]) -> Region:
        """The whole-voxel box around every part's posed bounding box."""
        return posed_bounds(self.asset, matrices)

    def render(
        self,
        matrices: dict[str, np.ndarray],
        view: str,
        scale: int,
        region: Region,
        hidden: frozenset[str] = frozenset(),
    ) -> ViewRender:
        lo = np.array(region[0], dtype=np.float64)
        hi = np.array(region[1], dtype=np.float64)
        dims = (hi - lo).astype(int)

        # Gather posed faces from every part.
        corners, normals, colors, ao = [], [], [], []
        for name, mesh in self.meshes.items():
            if not len(mesh.colors) or name in hidden:
                continue
            m = matrices.get(name, np.eye(4))
            corners.append(mesh.corners @ m[:3, :3].T + m[:3, 3])
            normals.append(mesh.normals @ m[:3, :3].T)
            colors.append(mesh.colors)
            ao.append(mesh.ao)

        if view == "iso":
            a, b, c = _iso_dims(scale)
            nx, ny, nz = dims
            width, height = (nx + nz) * a, (nx + nz) * b + ny * c
            toward = np.array([1.0, 2.0 * b / c, 1.0])
            toward /= np.linalg.norm(toward)

            def project(p):
                q = p - lo
                sx = (q[..., 0] - q[..., 2]) * a + nz * a
                sy = (q[..., 0] + q[..., 2]) * b - q[..., 1] * c + ny * c
                return sx, sy, -(p @ toward)

            light_of, step, cue_depth = _iso_light, ISO_EDGE_STEP, None
            u_ticks: list = []
            v_ticks: list = []
        else:
            if view not in _VIEWS or view == "bottom":
                raise ValueError(f"unknown posed view {view!r}; choose iso, front, back, ...")
            order, flips, (u_axis, u_desc), (v_axis, v_desc), _ = _VIEWS[view]

            def coord(p, k):
                axis = order[k]
                return hi[axis] - p[..., axis] if flips[k] else p[..., axis] - lo[axis]

            width, height = dims[order[0]] * scale, dims[order[1]] * scale
            toward = np.zeros(3)
            toward[order[2]] = 1.0 if flips[2] else -1.0
            nw = dims[order[2]]

            def project(p):
                return coord(p, 0) * scale, coord(p, 1) * scale, coord(p, 2)

            light_of, step, cue_depth = _ortho_light, EDGE_STEP, max(1, nw - 1)
            u_ticks = _ticks(dims[u_axis], scale, int(lo[u_axis]), int(hi[u_axis]), u_desc)
            v_ticks = _ticks(dims[v_axis], scale, int(lo[v_axis]), int(hi[v_axis]), v_desc)

        rgb = np.zeros((height * width, 3))
        depth = np.full(height * width, np.inf)
        if corners:
            self._raster(
                np.concatenate(corners),
                np.concatenate(normals),
                np.concatenate(colors),
                np.concatenate(ao),
                project,
                toward,
                light_of,
                cue_depth,
                width,
                height,
                rgb,
                depth,
            )
        rgb = rgb.reshape(height, width, 3)
        depth = depth.reshape(height, width)
        if view == "iso" and np.isfinite(depth).any():
            depth = depth - np.min(depth)
        return ViewRender(
            rgb=_finish(rgb, depth, step),
            mask=np.isfinite(depth),
            depth=depth,
            scale=scale,
            u_ticks=u_ticks,
            v_ticks=v_ticks,
        )

    def _raster(
        self, corners, normals, colors, ao, project, toward, light_of, cue_depth,
        width, height, rgb_out, depth_out,
    ) -> None:  # fmt: skip
        keep = normals @ toward > 1e-9  # faces turned toward the camera
        corners, normals, colors, ao = corners[keep], normals[keep], colors[keep], ao[keep]
        if not len(colors):
            return
        sx, sy, dist = project(corners)  # each (n, 4)
        p0 = np.stack([sx[:, 0], sy[:, 0]], -1)
        e1 = np.stack([sx[:, 1], sy[:, 1]], -1) - p0
        e2 = np.stack([sx[:, 2], sy[:, 2]], -1) - p0
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        ok = np.abs(det) > 1e-9
        x0 = np.floor(sx.min(axis=1)).astype(np.int64)
        y0 = np.floor(sy.min(axis=1)).astype(np.int64)
        x1 = np.ceil(sx.max(axis=1)).astype(np.int64)
        y1 = np.ceil(sy.max(axis=1)).astype(np.int64)
        ok &= (x1 > 0) & (y1 > 0) & (x0 < width) & (y0 < height)
        light = light_of(normals)
        base = self.table[colors, :3].astype(np.float64) * light[:, None]
        glowing = None
        if self.glow is not None and np.any(self.glow[colors] > 0):
            glowing = np.minimum(1.0, self.glow[colors])
        d0, d1, d2 = dist[:, 0], dist[:, 1] - dist[:, 0], dist[:, 2] - dist[:, 0]

        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        bw, bh = _next_pow2(x1 - x0), _next_pow2(y1 - y0)
        for i in np.nonzero(ok)[0]:
            buckets[int(bw[i]), int(bh[i])].append(int(i))

        pix, dep, col = [], [], []
        for (w, h), idx in sorted(buckets.items()):
            idx = np.array(idx)
            ix = x0[idx, None, None] + np.arange(w)[None, None, :]
            iy = y0[idx, None, None] + np.arange(h)[None, :, None]
            qx = ix + 0.5 - p0[idx, 0, None, None]
            qy = iy + 0.5 - p0[idx, 1, None, None]
            dt = det[idx, None, None]
            s = (qx * e2[idx, 1, None, None] - qy * e2[idx, 0, None, None]) / dt
            t = (e1[idx, 0, None, None] * qy - e1[idx, 1, None, None] * qx) / dt
            inside = (
                (s >= 0) & (s < 1) & (t >= 0) & (t < 1)
                & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
            )  # fmt: skip
            q, r, cidx = np.nonzero(inside)
            s, t = s[q, r, cidx], t[q, r, cidx]
            k = idx[q]
            a = ao[k]
            shade = (
                a[:, 0] * (1 - s) * (1 - t)
                + a[:, 1] * s * (1 - t)
                + a[:, 2] * (1 - s) * t
                + a[:, 3] * s * t
            )
            d = d0[k] + s * d1[k] + t * d2[k]
            if glowing is None:
                c = base[k] * shade[:, None]
                if cue_depth is not None:
                    c = c * (1.0 - DEPTH_CUE * d / cue_depth)[:, None]
            else:
                f = light[k] * shade
                if cue_depth is not None:
                    f = f * (1.0 - DEPTH_CUE * d / cue_depth)
                f = f + (1.0 - f) * glowing[k]
                c = self.table[colors[k], :3] * f[:, None]
            pix.append(iy[q, r, 0] * width + ix[q, 0, cidx])
            dep.append(d)
            col.append(c)
        if not pix:
            return
        pix_all, dep_all, col_all = np.concatenate(pix), np.concatenate(dep), np.concatenate(col)
        order = np.lexsort((dep_all, pix_all))
        first = np.r_[True, pix_all[order][1:] != pix_all[order][:-1]]
        win = order[first]
        rgb_out[pix_all[win]] = col_all[win]
        depth_out[pix_all[win]] = dep_all[win]
