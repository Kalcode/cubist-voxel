"""Headless CPU rasterizer for voxel grids: orthographic views and a 2:1 pixel-art iso view.

Shading is flat per face (top light, sides mid, bottom dark) with per-corner ambient
occlusion interpolated across each face, a depth cue in orthographic views, and dark
outlines just outside any surface that stands in front of what is behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from cubist.core.grid import VoxelGrid

Region = tuple[tuple[int, int, int], tuple[int, int, int]]

AO_LEVELS = np.array([0.55, 0.7, 0.85, 1.0])  # by count of open neighbors 0..3
FACE_LIGHT = {"top": 1.0, "side": 0.86, "bottom": 0.62}
ISO_LIGHT = {"top": 1.0, "front": 0.84, "right": 0.68}
DEPTH_CUE = 0.3  # farthest visible face is this much darker than the nearest
EDGE_STEP = 2.0  # depth jump (voxels) that draws an outline inside a view
ISO_EDGE_STEP = 0.9
OUTLINE = np.array([38, 36, 40], dtype=np.float64)
EDGE_DARKEN = 0.72
BACKGROUND = np.array([233, 230, 223], dtype=np.float64)
ISO_HEIGHT = 1.2247  # vertical edge length per horizontal unit for a 30 degree camera


@dataclass
class ViewRender:
    rgb: np.ndarray  # (H, W, 3) uint8
    mask: np.ndarray  # (H, W) bool, True on object pixels
    depth: np.ndarray  # (H, W) float, inf on background
    scale: int
    u_ticks: list[tuple[int, int]]  # (pixel x, world coordinate) at voxel boundaries
    v_ticks: list[tuple[int, int]]  # (pixel y, world coordinate)


def _region(grid: VoxelGrid, region: Region | None) -> Region:
    if region is not None:
        return tuple(map(int, region[0])), tuple(map(int, region[1]))  # type: ignore[return-value]
    b = grid.bounds()
    if b is None:
        return (0, 0, 0), (1, 1, 1)
    return b


def _dense(grid: VoxelGrid, region: Region) -> np.ndarray:
    lo, hi = np.array(region[0]), np.array(region[1])
    data = np.zeros(tuple(hi - lo), dtype=np.uint8)
    coords, colors = grid.voxels()
    local = coords - lo
    keep = np.all((local >= 0) & (local < hi - lo), axis=1)
    local = local[keep]
    data[local[:, 0], local[:, 1], local[:, 2]] = colors[keep]
    return data


# view name -> (axis order into (u, v, w), flips, u/v world axes and direction, light)
# u runs left to right, v top to bottom, w away from the camera.
_VIEWS = {
    "front": ((0, 1, 2), (False, True, True), (0, False), (1, True), "side"),
    "back": ((0, 1, 2), (True, True, False), (0, True), (1, True), "side"),
    "right": ((2, 1, 0), (True, True, True), (2, True), (1, True), "side"),
    "left": ((2, 1, 0), (False, True, False), (2, False), (1, True), "side"),
    "top": ((0, 2, 1), (False, False, True), (0, False), (2, False), "top"),
    "bottom": ((0, 2, 1), (False, True, False), (0, False), (2, True), "bottom"),
}
VIEWS = tuple(_VIEWS)


def _ticks(n: int, scale: int, lo: int, hi: int, descending: bool) -> list[tuple[int, int]]:
    return [(k * scale, hi - k if descending else lo + k) for k in range(n + 1)]


def _upscale(a: np.ndarray, s: int) -> np.ndarray:
    return np.repeat(np.repeat(a, s, axis=0), s, axis=1)


def _bilinear(corners: np.ndarray, s: int) -> np.ndarray:
    """(H, W, 4) corner values (tl, tr, bl, br) -> (H*s, W*s) smoothly interpolated."""
    f = (np.arange(s) + 0.5) / s
    fy, fx = np.meshgrid(f, f, indexing="ij")
    weights = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], -1)
    h, w = corners.shape[:2]
    tiled = np.tile(weights, (h, w, 1))
    return np.sum(_upscale(corners, s) * tiled, axis=-1)


def _outline(rgb: np.ndarray, depth: np.ndarray, step: float) -> np.ndarray:
    """Darken pixels that sit just outside a nearer surface; returns the new rgb."""
    h, w = depth.shape
    padded = np.pad(depth, 1, constant_values=np.inf)
    neighbors = [
        padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
    ]
    nearest = np.minimum.reduce(neighbors)
    background = np.isinf(depth)
    silhouette = background & np.isfinite(nearest)
    with np.errstate(invalid="ignore"):  # inf - inf on background pixels
        crease = ~background & (depth - nearest >= step)
    out = rgb.copy()
    out[silhouette] = OUTLINE
    out[crease] *= EDGE_DARKEN
    return out


def _finish(rgb: np.ndarray, depth: np.ndarray, step: float) -> np.ndarray:
    rgb = np.where(np.isinf(depth)[..., None], BACKGROUND, rgb)
    rgb = _outline(rgb, depth, step)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def ortho(
    grid: VoxelGrid,
    palette: np.ndarray,
    view: str,
    scale: int,
    region: Region | None = None,
    glow: np.ndarray | None = None,
) -> ViewRender:
    if view not in _VIEWS:
        raise ValueError(f"unknown view {view!r}; choose from {', '.join(VIEWS)}")
    order, flips, (u_axis, u_desc), (v_axis, v_desc), light = _VIEWS[view]
    region = _region(grid, region)
    vol = np.transpose(_dense(grid, region), order)
    vol = vol[tuple(slice(None, None, -1) if f else slice(None) for f in flips)]
    occ = vol > 0
    nu, nv, nw = occ.shape

    hit = occ.any(axis=2)
    w = occ.argmax(axis=2)
    color = np.take_along_axis(vol, w[..., None], axis=2)[..., 0]

    # Corner AO from the layer in front of each visible face.
    p = np.pad(occ, 1)
    uu, vv = np.meshgrid(np.arange(nu) + 1, np.arange(nv) + 1, indexing="ij")
    corners = []
    for sv, su in ((-1, -1), (-1, 1), (1, -1), (1, 1)):  # tl, tr, bl, br in image space
        side1 = p[uu + su, vv, w]
        side2 = p[uu, vv + sv, w]
        corner = p[uu + su, vv + sv, w]
        open_count = np.where(side1 & side2, 0, 3 - side1.astype(int) - side2 - corner)
        corners.append(AO_LEVELS[open_count])
    ao = np.stack(corners, axis=-1).transpose(1, 0, 2)  # (v, u, 4)

    cue = 1.0 - DEPTH_CUE * (w / max(1, nw - 1))
    if glow is not None and np.any(glow[color[hit]] > 0):
        # Glowing colors: shading lifted toward the full color by the glow strength.
        shade = _upscale((FACE_LIGHT[light] * cue).T, scale) * _bilinear(ao, scale)
        lift = _upscale(np.minimum(1.0, glow[color]).T, scale)
        shade = shade + (1.0 - shade) * lift
        rgb = _upscale(palette[color, :3].astype(np.float64).transpose(1, 0, 2), scale)
        rgb = rgb * shade[..., None]
    else:
        base = palette[color, :3].astype(np.float64) * (FACE_LIGHT[light] * cue)[..., None]
        rgb = _upscale(base.transpose(1, 0, 2), scale) * _bilinear(ao, scale)[..., None]
    depth = _upscale(np.where(hit, w, np.inf).T.astype(np.float64), scale)

    lo, hi = region
    return ViewRender(
        rgb=_finish(rgb, depth, EDGE_STEP),
        mask=np.isfinite(depth),
        depth=depth,
        scale=scale,
        u_ticks=_ticks(nu, scale, lo[u_axis], hi[u_axis], u_desc),
        v_ticks=_ticks(nv, scale, lo[v_axis], hi[v_axis], v_desc),
    )


# --- iso --------------------------------------------------------------------


def _iso_dims(scale: int) -> tuple[int, int, int]:
    if scale < 2 or scale % 2:
        raise ValueError(f"iso scale must be an even number >= 2, got {scale}")
    return scale, scale // 2, round(ISO_HEIGHT * scale)


@cache
def _sprites(scale: int) -> dict[str, tuple[np.ndarray, ...]]:
    """Pixels covered by each visible face of the unit cube at the origin.

    Returns per face: dx, dy (pixel offsets from the cube's projected origin), u, v (face
    coordinates in [0, 1] for AO interpolation), and depth along the view direction.
    Faces are rasterized at pixel centers; with these integer dimensions no pixel center
    ever lies on a face edge, so neighboring faces tile exactly.
    """
    a, b, c = _iso_dims(scale)
    view = np.array([1.0, 2.0 * b / c, 1.0])
    view /= np.linalg.norm(view)
    xs = np.arange(-a, a) + 0.5
    ys = np.arange(-c, 2 * b) + 0.5
    px, py = np.meshgrid(xs, ys, indexing="ij")
    px, py = px.ravel(), py.ravel()

    faces = {}
    # top (Y = 1): X - Z = px/a, X + Z = (py + c)/b
    s, t = px / a, (py + c) / b
    x, z = (s + t) / 2, (t - s) / 2
    faces["top"] = (x, z, np.stack([x, np.ones_like(x), z], -1))
    # front (Z = 1)
    x = px / a + 1
    y = ((x + 1) * b - py) / c
    faces["front"] = (x, y, np.stack([x, y, np.ones_like(x)], -1))
    # right (X = 1)
    z = 1 - px / a
    y = ((1 + z) * b - py) / c
    faces["right"] = (z, y, np.stack([np.ones_like(z), y, z], -1))

    out = {}
    for name, (u, v, point) in faces.items():
        inside = (u > 0) & (u < 1) & (v > 0) & (v < 1)
        out[name] = (
            np.floor(px[inside]).astype(np.int64),
            np.floor(py[inside]).astype(np.int64),
            u[inside],
            v[inside],
            point[inside] @ view,
        )
    return out


# face -> (outward normal, in-plane axis for u, in-plane axis for v)
_ISO_FACES = {
    "top": ((0, 1, 0), 0, 2),
    "front": ((0, 0, 1), 0, 1),
    "right": ((1, 0, 0), 2, 1),
}


def iso(
    grid: VoxelGrid,
    palette: np.ndarray,
    scale: int,
    region: Region | None = None,
    glow: np.ndarray | None = None,
) -> ViewRender:
    """3/4 view from the front-right, above: shows top, front (+Z), and right (+X) faces."""
    a, b, c = _iso_dims(scale)
    region = _region(grid, region)
    vol = _dense(grid, region)
    occ = vol > 0
    nx, ny, nz = occ.shape
    width, height = (nx + nz) * a, (nx + nz) * b + ny * c
    view = np.array([1.0, 2.0 * b / c, 1.0])
    view /= np.linalg.norm(view)
    p = np.pad(occ, 1)
    sprites = _sprites(scale)

    pix, dep, col = [], [], []
    for name, (normal, u_axis, v_axis) in _ISO_FACES.items():
        n = np.array(normal)
        vox = np.argwhere(occ & ~p[1 + n[0] :, 1 + n[1] :, 1 + n[2] :][:nx, :ny, :nz])
        if len(vox) == 0:
            continue
        dx, dy, u, v, d_off = sprites[name]
        ax = (vox[:, 0] - vox[:, 2]) * a + nz * a
        ay = (vox[:, 0] + vox[:, 2]) * b - vox[:, 1] * c + ny * c
        sx = ax[:, None] + dx[None, :]
        sy = ay[:, None] + dy[None, :]

        # Corner AO in the layer just outside the face, interpolated across it.
        out = vox + 1 + n
        e_u = np.eye(3, dtype=np.int64)[u_axis]
        e_v = np.eye(3, dtype=np.int64)[v_axis]
        ao_c = {}
        for cu in (0, 1):
            for cv in (0, 1):
                du, dv = e_u * (2 * cu - 1), e_v * (2 * cv - 1)
                side1 = p[tuple((out + du).T)]
                side2 = p[tuple((out + dv).T)]
                corner = p[tuple((out + du + dv).T)]
                open_count = np.where(side1 & side2, 0, 3 - side1.astype(int) - side2 - corner)
                ao_c[cu, cv] = AO_LEVELS[open_count][:, None]
        ao = (
            ao_c[0, 0] * ((1 - u) * (1 - v))
            + ao_c[1, 0] * (u * (1 - v))
            + ao_c[0, 1] * ((1 - u) * v)
            + ao_c[1, 1] * (u * v)
        )

        colors = vol[tuple(vox.T)]
        base = palette[colors, :3].astype(np.float64) * ISO_LIGHT[name]
        pix.append((sy * width + sx).ravel())
        dep.append((vox @ view)[:, None].repeat(len(dx), 1).ravel() + np.tile(d_off, len(vox)))
        if glow is not None and np.any(glow[colors] > 0):
            shade = ISO_LIGHT[name] * ao
            shade = shade + (1.0 - shade) * np.minimum(1.0, glow[colors])[:, None]
            col.append((palette[colors, :3][:, None, :] * shade[..., None]).reshape(-1, 3))
        else:
            col.append((base[:, None, :] * ao[..., None]).reshape(-1, 3))

    rgb = np.zeros((height * width, 3))
    depth = np.full(height * width, np.inf)
    if pix:
        pix_all, dep_all, col_all = np.concatenate(pix), np.concatenate(dep), np.concatenate(col)
        order = np.lexsort((-dep_all, pix_all))
        first = np.r_[True, pix_all[order][1:] != pix_all[order][:-1]]
        win = order[first]
        rgb[pix_all[win]] = col_all[win]
        # Store distance from the camera (smaller = nearer) for outlines.
        depth[pix_all[win]] = -dep_all[win]
    rgb = rgb.reshape(height, width, 3)
    depth = depth.reshape(height, width)
    depth = depth - (np.min(depth) if np.isfinite(depth).any() else 0)
    return ViewRender(
        rgb=_finish(rgb, depth, ISO_EDGE_STEP),
        mask=np.isfinite(depth),
        depth=depth,
        scale=scale,
        u_ticks=[],
        v_ticks=[],
    )
