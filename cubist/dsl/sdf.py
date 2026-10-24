"""Vectorized signed distance functions over (N, 3) point arrays (negative = inside).

Formulas follow Inigo Quilez's distance function notes; each takes points already in
the primitive's local frame.
"""

from __future__ import annotations

import numpy as np


def length(v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(v * v, axis=-1))


def box(p: np.ndarray, half: np.ndarray) -> np.ndarray:
    q = np.abs(p) - half
    return length(np.maximum(q, 0.0)) + np.minimum(q.max(axis=-1), 0.0)


def sphere(p: np.ndarray, r: float) -> np.ndarray:
    return length(p) - r


def ellipsoid(p: np.ndarray, radii: np.ndarray) -> np.ndarray:
    # Bound, not exact, but its sign is exact (inside iff sum((p/r)^2) <= 1).
    k0 = length(p / radii)
    k1 = length(p / (radii * radii))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = k0 * (k0 - 1.0) / k1
    return np.where(k1 > 0, d, -radii.min())


def capped_cylinder(radial: np.ndarray, axial: np.ndarray, r: float, half_h: float) -> np.ndarray:
    """Cylinder centered on the axis origin, radius r, extending +-half_h along the axis."""
    dx = radial - r
    dy = np.abs(axial) - half_h
    outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    return outside + np.minimum(np.maximum(dx, dy), 0.0)


def capped_cone(
    radial: np.ndarray, axial: np.ndarray, half_h: float, r_bottom: float, r_top: float
) -> np.ndarray:
    """Frustum centered on the axis origin: r_bottom at -half_h, r_top at +half_h."""
    qx, qy = radial, axial
    k1x, k1y = r_top, half_h
    k2x, k2y = r_top - r_bottom, 2.0 * half_h
    ca_x = qx - np.minimum(qx, np.where(qy < 0.0, r_bottom, r_top))
    ca_y = np.abs(qy) - half_h
    t = np.clip(((k1x - qx) * k2x + (k1y - qy) * k2y) / (k2x * k2x + k2y * k2y), 0.0, 1.0)
    cb_x = qx - k1x + k2x * t
    cb_y = qy - k1y + k2y * t
    s = np.where((cb_x < 0.0) & (ca_y < 0.0), -1.0, 1.0)
    return s * np.sqrt(np.minimum(ca_x * ca_x + ca_y * ca_y, cb_x * cb_x + cb_y * cb_y))


def segment(p: np.ndarray, a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    pa, ba = p - a, b - a
    denom = float(np.dot(ba, ba))
    t = np.clip(pa @ ba / denom, 0.0, 1.0) if denom > 0 else np.zeros(len(p))
    return length(pa - t[:, None] * ba) - r


def smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth minimum; k is the blend radius in voxels."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def torus(radial: np.ndarray, axial: np.ndarray, major: float, minor: float) -> np.ndarray:
    return np.hypot(radial - major, axial) - minor


def segment_chebyshev(p: np.ndarray, a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    """Distance (max-norm) from the nearest point on segment ab, minus r.

    With r = 0.5 this samples to a 26-connected one-voxel line."""
    pa, ba = p - a, b - a
    denom = float(np.dot(ba, ba))
    t = np.clip(pa @ ba / denom, 0.0, 1.0) if denom > 0 else np.zeros(len(p))
    return np.max(np.abs(pa - t[:, None] * ba), axis=-1) - r


def polygon(q: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Signed distance from 2D points (N, 2) to a closed polygon (M, 2), any winding."""
    d = np.sum((q - verts[0]) ** 2, axis=-1)
    s = np.ones(len(q))
    n = len(verts)
    for i in range(n):
        vi, vj = verts[i], verts[i - 1]
        e = vj - vi
        w = q - vi
        ee = float(np.dot(e, e))
        if ee > 0:
            t = np.clip((w @ e) / ee, 0.0, 1.0)
            b = w - t[:, None] * e
            d = np.minimum(d, np.sum(b * b, axis=-1))
        c1 = q[:, 1] >= vi[1]
        c2 = q[:, 1] < vj[1]
        c3 = e[0] * w[:, 1] > e[1] * w[:, 0]
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        s = np.where(flip, -s, s)
    return s * np.sqrt(d)


def round_cone(p: np.ndarray, a: np.ndarray, b: np.ndarray, r1: float, r2: float) -> np.ndarray:
    """Exact distance to the hull of a sphere r1 at a and a sphere r2 at b (Inigo Quilez)."""
    ba = b - a
    l2 = float(np.dot(ba, ba))
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = pa @ ba
    z = y - l2
    w = pa * l2 - y[:, None] * ba
    x2 = np.sum(w * w, axis=1)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d_cap2 = np.sqrt(x2 + z2) * il2 - r2
    d_cap1 = np.sqrt(x2 + y2) * il2 - r1
    d_side = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
    return np.where(
        np.sign(z) * a2 * z2 > k, d_cap2, np.where(np.sign(y) * a2 * y2 < k, d_cap1, d_side)
    )
