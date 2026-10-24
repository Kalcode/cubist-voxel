"""Cubist: agent-native voxel asset authoring.

Asset scripts start with ``from cubist import *``.
"""

from cubist.core.model import Model, ModelError, asset
from cubist.core.part import Part
from cubist.dsl import (
    Shape,
    ShapeError,
    band,
    box,
    capsule,
    colored,
    cone,
    cylinder,
    ellipsoid,
    extrude,
    faces,
    heightmap,
    inside,
    lathe,
    line,
    region,
    sphere,
    surface,
    torus,
    wedge,
)

__all__ = [
    "Model",
    "ModelError",
    "Part",
    "Shape",
    "ShapeError",
    "asset",
    "band",
    "box",
    "capsule",
    "colored",
    "cone",
    "cylinder",
    "ellipsoid",
    "extrude",
    "faces",
    "heightmap",
    "inside",
    "lathe",
    "line",
    "region",
    "sphere",
    "surface",
    "torus",
    "wedge",
]
