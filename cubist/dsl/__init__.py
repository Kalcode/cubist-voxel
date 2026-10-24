"""What asset scripts import: primitives, CSG, transforms, shape ops, paint, selections."""

from cubist.dsl.context import building
from cubist.dsl.primitives import (
    box,
    capsule,
    cone,
    cylinder,
    ellipsoid,
    extrude,
    heightmap,
    lathe,
    line,
    sphere,
    torus,
    wedge,
)
from cubist.dsl.select import band, colored, faces, inside, region, surface
from cubist.dsl.shape import Shape, ShapeError

__all__ = [
    "Shape",
    "ShapeError",
    "band",
    "box",
    "building",
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
