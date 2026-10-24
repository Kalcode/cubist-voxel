"""What asset scripts import: primitives, CSG, transforms, shape ops, paint, selections."""

from cubist.dsl.context import building
from cubist.dsl.paint import carve, decal
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
    panel,
    polyline,
    pyramid,
    sphere,
    torus,
    wedge,
)
from cubist.dsl.select import band, checker, colored, faces, inside, region, stripes, surface
from cubist.dsl.shape import Shape, ShapeError

__all__ = [
    "Shape",
    "ShapeError",
    "band",
    "box",
    "building",
    "capsule",
    "carve",
    "checker",
    "colored",
    "cone",
    "cylinder",
    "decal",
    "ellipsoid",
    "extrude",
    "faces",
    "heightmap",
    "inside",
    "lathe",
    "line",
    "panel",
    "polyline",
    "pyramid",
    "region",
    "sphere",
    "stripes",
    "surface",
    "torus",
    "wedge",
]
