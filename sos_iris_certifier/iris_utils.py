from pydrake.all import (
    HPolyhedron, VPolytope, Sphere, Ellipsoid,
    Iris, IrisOptions, MakeIrisObstacles, Variable
)



def MakeFromHPolyhedronSceneGraph(query, geom, expressed_in=None):
    shape = query.inspector().GetShape(geom)
    if isinstance(shape, (Sphere, Ellipsoid)):
        raise ValueError(f"Sphere or Ellipsoid not Supported")
    return HPolyhedron(query, geom, expressed_in)
def MakeFromVPolytopeSceneGraph(query, geom, expressed_in=None):
    shape = query.inspector().GetShape(geom)
    if isinstance(shape, (Sphere, Ellipsoid)):
        raise ValueError(f"Sphere or Ellipsoid not Supported")
    return VPolytope(query, geom, expressed_in)
