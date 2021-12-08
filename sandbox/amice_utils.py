import sys
import os
import time
import numpy as np
from functools import partial
import itertools
import pydrake
import meshcat

from pydrake.all import BsplineTrajectoryThroughUnionOfHPolyhedra, IrisInConfigurationSpace, IrisOptions
from pydrake.common import FindResourceOrThrow
from pydrake.geometry import SceneGraph
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.optimization import CalcGridPointsOptions, Toppra
from pydrake.multibody.parsing import LoadModelDirectives, Parser, ProcessModelDirectives
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import RevoluteJoint
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.solvers.mosek import MosekSolver
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import TrajectorySource
from pydrake.trajectories import PiecewisePolynomial
from pydrake.all import Variable, Expression, RotationMatrix
from pydrake.all import MultibodyPositionToGeometryPose, ConnectMeshcatVisualizer, Role, Sphere
from pydrake.all import (
    ConvexSet, HPolyhedron, Hyperellipsoid,
    MathematicalProgram, Solve, le, IpoptSolver,
    Role, Sphere, VPolytope,
    Iris, IrisOptions, MakeIrisObstacles, Variable
)
from pydrake.all import (
    eq, SnoptSolver,
    Sphere, Ellipsoid, GeometrySet,
    RigidBody_, AutoDiffXd, initializeAutoDiff, InverseKinematics
)

import pydrake.symbolic as sym
import symbolic_parsing_helpers as symHelpers
from pydrake.all import RationalForwardKinematics, FindBodyInTheMiddleOfChain
from pydrake.all import GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo, GenerateMonomialBasisWithOrderUpToOne


def MakeFromHPolyhedronOrEllipseSceneGraph(query, geom, expressed_in=None):
    shape = query.inspector().GetShape(geom)
    if isinstance(shape, (Sphere, Ellipsoid)):
        return Hyperellipsoid(query, geom, expressed_in)
    return HPolyhedron(query, geom, expressed_in)
def MakeFromVPolytopeOrEllipseSceneGraph(query, geom, expressed_in=None):
    shape = query.inspector().GetShape(geom)
    print(geom)
    if isinstance(shape, (Sphere, Ellipsoid)):
        return Hyperellipsoid(query, geom, expressed_in)
    return VPolytope(query, geom, expressed_in)
