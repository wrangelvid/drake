from pydrake.all import (
    ConvexSet, HPolyhedron, Hyperellipsoid,
    MathematicalProgram, Solve, le, IpoptSolver,
    Role, Sphere, VPolytope,
    Iris, IrisOptions, MakeIrisObstacles, Variable,
    BsplineTrajectoryThroughUnionOfHPolyhedra,
    eq, SnoptSolver,
    Sphere, Ellipsoid, GeometrySet,
    RigidBody_, AutoDiffXd, initializeAutoDiff, InverseKinematics,
    RationalForwardKinematics, FindBodyInTheMiddleOfChain
)
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



def set_up_iris_t_space(plant, scene_graph, context, settings = None):
    

    #hardcoded
    dReal_polytope_tol = 1e-10
    starting_vol_eps = 1e-3


    forward_kin = RationalForwardKinematics(plant)
    query = scene_graph.get_query_output_port().Eval(scene_graph.GetMyContextFromRoot(context))
    q_star = np.zeros(forward_kin.t().shape[0])

    def convert_RationalForwardPoseList_to_TransformExpressionList(pose_list):
        ret = []
        for p in pose_list:
            ret.append(p.asRigidTransformExpr())
        return ret

    def MakeFromSceneGraph(query, geom, expressed_in=None):
        shape = query.inspector().GetShape(geom)
        if isinstance(shape, (Sphere, Ellipsoid)):
            return Hyperellipsoid(query, geom, expressed_in)
        return HPolyhedron(query, geom, expressed_in)

    def GrowthVolumeRational(E, X_WA, X_WB, setA, setB, A, b, guess=None):
        snopt = SnoptSolver()
        prog = MathematicalProgram()
        t = forward_kin.t()
        prog.AddDecisionVariables(t)
        
        if guess is not None:
            prog.SetInitialGuess(t, guess)

        prog.AddLinearConstraint(A, b-np.inf, b, t)
        p_AA = prog.NewContinuousVariables(3, "p_AA")
        p_BB = prog.NewContinuousVariables(3, "p_BB")
        setA.AddPointInSetConstraints(prog, p_AA)
        setB.AddPointInSetConstraints(prog, p_BB)
        prog.AddQuadraticErrorCost(E.A().T @ E.A(), E.center(), t)

        p_WA = X_WA.multiply(p_AA+0)

        p_WB = X_WB.multiply(p_BB+0)

        prog.AddConstraint(eq(p_WA, p_WB))
        result = snopt.Solve(prog)
        return result.is_success(), result.get_optimal_cost(), result.GetSolution(t)

    def TangentPlane(self, point):
        a = 2 * self.A().T @ self.A() @ (point - self.center())
        a = a / np.linalg.norm(a)
        b = a.dot(point)
        return a, b

    Hyperellipsoid.TangentPlane = TangentPlane

    def iris_rational_space(query, point, require_containment_points=[], termination_threshold=2e-2, iteration_limit=100):
        dim = plant.num_positions()
        lb = plant.GetPositionLowerLimits()
        rational_lb = forward_kin.ComputeTValue(lb, q_star)
        ub = plant.GetPositionUpperLimits()
        rational_ub = forward_kin.ComputeTValue(ub, q_star)
        volume_of_unit_sphere = 4.0*np.pi/3.0
        
        E = Hyperellipsoid(np.eye(3)/starting_vol_eps, point)

        best_volume = starting_vol_eps**dim * volume_of_unit_sphere
        

        max_faces = 10
        
        link_poses_by_body_index_rat_pose = forward_kin.CalcLinkPoses(q_star, 
                                                            plant.world_body().index())
        X_WA_list = convert_RationalForwardPoseList_to_TransformExpressionList(link_poses_by_body_index_rat_pose)
        X_WB_list = convert_RationalForwardPoseList_to_TransformExpressionList(link_poses_by_body_index_rat_pose)

        inspector = query.inspector()
        pairs = inspector.GetCollisionCandidates()

        P = HPolyhedron.MakeBox(rational_lb, rational_ub)
        A = np.vstack((P.A(), np.zeros((max_faces*len(pairs),3))))  # allow up to 10 faces per pair.
        b = np.concatenate((P.b(), np.zeros(max_faces*len(pairs))))

        geom_ids = inspector.GetGeometryIds(GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity)
        sets = {geom:MakeFromSceneGraph(query, geom, inspector.GetFrameId(geom)) for geom in geom_ids}
        body_indexes_by_geom_id = {geom:
                                plant.GetBodyFromFrameId(inspector.GetFrameId(geom)).index() for geom in geom_ids} 
        
        #Turn onto true to certify regions using SOS at each iteration.
        certify = False
        # refine polytopes if not certified collision free
        refine = False and certify
            
        iteration = 0
        num_faces = 2*len(lb)
        while True:
            ## Find separating hyperplanes

            for geomA, geomB in pairs:
                #print(f"geomA={inspector.GetName(geomA)}, geomB={inspector.GetName(geomB)}")
                # Run snopt at the beginning
                while True:
                    X_WA = X_WA_list[int(body_indexes_by_geom_id[geomA])]
                    X_WB = X_WB_list[int(body_indexes_by_geom_id[geomB])]
                    success, growth, qstar = GrowthVolumeRational(E,
                        X_WA, X_WB,
                        sets[geomA], sets[geomB], 
                        A[:num_faces,:], b[:num_faces] - dReal_polytope_tol, 
                        point)
                    if success:
                        #print(f"snopt_example={qstar}, growth = {growth}")
                        # Add a face to the polytope
                        A[num_faces,:], b[num_faces] = E.TangentPlane(qstar)
                        num_faces += 1
                        if num_faces > max_faces:
                            break
                        #     A = np.vstack((A, np.zeros((max_faces*len(pairs),3))))  # allow up to 10 faces per pair.
                        #     b = np.concatenate((b, np.zeros(max_faces*len(pairs))))
                    else:
                        break

                if certify:
                    pass
                

            if any([np.any(A[:num_faces,:] @ p > b[:num_faces]) for p in require_containment_points]):
                print("terminating because a required containment point would have not been contained")
                break

            P = HPolyhedron(A[:num_faces,:],b[:num_faces])

            E = P.MaximumVolumeInscribedEllipsoid()
            print(iteration)

            iteration += 1
            if iteration >= iteration_limit:
                break

            volume = volume_of_unit_sphere / np.linalg.det(E.A())
            if volume - best_volume <= termination_threshold:
                break
            best_volume = volume

        return P
    return iris_rational_space, query, forward_kin

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

def EvaluatePlanePair(plane_pair, eval_dict):
    a_res = []
    for ai in plane_pair[0]:
        a_res.append(ai.Evaluate(eval_dict))
    return (np.array(a_res), plane_pair[1].Evaluate(eval_dict))

class RegionCertifier:
    def __init__(self, plant, scene_graph, context):
        self.query = scene_graph.get_query_output_port().Eval(scene_graph.GetMyContextFromRoot(context))

        self.inspector = self.query.inspector()
        self.pairs = self.inspector.GetCollisionCandidates()

        self.geom_ids = self.inspector.GetGeometryIds(GeometrySet(self.inspector.GetAllGeometryIds()), Role.kProximity)
        self.HPolyhedronSets = {geom: MakeFromHPolyhedronOrEllipseSceneGraph(self.query, geom, self.inspector.GetFrameId(geom)) for
                           geom in self.geom_ids}
        self.VPolyhedronSets = {geom: MakeFromVPolytopeOrEllipseSceneGraph(self.query, geom, self.inspector.GetFrameId(geom)) for geom
                           in self.geom_ids}

        self.body_indexes_by_geom_id = {geom:
                                       plant.GetBodyFromFrameId(self.inspector.GetFrameId(geom)).index() for geom in
                                   self.geom_ids}
        self.forward_kin = RationalForwardKinematics(plant)
        self.convSolver = MosekSolver()
        self.t_kin = self.forward_kin.t()

        self.link_poses_by_body_index_multilinear_pose = self.forward_kin.CalcLinkPosesAsMultilinearPolynomials(np.zeros_like(self.t_kin),
                                                                                                      plant.world_body().index())
        self.X_WA_multilinear_list = [(r.rotation().copy(), r.translation().copy()) for r in
                                 self.link_poses_by_body_index_multilinear_pose]

    def construct_first_order_separating_hyperplane(self, prog, t, order=2, plane_name=''):
        if plane_name != '':
            plane_name = '_' + plane_name
        t_basis = sym.MonomialBasis(t, order)
        t_basis = np.array([sym.Polynomial(v) for v in t_basis])
        a_A_coeffs = prog.NewContinuousVariables(3, t_basis.shape[0], 'a_A' + plane_name)
        a_poly = a_A_coeffs @ t_basis

        b_A_coeffs = prog.NewContinuousVariables(1, t_basis.shape[0], 'b_A' + plane_name)
        b_poly = b_A_coeffs @ t_basis

        for i, p in enumerate(a_poly):
            a_poly[i].SetIndeterminates(sym.Variables(t))
        for i, p in enumerate(b_poly):
            b_poly[i].SetIndeterminates(sym.Variables(t))
        dec_vars = [*a_A_coeffs.flatten().tolist(), *b_A_coeffs.flatten().tolist()]
        ntmp = a_A_coeffs.flatten().shape[0]
        Qtmp, btmp = np.eye(ntmp), np.zeros(ntmp)
        prog.AddQuadraticCost(Qtmp, btmp, a_A_coeffs.flatten())
        return prog, a_poly, b_poly.item(), dec_vars


    def putinarPsatConstraint(self, prog, p, t,
                              poly_to_cert, lagrange_mult_degree=2, var_epsilon=None):
        A = poly_to_cert.A()
        b = poly_to_cert.b()
        n = b.shape[0]
        if var_epsilon is None:
            var_epsilon = np.zeros(n)
        #     lagrange_poly, Q = prog.NewSosPolynomial(sym.Variables(t), lagrange_mult_degree)
        lagrange_poly, Q = prog.NewSosPolynomial(GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(
            sym.Variables(t)))
        lagrange_poly.SetIndeterminates(sym.Variables(t))
        prog.AddSosConstraint(lagrange_poly)
        constraint_poly = lagrange_poly
        for i in range(n):
            #         lagrange_poly, Q = prog.NewSosPolynomial(sym.Variables(t), lagrange_mult_degree)
            lagrange_poly, Q = prog.NewSosPolynomial(
                GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(sym.Variables(t)))
            lagrange_poly.SetIndeterminates(sym.Variables(t))
            prog.AddSosConstraint(lagrange_poly)
            constraint_poly += lagrange_poly * sym.Polynomial(b[i] - var_epsilon[i] - A[i, :] @ t)
        constraint_poly.SetIndeterminates(sym.Variables(t))
        tol = 1e-3
        prog.AddEqualityConstraintBetweenPolynomials(constraint_poly, p - tol)
        #     prog.AddSosConstraint(p-constraint_poly)
        return prog, (constraint_poly, p - tol)


    def makeBodyHyperplaneSidePolynomials(self, prog, a_plane_poly, b_plane_poly,
                                          VPoly, R_WA, p_WA, t, poly_to_cert, leq_or_geq,
                                          lagrange_mult_degree=2, var_epsilon=None, base_point_as_multilinear_poly=None):
        num_verts = VPoly.vertices().shape[1]

        vertex_pos = R_WA @ (VPoly.vertices()) + np.repeat(p_WA[:, np.newaxis], num_verts, 1)
        zero_poly = sym.Polynomial(0)

        dens = np.ndarray(shape=vertex_pos.shape, dtype=object)
        nums = np.ndarray(shape=vertex_pos.shape, dtype=object)
        for i, row in enumerate(vertex_pos):

            for j, v in enumerate(row):
                vertex_pos[i, j] = self.forward_kin.ConvertMultilinearPolynomialToRationalFunction(v)
                dens[i, j] = vertex_pos[i, j].denominator()
                nums[i, j] = vertex_pos[i, j].numerator()

        for c in range(dens.shape[1]):
            for r in range(dens.shape[0] - 1):
                if not dens[r, c].EqualTo(dens[r + 1, c]):
                    raise ValueError("problem")

        #     plane_rats = a_plane_poly.dot(vertex_pos)
        plane_polys = np.array([None for _ in range(vertex_pos.shape[1])])
        for i in range(vertex_pos.shape[1]):
            #         p.SetIndeterminates(sym.Variables(t))

            if leq_or_geq == 'leq':
                # a^Tx + b <= -1 -> -(a^Tx+b+1)>=0
                plane_polys[i] = (-a_plane_poly.dot(nums[:, i]) - (b_plane_poly + 1) * (dens[:, i].sum()))
            elif leq_or_geq == 'geq':
                # a^Tx+b >= 1 -> a^Tx+b -1 >= 0
                plane_polys[i] = (a_plane_poly.dot(nums[:, i]) + (b_plane_poly - 1) * (dens[:, i].sum()))
            else:
                raise ValueError("leq_or_geq arg must be leq or geq not {}".format(leq_or_geq))
            plane_polys[i].SetIndeterminates(sym.Variables(t))
        return plane_polys


    def add_pair_constraint(self, geomA, geomB, prog, poly_to_cert,
                            lagrange_mult_degree=2, var_epsilon=None):
        VPolyA, VPolyB = self.VPolyhedronSets[geomA], self.VPolyhedronSets[geomB]
        prog, a_plane_poly, b_plane_poly, dec_vars = self.construct_first_order_separating_hyperplane(prog, self.t_kin)

        R_WA, p_WA = self.X_WA_multilinear_list[int(self.body_indexes_by_geom_id[geomA])]
        R_WB, p_WB = self.X_WA_multilinear_list[int(self.body_indexes_by_geom_id[geomB])]

        base_point_poly = None  # R_WA@VPolyA.vertices().mean(axis = 1)-p_WA
        #     base_point = np.array([forward_kin.ConvertMultilinearPolynomialToRationalFunction(p) for p in base_point_poly])

        plane_polys_A = self.makeBodyHyperplaneSidePolynomials(prog, a_plane_poly, b_plane_poly,
                                                          VPolyA, R_WA, p_WA, self.t_kin,
                                                          poly_to_cert, 'leq', lagrange_mult_degree,
                                                          var_epsilon, base_point_poly)
        plane_polys_B = self.makeBodyHyperplaneSidePolynomials(prog, a_plane_poly, b_plane_poly,
                                                          VPolyB, R_WB, p_WB, self.t_kin,
                                                          poly_to_cert, 'geq', lagrange_mult_degree,
                                                          var_epsilon, base_point_poly)
        plane_polys = np.array(plane_polys_A.tolist() + plane_polys_B.tolist()).squeeze()

        s_prod_pairs = []

        for p in plane_polys:
            prog, (constraint_poly, p_with_tol) = self.putinarPsatConstraint(prog, p, self.t_kin,
                                                                        poly_to_cert,
                                                                        lagrange_mult_degree=lagrange_mult_degree,
                                                                        var_epsilon=var_epsilon)
            s_prod_pairs.append((constraint_poly, p_with_tol))
        return prog, plane_polys, (a_plane_poly, b_plane_poly), dec_vars, s_prod_pairs


    def construct_region_safety_problem(self, poly_to_cert, lagrange_mult_degree=2, var_epsilon=None, check_archimedean=True,
                                        plane_name=""):
        if check_archimedean:
            is_Archimedean, _ = self.certify_archimedean(poly_to_cert, self.t_kin)
            if not is_Archimedean:
                raise ValueError("region is not archimedean")
        prog = MathematicalProgram()
        prog.AddIndeterminates(self.t_kin)
        plane_pairs = {}
        plane_polys_list = []
        s_prod_pairs = {}
        dec_vars = []
        for i, (geomA, geomB) in enumerate(self.pairs):
            print("pair {}/{}".format(i + 1, len(self.pairs)))
            prog, plane_polys, (a_poly, b_poly), dec_vars0, s_prod_pairs0 = self.add_pair_constraint(geomA, geomB,
                                                                                                prog, poly_to_cert,
                                                                                                lagrange_mult_degree=lagrange_mult_degree,
                                                                                                var_epsilon=var_epsilon
                                                                                                )
            plane_polys_list.append(plane_polys)
            plane_pairs[(geomA, geomB)] = (a_poly, b_poly)
            s_prod_pairs[(geomA, geomB)] = s_prod_pairs0
            dec_vars += dec_vars0
        #         if i >= 0:
        #             break
        return prog, plane_pairs, plane_polys_list, np.array(dec_vars), s_prod_pairs


    def certify_region(self, poly_to_cert, lagrange_mult_degree=4, var_epsilon=None, check_archimedean=True, numeric_tol=1e-10):
        prog, plane_pairs, plane_polys_list, dec_vars, s_prod_pairs = self.construct_region_safety_problem(poly_to_cert,
                                                                                                      lagrange_mult_degree=lagrange_mult_degree,
                                                                                                      var_epsilon=var_epsilon,
                                                                                                      check_archimedean=check_archimedean)
        result = self.convSolver.Solve(prog)
        numerically_cert = result.is_success() and (np.linalg.norm(result.GetSolution(dec_vars)) >= numeric_tol)
        print(np.linalg.norm(result.GetSolution(dec_vars)))

        plane_polys = self.extract_planes(plane_pairs, result)
        s_prod_pairs = self.extract_planes(s_prod_pairs, result)
        return result, plane_polys, s_prod_pairs


    def certify_archimedean(self, poly_to_cert, t, lagrange_mult_degree=2):
        # need to fix
        convSolver = MosekSolver()
        prog = MathematicalProgram()
        prog.AddIndeterminates(t)

        t_bar = sym.Polynomial(t @ t)

        poly = t_bar
        A = poly_to_cert.A()
        b = poly_to_cert.b()
        n = b.shape[0]
        for i in range(n):
            lagrange_poly, Q = prog.NewSosPolynomial(sym.Variables(t), lagrange_mult_degree)
            lagrange_poly.SetIndeterminates(sym.Variables(t))
            prog.AddSosConstraint(lagrange_poly)
            poly += lagrange_poly * sym.Polynomial(b[i] - A[i, :] @ t)
        poly.SetIndeterminates(sym.Variables(t))
        prog.AddSosConstraint(poly)
        result = convSolver.Solve(prog)
        return result.is_success(), prog

    def extract_planes(self, plane_pairs, result):
        resulting_plane_pairs = {}

        for k, (a, b) in plane_pairs.items():
            a_list = []
            for ai in a:
                a_list.append(result.GetSolution(ai))
            resulting_plane_pairs[k] = (np.array(a_list), result.GetSolution(b))
        return resulting_plane_pairs


    def extract_s_prod_pairs(self, s_prod_pairs, result):
        evaled_pairs = {}
        for k, poly_list in s_prod_pairs.items():
            cur_list = []
            for (const, vert) in poly_list:
                constraint_poly_evaled = result.GetSolution(const)
                vert_poly_evaled = result.GetSolution(vert)
                cur_list.append((constraint_poly_evaled, vert_poly_evaled))
            evaled_pairs[k] = cur_list
        return evaled_pairs
     