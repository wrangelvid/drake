import numpy as np
from pydrake.geometry import SceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.common import FindResourceOrThrow
from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import Parser, LoadModelDirectives, ProcessModelDirectives
from pydrake.all import GeometrySet
from functools import partial
from pydrake.all import SnoptSolver, MosekSolver, eq, MathematicalProgram, Hyperellipsoid, HPolyhedron, VPolytope
from pydrake.all import RationalForwardKinematics, FindBodyInTheMiddleOfChain
from pydrake.all import GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo, GenerateMonomialBasisWithOrderUpToOne
from iris_utils import MakeFromHPolyhedronSceneGraph, MakeFromVPolytopeSceneGraph
import time

class CertifiedIrisRegionGenerator():
    def __init__(self, diagram, plant, scene_graph, **kwargs):
        self.diagram = diagram
        self.plant = plant
        self.scene_graph = scene_graph

        # Set up drake necessary queries
        self.parser = Parser(plant)
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = plant.GetMyContextFromRoot(self.diagram_context)
        self.diagram.Publish(self.diagram_context)
        self.query = self.scene_graph.get_query_output_port().Eval(self.scene_graph.GetMyContextFromRoot(self.diagram_context))
        self.inspector = self.query.inspector()

        # Construct Rational Forward Kinematics
        self.forward_kin = RationalForwardKinematics(plant)
        # the point around which we construct the stereographic projection
        self.q_star = kwargs.get('q_star', np.zeros(self.plant.num_positions()))
        self.q_lower_limits = plant.GetPositionLowerLimits()
        self.t_lower_limits = self.forward_kin.ComputeTValue(self.q_lower_limits, self.q_star)
        self.q_upper_limits = plant.GetPositionUpperLimits()
        self.t_upper_limits = self.forward_kin.ComputeTValue(self.q_upper_limits, self.q_star)

        # initialize Solvers
        self.snopt = SnoptSolver()
        self.mosek = MosekSolver()

        #handle kwargs
        self._handle_kwargs(**kwargs)

        #construct transformation logistics
        self._initialize_transform_logistics()
        self._construct_collision_candidates()

    #region IRIS
    def iris_in_rational_space(self, seed_points, termination_threshold=2e-2, iteration_limit=100):
        "pass seed points with row as coordinates"
        seed_points = np.atleast_2d(seed_points)
        regions = []
        ellipses = []
        for i in range(seed_points.shape[0]):
            start_time = time.time()
            hpoly, ell = self._iris_rational_space(seed_points[i, :], require_containment_points=[seed_points[i, :]],
                                                    termination_threshold = termination_threshold,
                                                    iteration_limit=iteration_limit)
            print("Time: %6.2f \tVolume: %6.2f \tCenter:" % (time.time() - start_time, ell.Volume()),
                  ell.center(), flush=True)
            regions.append(hpoly)
            ellipses.append(ell)
        return regions, ellipses


    def _iris_rational_space(self, point, require_containment_points=None, termination_threshold=2e-2,
                            iteration_limit=100):
        require_containment_points = require_containment_points if require_containment_points is not None else []
        E = Hyperellipsoid(np.eye(3) / self._iris_starting_ellipse_vol, point)
        best_volume = E.Volume()

        P = HPolyhedron.MakeBox(self.t_lower_limits, self.t_upper_limits)
        A = np.vstack((P.A(), np.zeros((self._iris_default_num_faces * len(self.pairs), 3))))
        b = np.concatenate((P.b(), np.zeros(self._iris_default_num_faces * len(self.pairs))))

        iteration = 0
        num_faces = P.A().shape[0]
        while True:
            ## Find separating hyperplanes

            for geomA, geomB in self.pairs:
                print(f"geomA={self.inspector.GetName(geomA)}, geomB={self.inspector.GetName(geomB)}")
                # Run snopt at the beginning
                while True:
                    X_WA = self.X_WA_list[int(self.body_indexes_by_geom_id[geomA])]
                    X_WB = self.X_WA_list[int(self.body_indexes_by_geom_id[geomB])]
                    hpoly_A = self.hpoly_sets_in_self_frame_by_geom_id[geomA]
                    hpoly_B = self.hpoly_sets_in_self_frame_by_geom_id[geomB]
                    success, growth, qstar = self.GrowthVolumeRational(E,
                                                                  X_WA, X_WB,
                                                                  hpoly_A, hpoly_B,
                                                                  A[:num_faces, :], b[:num_faces] - self._iris_plane_pullback,
                                                                  point)
                    if success:
                        print(f"snopt_example={qstar}, growth = {growth}")
                        # Add a face to the polytope
                        A[num_faces, :], b[num_faces] = E.TangentPlane(qstar)
                        num_faces += 1
                        if self._iris_max_faces > 0 and num_faces > self._iris_max_faces:
                            break
                        if num_faces >= A.shape[0]:
                            # double number of faces if we exceed the number of faces preallocated
                            A = np.vstack((A, np.zeros((A.shape[0], 3))))
                            b = np.concatenate((P.b(), np.zeros(self._iris_default_num_faces * len(self.pairs))))
                    else:
                        break


            if any([np.any(A[:num_faces, :] @ p > b[:num_faces]) for p in require_containment_points]):
                print("terminating because a required containment point would have not been contained")
                break

            P = HPolyhedron(A[:num_faces, :], b[:num_faces])
            E = P.MaximumVolumeInscribedEllipsoid()
            print(iteration)

            iteration += 1
            if iteration >= iteration_limit:
                break

            volume = E.Volume()
            if volume - best_volume <= termination_threshold:
                break
            best_volume = volume
        return P, E


    def GrowthVolumeRational(self, E, X_WA, X_WB, setA, setB, A, b, guess=None):
        prog = MathematicalProgram()
        prog.AddDecisionVariables(self.forward_kin.t)

        if guess is not None:
            prog.SetInitialGuess(self.forward_kin.t, guess)

        prog.AddLinearConstraint(A, b - np.inf, b, self.forward_kin.t)
        p_AA = prog.NewContinuousVariables(3, "p_AA")
        p_BB = prog.NewContinuousVariables(3, "p_BB")
        setA.AddPointInSetConstraints(prog, p_AA)
        setB.AddPointInSetConstraints(prog, p_BB)
        prog.AddQuadraticErrorCost(E.A().T @ E.A(), E.center(), self.forward_kin.t)

        p_WA = X_WA.multiply(p_AA + 0)
        p_WB = X_WB.multiply(p_BB + 0)
        prog.AddConstraint(eq(p_WA, p_WB))
        result = self.snopt.Solve(prog)

        return result.is_success(), result.get_optimal_cost(), result.GetSolution(t)

    def TangentPlaneOfEllipse(self, E,  point):
        a = 2 * E.A().T @ E.A() @ (point - E.center())
        a = a / np.linalg.norm(a)
        b = a.dot(point)
        return a, b
    #endregion

    def _handle_kwargs(self, **kwargs):
        self._iris_starting_ellipse_vol = kwargs.get('iris_starting_ellipse_vol', 1e-3)

        self._iris_plane_pullback = kwargs.get('iris_plane_pullback', 1e-5)

        self._iris_max_faces = kwargs.get('iris_max_faces', 10)
        self._iris_default_num_faces = kwargs.get('iris_default_num_face', self._iris_max_faces if self._iris_max_faces > 0 else 10)

    def _initialize_transform_logistics(self):

        #TODO
        #currently, we do everything in world pose. As hongkai points out this can make very high
        #degree polynomials. We should exploit his FindBodyInTheMiddleOfChain method
        self.link_poses_by_body_index_rat_pose = self.forward_kin.CalcLinkPoses(self.q_star,
                                                                      self.plant.world_body().index())
        self.X_WA_list = [p.asRigidTransformExpr() for p in self.link_poses_by_body_index_rat_pose]

    def _construct_collision_candidates(self):
        self.pairs = self.inspector.GetCollisionCandidates()

        # only gets kProximity pairs. Might be more efficient?
        # geom_ids = inspector.GetGeometryIds(GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity)
        pair_set = set()
        for p in self.pairs:
            pair_set.add(p[0])
            pair_set.add(p[1])
        self.geom_ids = self.inspector.GetGeometryIds(GeometrySet(list(pair_set)))

        self.hpoly_sets_in_self_frame_by_geom_id = {geom: MakeFromHPolyhedronSceneGraph(self.query, geom, self.inspector.GetFrameId(geom)) for geom in self.geom_ids}
        self.body_indexes_by_geom_id = {geom:
                                       self.plant.GetBodyFromFrameId(self.inspector.GetFrameId(geom)).index() for geom in
                                   self.geom_ids}