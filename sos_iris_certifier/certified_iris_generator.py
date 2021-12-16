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
import pydrake.symbolic as sym
import iris_utils
import time
from joblib import Parallel, delayed

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
        self.t_variables = sym.Variables(self.forward_kin.t())
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

        self.regions = None
        self.ellipses = None

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
        self.regions = regions
        self.ellipses = ellipses
        self._iris_run_once = True
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
                    success, growth, t_sol = self.GrowthVolumeRational(E,
                                                                  X_WA, X_WB,
                                                                  hpoly_A, hpoly_B,
                                                                  A[:num_faces, :], b[:num_faces] - self._iris_plane_pullback,
                                                                  point)
                    if success:
                        print(f"snopt_example={t_sol}, growth = {growth}")
                        # Add a face to the polytope
                        A[num_faces, :], b[num_faces] = self.TangentPlaneOfEllipse(E,t_sol)
                        num_faces += 1

                        if self._iris_max_faces > 0 and num_faces > self._iris_max_faces+1:
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
            if self._iris_max_faces > 0 and num_faces > self._iris_max_faces+1:
                print("terminating because too many faces")
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
        prog.AddDecisionVariables(self.forward_kin.t())

        if guess is not None:
            prog.SetInitialGuess(self.forward_kin.t(), guess)

        prog.AddLinearConstraint(A, b - np.inf, b, self.forward_kin.t())
        p_AA = prog.NewContinuousVariables(3, "p_AA")
        p_BB = prog.NewContinuousVariables(3, "p_BB")
        setA.AddPointInSetConstraints(prog, p_AA)
        setB.AddPointInSetConstraints(prog, p_BB)
        prog.AddQuadraticErrorCost(E.A().T @ E.A(), E.center(), self.forward_kin.t())

        p_WA = X_WA.multiply(p_AA + 0)
        p_WB = X_WB.multiply(p_BB + 0)
        prog.AddConstraint(eq(p_WA, p_WB))
        result = self.snopt.Solve(prog)

        return result.is_success(), result.get_optimal_cost(), result.GetSolution(self.forward_kin.t())

    def TangentPlaneOfEllipse(self, E,  point):
        a = 2 * E.A().T @ E.A() @ (point - E.center())
        a = a / np.linalg.norm(a)
        b = a.dot(point)
        return a, b
    #endregion

    # region certification
    def _build_certification_problems_from_maps(self, region_to_prog_map, pos_poly_to_lagrange_mult_map,
                                               strict_pos_tol = 1e-5):
        """
        build the certification problem for each region. This step can be parallelized in theory
        but cannot pickle drake objects.
        :param region_to_prog_map:
        :param pos_poly_to_lagrange_mult_map:
        :param strict_pos_tol:
        :return:
        """
        for i, (region, prog) in enumerate(region_to_prog_map.items()):
            t0 = time.time()
            prog = self.add_positive_on_region_contraint_to_prog(prog, region,
                                                            pos_poly_to_lagrange_mult_map,
                                                            strict_pos_tol = strict_pos_tol)
            region_to_prog_map[region] = prog
            t1 = time.time()
            print(f"Region {i+1}/{len(region_to_prog_map.keys())}. Problem built in {t1-t0}secs")
        return region_to_prog_map


    def _initialize_certifier(self, plane_order = -1, penalize_coeffs = True):
        """
        create all the planes and polynomial conditions to certify safe regions. also pre-initializes the lagrange multipliers
        Precondition: iris regions must have been generated once.
        :param plane_order: order of polynomial hyperplane
        :param penalize_coeffs: whether to add penalty on plane weights
        :return:
        """
        if self.regions is None:
            raise ValueError("generate iris regions before attempting certification")

        # create one program for each region
        region_to_prog_map = dict.fromkeys(self.regions)
        max_faces = 0
        for r in region_to_prog_map:
            prog = MathematicalProgram()
            prog.AddIndeterminates(self.forward_kin.t())
            region_to_prog_map[r] = prog
            max_faces = np.max([max_faces, r.b().shape[0]])

        # initialize hyperplane per collision pair
        collision_pair_to_plane_map = dict.fromkeys(self.pairs)
        collision_pair_to_positive_constraint_map = dict.fromkeys(self.pairs)
        for pair in self.pairs:
            (a_poly, b_poly, a_vars, b_vars), (plane_constraint_polynomials_A, plane_constraint_polynomials_B) =\
                self.initialize_separating_hyperplanes_polynomials(*pair, plane_order=-plane_order)
            collision_pair_to_plane_map[pair] = (a_poly, b_poly)
            collision_pair_to_positive_constraint_map[pair] = (plane_constraint_polynomials_A, plane_constraint_polynomials_B)
            # add the plane as a decision variable to the program
            for r in region_to_prog_map:
                self.add_plane_to_prog(region_to_prog_map[r], a_vars, b_vars, penalize_coeffs = penalize_coeffs)

        # initialize the lagrange multipliers by polynomial
        pos_poly_to_lagrange_mult_map = {}
        for (plane_constraint_polynomials_A, plane_constraint_polynomials_B) in collision_pair_to_positive_constraint_map.values():
            poly_list = plane_constraint_polynomials_A.tolist() + plane_constraint_polynomials_B.tolist()
            for p in poly_list:
                pos_poly_to_lagrange_mult_map[p] = self.initialize_N_lagrange_multipliers_by_poly(p, max_faces)
        self.region_to_prog_map = region_to_prog_map
        self.collision_pair_to_plane_map = collision_pair_to_plane_map
        self.collision_pair_to_positive_constraint_map = collision_pair_to_positive_constraint_map
        self.pos_poly_to_lagrange_mult_map = pos_poly_to_lagrange_mult_map
        return region_to_prog_map, collision_pair_to_plane_map, collision_pair_to_positive_constraint_map, pos_poly_to_lagrange_mult_map


    def add_positive_on_region_contraint_to_prog(self, prog, region, pos_poly_to_lagrange_mult_map, strict_pos_tol = 1e-5):
        """
        add the constraint that the polynomials that are the keys in pos_poly_to_lagrange_mult_map are positive
        in the polytopic region
        :param prog: prog to add constraint
        :param region: region positivity is required
        :param pos_poly_to_lagrange_mult_map: dictionary {polynomial -> pre-initialized lagrange multipliers}
        :param strict_pos_tol: p(t) >= strict_pos_tol > 0
        :return:
        """
        t0 = time.time()
        for i, (poly, mults) in enumerate(pos_poly_to_lagrange_mult_map.items()):
            prog = self.add_poly_positive_on_polytope_constraint_with_map_to_prog(prog, poly,
                                                                      region, mults, strict_pos_tol=strict_pos_tol)
        t1 = time.time()
        print(f"Region added in {t1-t0} seconds")
        return prog


    def add_poly_positive_on_polytope_constraint_with_map_to_prog(self, prog, polynomial_to_assert_positive,
                                                          polytope, multiplier_map, strict_pos_tol = 1e-5):
        """
        Putinar's psatz asserts that a polynomial is strictly positive on an Archimedean polytope if and only if it can
        be expressed as p(t) = s_(-1)(t) + \sum_{i=0}^(n-1) s_i(t)(d_i - c^T_i t) where s are SOS
        :param prog:
        :param polynomial_to_assert_positive:
        :param polytope:
        :param strict_pos_tol: tolerance for p(t) >= strict_pos_tol > 0
        :return: the modified program and a dictionary mapping constraint number to multiplier
        """
        assert strict_pos_tol > 0

        C = polytope.A()
        d = polytope.b()

        l, Q = multiplier_map[-1]
        prog.AddDecisionVariables(Q[np.triu_indices(Q.shape[0])])
        prog.AddSosConstraint(l)

        putinar_cone_poly = l
        # build putinar cone poly
        for i in range(C.shape[0]):
            l, Q = multiplier_map[i]
            prog.AddDecisionVariables(Q[np.triu_indices(Q.shape[0])])
            prog.AddSosConstraint(l)
            putinar_cone_poly += l * sym.Polynomial(d[i] - C[i, :] @ self.forward_kin.t())
        putinar_cone_poly.SetIndeterminates(self.t_variables)
        prog.AddEqualityConstraintBetweenPolynomials(putinar_cone_poly, polynomial_to_assert_positive - strict_pos_tol)
        return prog
    #
    # def add_poly_positive_on_polytope_constraint_by_basis(self, prog, polynomial_to_assert_positive,
    #                                                       polytope, strict_pos_tol = 1e-5):
    #     """
    #     Putinar's psatz asserts that a polynomial is strictly positive on an Archimedean polytope if and only if it can
    #     be expressed as p(t) = s_(-1)(t) + \sum_{i=0}^(n-1) s_i(t)(d_i - c^T_i t) where s are SOS
    #     :param prog:
    #     :param polynomial_to_assert_positive:
    #     :param polytope:
    #     :param strict_pos_tol: tolerance for p(t) >= strict_pos_tol > 0
    #     :return: the modified program and a dictionary mapping constraint number to multiplier
    #     """
    #     assert strict_pos_tol > 0
    #     s_min1_basis = iris_utils.sparsest_sos_basis_poly(polynomial_to_assert_positive)
    #     # TODO reduce this basis by degree 1
    #     si_basis = iris_utils.sparsest_sos_basis_poly(polynomial_to_assert_positive)
    #     C = polytope.A()
    #     d = polytope.b()
    #     multiplier_map = dict.fromkeys([-1] + [i for i in range(C.shape[0])])
    #
    #     l, Q = prog.NewSosPolynomial(s_min1_basis)
    #     l.SetIndeterminates(self.t_variables)
    #     multiplier_map[-1] = l
    #     prog.AddSosConstraint(l)
    #
    #     putinar_cone_poly = l
    #     # build putinar cone poly
    #     for i in range(C.shape[0]):
    #         l, Q = prog.NewSosPolynomial(si_basis)
    #         l.SetIndeterminates(self.t_variables)
    #         multiplier_map[i] = l
    #         prog.AddSosConstraint(l)
    #         putinar_cone_poly += l*(d[i]-C[i,:]@self.t_variables)
    #     putinar_cone_poly.SetIndeterminates(self.t_variables)
    #     prog.AddEqualityConstraintBetweenPolynomials(putinar_cone_poly, polynomial_to_assert_positive - strict_pos_tol)
    #     return prog, multiplier_map

    def construct_separating_hyperplane_by_basis(self, basis, plane_name=''):
        """
        constructs a hyperplane a(t)^T x + b(t) = 0 for t the indeterminate of the plant dynamics
        :param prog: program to add the plane
        :param basis: basis of the variable t to parametrize the polynomial plane
        :param plane_name: name of plane if desired
        :param penalize_coeffs: whether to add the quadratic cost (a,b)^TQ(a,b) to the program
        :return: program and tuple of the coefficient (a(t), b(t))
        """
        basis = np.array([sym.Polynomial(v) for v in basis])

        a_coeffs, a_vars = iris_utils.NewContinuousVariablesWithoutProg(3, basis.shape[0], 'a_coeffs' + plane_name)
        a_poly = a_coeffs @ basis

        b_coeffs, b_vars = iris_utils.NewContinuousVariablesWithoutProg(1, basis.shape[0], 'b_coeffs' + plane_name)
        b_poly = (b_coeffs @ basis).item()

        for i, p in enumerate(a_poly):
            a_poly[i].SetIndeterminates(self.t_variables)
        b_poly.SetIndeterminates(self.t_variables)
        return a_poly, b_poly, a_vars, b_vars

    def construct_separating_hyperplane_by_order(self, order, plane_name='', subset = None):
        """
        shortcute for constructing separating hyperplane with dense monomial basis of order order.
        see construct_separating_hyperplane_by_basis for parameter function
        :param order: order of dense monomial basis
        :param subset: subset of variables t to use in dense monomial basis
        :return: program and tuple of the coefficient (a(t), b(t))
        """
        t = self.t_variables[subset] if subset is not None else self.t_variables
        basis = sym.MonomialBasis(t, order)
        return self.construct_separating_hyperplane_by_basis(basis, plane_name=plane_name)

    def makeBodyHyperplaneSidePolynomial(self, a_poly, b_poly,
                                          VPoly, R_WA, p_WA, leq_or_geq):
        """
        create one polynomial per vertex in VPoly such that p_i(t) >= 0 implies that the vertex is on a given side of
        the plane
        :param a_poly: a of the hyperplane a^T x + b =0
        :param b_poly: b of the hyperplane a^T x + b =0
        :param VPoly: VPolytope objects which defines the vertices of collision geometry A in the frame A
        :param R_WA: Forward kinematics rotation matrix as a multilinear polynomial in variables cos(q_i), sin(q_i)
        :param p_WA: Forward kinematics translation matrix as a multilinear polynomial in variables cos(q_i), sin(q_i)
        :param leq_or_geq: whether to use side a^Tx + b <= -1 or a^Tx + b >= 1
        :return: an array of polynomials whose positivity implies the given side of plane
        """
        num_verts = VPoly.vertices().shape[1]

        vertex_pos = R_WA @ (VPoly.vertices()) + np.repeat(p_WA[:, np.newaxis], num_verts, 1)

        dens = np.ndarray(shape=vertex_pos.shape, dtype=object)
        nums = np.ndarray(shape=vertex_pos.shape, dtype=object)
        for i, row in enumerate(vertex_pos):
            for j, v in enumerate(row):
                vertex_pos[i, j] = self.forward_kin.ConvertMultilinearPolynomialToRationalFunction(v)
                dens[i, j] = vertex_pos[i, j].denominator()
                nums[i, j] = vertex_pos[i, j].numerator()

        unique_dens_c = [[sym.Polynomial(1)] for _ in range(dens.shape[1])]
        col_den = [sym.Polynomial(1) for _ in range(dens.shape[1])]
        for c in range(dens.shape[1]):
            for r in range(dens.shape[0]):
                is_unique = True
                for d in unique_dens_c[c]:
                    is_unique = False if dens[r, c].EqualTo(d) else True
                    if not is_unique:
                        break
                if is_unique:
                    unique_dens_c[c].append(dens[r, c])

            for d in unique_dens_c[c]:
                col_den[c] *= d

        for c in range(nums.shape[1]):
            for r in range(nums.shape[0]):
                for d in unique_dens_c[c]:
                    if not dens[r, c].EqualTo(d):
                        nums[r, c] *= d

        plane_polys_per_vertex = np.array([None for _ in range(vertex_pos.shape[1])])
        for i in range(vertex_pos.shape[1]):
            if leq_or_geq == 'leq':
                # a^Tx + b <= -1 -> -(a^Tx+b+1)>=0
                plane_polys_per_vertex[i] = -a_poly.dot(nums[:, i]) - (b_poly + 1) * (col_den[i])

            elif leq_or_geq == 'geq':
                # a^Tx+b >= 1 -> a^Tx+b -1 >= 0
                plane_polys_per_vertex[i] = a_poly.dot(nums[:, i]) + (b_poly - 1) * (col_den[i])
            else:
                raise ValueError("leq_or_geq arg must be leq or geq not {}".format(leq_or_geq))
            plane_polys_per_vertex[i].SetIndeterminates(self.t_variables)

        return plane_polys_per_vertex

    def initialize_separating_hyperplanes_polynomials(self, geomA, geomB, plane_order = -1):
        VPolyA, VPolyB = self.vpoly_sets_in_self_frame_by_geom_id[geomA], self.vpoly_sets_in_self_frame_by_geom_id[geomB]

        R_WA, p_WA = self.X_WA_multilinear_list[int(self.body_indexes_by_geom_id[geomA])]
        R_WB, p_WB = self.X_WA_multilinear_list[int(self.body_indexes_by_geom_id[geomB])]

        if plane_order >= 0:
            a_poly, b_poly, a_vars, b_vars = self.construct_separating_hyperplane_by_order(plane_order)
        else:
            # TODO use only the subset of the variables that are currently present
            basis = GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(self.t_variables)
            a_poly, b_poly, a_vars, b_vars = self.construct_separating_hyperplane_by_basis(basis)
        plane_constraint_polynomials_A = self.makeBodyHyperplaneSidePolynomial(a_poly, b_poly,
                                          VPolyA, R_WA, p_WA, 'leq')
        plane_constraint_polynomials_B = self.makeBodyHyperplaneSidePolynomial(a_poly, b_poly,
                                                                               VPolyB, R_WB, p_WB, 'leq')
        return (a_poly, b_poly, a_vars, b_vars), (plane_constraint_polynomials_A, plane_constraint_polynomials_B)

    def initialize_N_lagrange_multipliers_by_basis(self, basis, num_multipliers, s_min1_basis = None):
        """
        :param basis: basis of the lagrange multiplier
        :param num_multipliers: number of multipliers
        :param s_min1_basis: basis of the constant term if different
        :return:
        """
        dummy_prog = MathematicalProgram()
        s_min1_basis = s_min1_basis if s_min1_basis is not None else basis
        multiplier_map = dict.fromkeys([-1] + [i for i in range(num_multipliers)])
        l, Q = dummy_prog.NewSosPolynomial(s_min1_basis)
        l.SetIndeterminates(self.t_variables)
        multiplier_map[-1] = (l,Q)
        for i in range(num_multipliers):
            l, Q = dummy_prog.NewSosPolynomial(basis)
            l.SetIndeterminates(self.t_variables)
            multiplier_map[i] = (l,Q)
        return multiplier_map

    def initialize_N_lagrange_multipliers_by_poly(self, poly, num_multipliers):
        basis = iris_utils.sparsest_sos_basis_poly(poly)
        return self.initialize_N_lagrange_multipliers_by_basis(basis, num_multipliers)

    def add_plane_to_prog(self, prog, a_coeffs, b_coeffs, penalize_coeffs = True):
        prog.AddDecisionVariables(a_coeffs)
        prog.AddDecisionVariables(b_coeffs)

        if penalize_coeffs:
            ntmp = a_coeffs.shape[0] + b_coeffs.shape[0]
            Qtmp, btmp = np.eye(ntmp), np.zeros(ntmp)
            prog.AddQuadraticCost(Qtmp, btmp, np.concatenate([a_coeffs, b_coeffs]), is_convex=True)
        return prog


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

        self.link_poses_by_body_index_multilinear= self.forward_kin.CalcLinkPosesAsMultilinearPolynomials(
            self.q_star, self.plant.world_body().index())

        self.X_WA_multilinear_list = [(r.rotation().copy(), r.translation().copy()) for r in
                                 self.link_poses_by_body_index_multilinear]

    def _construct_collision_candidates(self):
        self.pairs = self.inspector.GetCollisionCandidates()

        # only gets kProximity pairs. Might be more efficient?
        # geom_ids = inspector.GetGeometryIds(GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity)
        pair_set = set()
        for p in self.pairs:
            pair_set.add(p[0])
            pair_set.add(p[1])
        self.geom_ids = self.inspector.GetGeometryIds(GeometrySet(list(pair_set)))

        self.hpoly_sets_in_self_frame_by_geom_id = {geom: iris_utils.MakeFromHPolyhedronSceneGraph(self.query, geom, self.inspector.GetFrameId(geom))
                                                    for geom in self.geom_ids}
        self.vpoly_sets_in_self_frame_by_geom_id = {geom: iris_utils.MakeFromVPolytopeSceneGraph(self.query, geom, self.inspector.GetFrameId(geom))
                                                    for geom in self.geom_ids}
        self.body_indexes_by_geom_id = {geom:
                                       self.plant.GetBodyFromFrameId(self.inspector.GetFrameId(geom)).index() for geom in
                                   self.geom_ids}