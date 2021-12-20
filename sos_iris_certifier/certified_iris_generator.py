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
from contextlib import contextmanager
import logging


do_timing = True
@contextmanager
def _log_time_usage(prefix=""):
    '''log the time usage in a code block
    prefix: the prefix text to show
    '''
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        elapsed_seconds = float("%.4f" % (end - start))
        if do_timing:
            print(f'{prefix}: {elapsed_seconds}')
        # logging.debug('%s: elapsed seconds: %s', prefix, elapsed_seconds)

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

    def _initalize_certifier(self, plane_order = -1, strict_pos_tol = 1e-5, penalize_coeffs = True):
        if self.regions is None:
            raise ValueError("generate iris regions before attempting certification")
        self.certification_problems = dict.fromkeys(self.regions)
        for region in self.regions:
            self.certification_problems[region] = self._initialize_certifier_for_region(region, plane_order,
                                                                                        strict_pos_tol, penalize_coeffs)
            print()
        return self.certification_problems

    def _initialize_certifier_for_region(self, region, plane_order = -1,
                                         strict_pos_tol = 1e-5, penalize_coeffs = True):
        with _log_time_usage("Time to initialize region program"):
            num_faces = region.A().shape[0]

            certify_for_fixed_eps_prog = MathematicalProgram()
            certify_for_fixed_multiplier_prog = MathematicalProgram()

            with _log_time_usage("time to create maps: "):
                # maps a collision pair to a plane (a(t), b(t)) that will certify its collision freeness
                collision_pair_to_plane_variable_map = dict.fromkeys(self.pairs)
                collision_pair_to_plane_poly_map = dict.fromkeys(self.pairs)

                # maps a geom id and vertex number to a polynomial which must be positive to be certified and a set of multipliers
                geom_id_and_vert_to_pos_poly_and_multiplier_variable_map = {}

                for geomA, geomB in self.pairs:
                    (a_poly, b_poly, a_vars, b_vars), (plane_constraint_polynomials_A, plane_constraint_polynomials_B) = \
                        self.initialize_separating_hyperplanes_polynomials(geomA, geomB, plane_order=plane_order)
                    collision_pair_to_plane_variable_map[(geomA, geomB)] = (a_vars, b_vars)
                    collision_pair_to_plane_poly_map[(geomA, geomB)] = (a_poly, b_poly)
                    for i, p in enumerate(plane_constraint_polynomials_A):
                        multipliers_map = self.initialize_N_lagrange_multipliers_by_poly(p, num_faces)
                        geom_id_and_vert_to_pos_poly_and_multiplier_variable_map[(geomA, i)] = (p, multipliers_map)
                    for i, p in enumerate(plane_constraint_polynomials_B):
                        multipliers_map = self.initialize_N_lagrange_multipliers_by_poly(p, num_faces)
                        geom_id_and_vert_to_pos_poly_and_multiplier_variable_map[(geomB, i)] = (p, multipliers_map)
                    # add the plane as a decision variable to the program
                    # self.add_plane_to_prog(certify_for_fixed_eps_prog, a_vars, b_vars, penalize_coeffs=penalize_coeffs)
                    # self.add_plane_to_prog(certify_for_fixed_multiplier_prog, a_vars, b_vars, penalize_coeffs=False)

        #TODO construct problems
        with _log_time_usage("time to create Region Certification Problem: "):
            certification_problem = RegionCertificationProblem(
                region, certify_for_fixed_eps_prog, certify_for_fixed_multiplier_prog,
                collision_pair_to_plane_variable_map, collision_pair_to_plane_poly_map,
                geom_id_and_vert_to_pos_poly_and_multiplier_variable_map,
                self.forward_kin.t(), self.mosek, strict_pos_tol, penalize_coeffs
            )

        return certification_problem


    #TODO modify this
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


    def add_poly_positive_on_polytope_constraint_with_map_to_prog(self, prog, polytope, polynomial_to_assert_positive,
                                                           multiplier_map, strict_pos_tol = 1e-5):
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
                plane_polys_per_vertex[i] = -(a_poly.dot(nums[:, i]) + (b_poly + 1) * (col_den[i]))

            elif leq_or_geq == 'geq':
                # a^Tx+b >= 1 -> a^Tx+b -1 >= 0
                plane_polys_per_vertex[i] = a_poly.dot(nums[:, i]) + (b_poly - 1) * (col_den[i])
            else:
                raise ValueError("leq_or_geq arg must be leq or geq not {}".format(leq_or_geq))
            plane_polys_per_vertex[i].SetIndeterminates(self.t_variables)

        return plane_polys_per_vertex

    def initialize_separating_hyperplanes_polynomials(self, geomA, geomB, plane_order = -1):
        VPolyA = self.vpoly_sets_in_self_frame_by_geom_id[geomA]
        VPolyB = self.vpoly_sets_in_self_frame_by_geom_id[geomB]

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
                                                                               VPolyB, R_WB, p_WB, 'geq')
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

class RegionCertificationProblem:
    # ret = {
    #     'certify_for_fixed_eps_prog': certify_for_fixed_eps_prog,
    #     'certify_for_fixed_multiplier_prog': certify_for_fixed_multiplier_prog,
    #     'collision_pair_to_plane_variable_map': collision_pair_to_plane_variable_map,
    #     'geom_id_and_vert_to_pos_poly_and_multiplier_variable_map': geom_id_and_vert_to_pos_poly_and_multiplier_variable_map,
    #     'cur_eps_value': 0
    # }

    def __init__(self, region, certify_for_fixed_eps_prog,
                 certify_for_fixed_multiplier_prog,
                 collision_pair_to_plane_variable_map,
                 collision_pair_to_plane_poly_map,
                 geom_id_and_vert_to_pos_poly_and_multiplier_variable_map,
                 t_array,
                 solver,
                 strict_pos_tol = 1e-5,
                 penalize_plane_coeffs = True):

        self.certify_for_fixed_eps_prog = certify_for_fixed_eps_prog
        self.certify_for_fixed_multiplier_prog = certify_for_fixed_multiplier_prog
        self.t_array = t_array
        self.t_variables = sym.Variables(t_array)

        self.certify_for_fixed_eps_prog.AddIndeterminates(self.t_array)
        self.certify_for_fixed_multiplier_prog.AddIndeterminates(self.t_array)

        self.collision_pair_to_plane_variable_map = collision_pair_to_plane_variable_map
        self.collision_pair_to_plane_poly_map = collision_pair_to_plane_poly_map
        self.collision_pair_to_plane_result_map = dict.fromkeys(self.collision_pair_to_plane_poly_map.keys())

        self.geom_id_and_vert_to_pos_poly_and_multiplier_variable_map = geom_id_and_vert_to_pos_poly_and_multiplier_variable_map
        self.geom_id_and_vert_to_pos_poly_and_multiplier_result_map = dict.fromkeys(geom_id_and_vert_to_pos_poly_and_multiplier_variable_map.keys())

        self.penalize_plane_coeffs = penalize_plane_coeffs

        self.region = region
        self.strict_pos_tol = strict_pos_tol
        self.num_faces = region.b().shape[0]

        self.cur_eps = np.zeros(self.num_faces)
        self.fixed_epsilon_result = None
        self.fixed_multiplier_result = None

        self.var_eps = self.certify_for_fixed_multiplier_prog.NewContinuousVariables(self.num_faces, 'eps')

        self._prepare_fixed_epsilon_problem()
        self._prepare_fixed_multiplier_problem()

        self.solver = solver

    @property
    def cur_eps(self):
        return self._cur_eps

    @cur_eps.setter
    def cur_eps(self, val):
        self._cur_eps = val

    def _extract_planes(self, result):
        for k, (a, b) in self.collision_pair_to_plane_poly_map.items():
            a_list = []
            for ai in a:
                a_list.append(result.GetSolution(ai))
            self.collision_pair_to_plane_result_map[k] = (np.array(a_list), result.GetSolution(b))

    def _extract_multipliers(self, result):
        for k, pos_poly_multiplier_dict in self.geom_id_and_vert_to_pos_poly_and_multiplier_variable_map.items():
            multiplier_dict = pos_poly_multiplier_dict[1]
            multiplier_result_dict = multiplier_dict.fromkeys(multiplier_dict.keys())
            pos_poly_result = result.GetSolution(pos_poly_multiplier_dict[0])
            for i, multiplier in multiplier_dict.items():
                multiplier_result_dict[i] = result.GetSolution(multiplier[0]), result.GetSolution(multiplier[1]),

            self.geom_id_and_vert_to_pos_poly_and_multiplier_result_map[k] = (pos_poly_result, multiplier_result_dict)

    def attempt_certification_for_epsilon(self, epsilon):
        """
        builds the certification problem for a fixed epsilon.
        If the problem solves: return True, collision_pair -> plane, (geomId, vert_number) -> multiplier_dict
        else: return False, None, None
        :param epsilon:
        :return:
        """
        self.build_problem_for_epsilon(epsilon)
        self.fixed_epsilon_result = self.solver.Solve(self.certify_for_fixed_eps_prog)
        if self.fixed_epsilon_result.is_success():
            self._extract_planes(self.fixed_epsilon_result)
            self._extract_multipliers(self.fixed_epsilon_result)
            return self.fixed_epsilon_result.is_success(),\
                   self.collision_pair_to_plane_result_map,\
                   self.geom_id_and_vert_to_pos_poly_and_multiplier_result_map
        return False, None, None

    def build_problem_for_epsilon(self, epsilon):
        """
        Precondition: _prepare_fixed_epsilon_problem must be called before.
        Completes the construction of the rhs of p(t) = s_(-1)(t) + \sum_{i=0}^(n-1) s_i(t)(d_i - eps - c^T_i t).
        Removes any existing equality constraints of the above form and replaces it with the newly constructed equality
        above.
        :param epsilon:
        """
        if len(epsilon.shape) != 1:
            raise ValueError(f"epsilon wrong shape. Must be 1D, is {len(epsilon.shape)}D")
        if epsilon.shape[0] != self.num_faces:
            raise ValueError(f"epsilon wrong shape expected {self.num_faces} for {epsilon.shape[0]}" )

        self.geom_id_and_vert_to_pos_poly_and_putinar_equality = dict.fromkeys(self.geom_id_and_vert_to_pos_poly_and_putinar_preallocated_cone.keys())
        for geom_id_and_vert, poly_and_putinar_preallocated in self.geom_id_and_vert_to_pos_poly_and_putinar_preallocated_cone.items():
            if self.geom_id_and_vert_to_positive_poly_eq_constraints[geom_id_and_vert] is not None:
                for constraint in self.geom_id_and_vert_to_positive_poly_eq_constraints[geom_id_and_vert]:
                    self.certify_for_fixed_eps_prog.RemoveConstraint(constraint
                        )
            poly, putinar_cone_poly = poly_and_putinar_preallocated[0], poly_and_putinar_preallocated[1]
            multiplier_map = self.geom_id_and_vert_to_pos_poly_and_multiplier_variable_map[geom_id_and_vert][1]
            # check that this isn't going to change the value of putinar preallocated
            for i in range(epsilon.shape[0]):
                if np.abs(epsilon[i]) > 1e-5:
                    putinar_cone_poly += -epsilon[i]*multiplier_map[i][0]
            putinar_cone_poly.SetIndeterminates(self.t_variables)

            self.geom_id_and_vert_to_positive_poly_eq_constraints[geom_id_and_vert] =\
                self.ManualAddEqualityConstraintsBetweenPolynomials(self.certify_for_fixed_eps_prog,
                                                                    putinar_cone_poly,
                                                                    poly - self.strict_pos_tol)
            self.geom_id_and_vert_to_pos_poly_and_putinar_equality[geom_id_and_vert] = (poly - self.strict_pos_tol, putinar_cone_poly)

    def _prepare_fixed_epsilon_problem(self):
        """
        Putinar's psatz asserts that a polynomial is strictly positive on an Archimedean polytope Ct <= d-eps if and
        only if it can be expressed as p(t) = s_(-1)(t) + \sum_{i=0}^(n-1) s_i(t)(d_i - eps - c^T_i t) where s are SOS.
        This method creates the term on the right hand side of the equation with eps = 0 and adds all the multipliers
        to the program certify_for_fixed_eps_prog. It does NOT add the equality constraint as this will be added when
        a eps is provided later.

        This method also adds the plane variables to the program certify_for_fixed_eps_prog

        In this method we construct:
        self.geom_id_and_vert_to_pos_poly_and_putinar_preallocated_cone with pairs:
         (geom_id, vertex number) -> (lhs, rhs (with eps = 0))
         self.geom_id_and_vert_to_positive_poly_eq_constraint with pairs:
         (geom_id, vertex number) -> None
         the latter will hold the constraints of the program so we can remove them later and re-use in the future
        """
        # add plane variables to problem
        for (a_coeff, b_coeff) in self.collision_pair_to_plane_variable_map.values():
            self._add_plane_to_prog(self.certify_for_fixed_eps_prog, a_coeff, b_coeff,
                                    penalize_coeffs=self.penalize_plane_coeffs)

        # create putinar cone
        C = self.region.A()
        d = self.region.b()
        self.geom_id_and_vert_to_pos_poly_and_putinar_preallocated_cone = \
            dict.fromkeys(self.geom_id_and_vert_to_pos_poly_and_multiplier_variable_map.keys())
        self.geom_id_and_vert_to_positive_poly_eq_constraints = \
            dict.fromkeys(self.geom_id_and_vert_to_pos_poly_and_multiplier_variable_map.keys())

        for geom_id_and_vert, poly_and_mult in self.geom_id_and_vert_to_pos_poly_and_multiplier_variable_map.items():
            (poly, multiplier_map) = poly_and_mult[0], poly_and_mult[1]
            l, Q = multiplier_map[-1]
            self.certify_for_fixed_eps_prog.AddDecisionVariables(Q[np.triu_indices(Q.shape[0])])
            self.certify_for_fixed_eps_prog.AddSosConstraint(l)

            putinar_cone_poly = l
            # build putinar cone poly
            for i in range(C.shape[0]):
                l, Q = multiplier_map[i]
                self.certify_for_fixed_eps_prog.AddDecisionVariables(Q[np.triu_indices(Q.shape[0])])
                self.certify_for_fixed_eps_prog.AddSosConstraint(l)
                putinar_cone_poly += l * sym.Polynomial(d[i] - C[i, :] @ self.t_array)
            putinar_cone_poly.SetIndeterminates(self.t_variables)
            self.geom_id_and_vert_to_pos_poly_and_putinar_preallocated_cone[geom_id_and_vert] = \
                (poly, putinar_cone_poly)

            # self.geom_id_and_vert_to_positive_poly_eq_constraint[geom_id_and_vert] = \

    def attempt_reduce_eps_in_certificate(self, geom_id_and_vert_to_pos_poly_and_multiplier_map):
        """
        for a fixed set of multipliers, try to reduce epsilon as much as possible. This program should always be feasible
        if the multipliers constitute a valid certificate
        :param geom_id_and_vert_to_pos_poly_and_multiplier_map: (geom_id, vertex_number) ->
        (polynomial defined by plane, multiplier certifying positivity)
        :return: True, (collision_pair) -> (a(t), b(t)), certified_region, pos_poly_to_multiplier_map certificate
        """
        self.build_fixed_multiplier_problem(geom_id_and_vert_to_pos_poly_and_multiplier_map)
        self.fixed_multiplier_result = self.solver.Solve(self.certify_for_fixed_multiplier_prog)
        if self.fixed_multiplier_result.is_success():
            self._extract_planes(self.fixed_multiplier_result)
            new_region = self._extract_new_region(self.fixed_multiplier_result)
            return self.fixed_epsilon_result.is_success(), \
                   self.collision_pair_to_plane_result_map, \
                   new_region, \
                   geom_id_and_vert_to_pos_poly_and_multiplier_map
        raise ValueError("program was infeasible. Are the multipliers valid?")
        return False, None, None

    def build_fixed_multiplier_problem(self, geom_id_and_vert_to_pos_poly_and_multiplier_map):
        """
        Precondition: call _prepare_fixed_multiplier_problem first
        :return:
        """
        C = self.region.A()
        d = self.region.b()
        for geom_id_and_vert, pos_poly_and_multiplier_map in geom_id_and_vert_to_pos_poly_and_multiplier_map.items():
            if self.geom_id_and_vert_to_positive_poly_eq_constraint_fixed_multiplier[geom_id_and_vert] is not None:
                self.certify_for_fixed_multiplier_prog.RemoveConstraint(
                    self.geom_id_and_vert_to_positive_poly_eq_constraint_fixed_multiplier[geom_id_and_vert])
            pos_poly, multiplier_map = pos_poly_and_multiplier_map[0], pos_poly_and_multiplier_map[1]
            rhs = multiplier_map[-1]
            for i in range(self.var_eps.shape[0]):
                e = self.var_eps[i]
                rhs += multiplier_map[i]*sym.Polynomial(d[i]-e-C[i,:]@self.t_array)
            self.geom_id_and_vert_to_positive_poly_eq_constraint_fixed_multiplier[geom_id_and_vert] = \
                self.ManualAddEqualityConstraintsBetweenPolynomials(self.certify_for_fixed_multiplier_prog,
                                                                    pos_poly, rhs)

    def _prepare_fixed_multiplier_problem(self):
        """
        add plane coeffs as variables in program certify_for_fixed_multipliers. Adds quadratic cost on epsilon to program.
        creates empty dictionary of (geomId, vertexNumber) -> constraint_for_fixed_multiplier
        :return:
        """
        # add plane variables to problem
        for (a_coeff, b_coeff) in self.collision_pair_to_plane_variable_map.values():
            self._add_plane_to_prog(self.certify_for_fixed_multiplier_prog, a_coeff, b_coeff,
                                    penalize_coeffs=False)
        self.certify_for_fixed_multiplier_prog.AddQuadraticCost(self.var_eps.dot(self.var_eps))
        self.geom_id_and_vert_to_positive_poly_eq_constraint_fixed_multiplier = dict.fromkeys(
            self.geom_id_and_vert_to_pos_poly_and_multiplier_variable_map.keys()
        )
        # var_eps must be positive
        self.certify_for_fixed_multiplier_prog.AddBoundingBoxConstraint(np.zeros(self.num_faces), np.inf*np.ones(self.num_faces),
                                                                        self.var_eps)


    def _add_plane_to_prog(self, prog, a_coeffs, b_coeffs, penalize_coeffs=True):
        prog.AddDecisionVariables(a_coeffs)
        prog.AddDecisionVariables(b_coeffs)

        if penalize_coeffs:
            ntmp = a_coeffs.shape[0] + b_coeffs.shape[0]
            Qtmp, btmp = np.eye(ntmp), np.zeros(ntmp)
            prog.AddQuadraticCost(Qtmp, btmp, np.concatenate([a_coeffs, b_coeffs]), is_convex=True)

        return prog

    def ManualAddEqualityConstraintsBetweenPolynomials(self, prog, p1, p2):
        """
        adds the equality constraint between polynomials and returns list of all the equality constraints imposed
        :param prog: prog to add constraint
        :param p1: polynomial 1
        :param p2: polynomial 2
        :return: list of linear equality constraint bindings
        """
        diff = p1-p2
        constraint_list = []
        for mono, coeff in diff.monomial_to_coefficient_map().items():
            linear_constraint_binding = prog.AddLinearEqualityConstraint(coeff, 0)
            constraint_list.append(linear_constraint_binding)
        return constraint_list

    def _extract_new_region(self, result):
        self.cur_eps = result.GetSolution(self.var_eps)
        return HPolyhedron(self.region.A(), self.region.b()-self.var_eps)