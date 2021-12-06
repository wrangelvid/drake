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

import numpy as np

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
                print(f"geomA={inspector.GetName(geomA)}, geomB={inspector.GetName(geomB)}")
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
                        print(f"snopt_example={qstar}, growth = {growth}")
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


     