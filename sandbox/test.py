#Based off of https://github.com/RussTedrake/manipulation/blob/iris/iris.ipynb
import sys
import os
import time
import numpy as np
import pydrake
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
from pydrake.all import Variable
from pydrake.all import MultibodyPositionToGeometryPose, ConnectMeshcatVisualizer, Role, Sphere
from pydrake.all import (
    ConvexSet, HPolyhedron, Hyperellipsoid,
    MathematicalProgram, Solve, le, IpoptSolver,
)

from meshcat import Visualizer

#%%

# Setup meshcat
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=[])

# Sporadically need to run `pkill -f meshcat`

#%%

simple_collision = True
# gripper_welded = True

vis = Visualizer(zmq_url=zmq_url)
vis.delete()
# display(vis.jupyter_cell())

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
parser = Parser(plant)
parser.package_map().Add( "wsg_50_description", os.path.dirname(FindResourceOrThrow(
            "drake/manipulation/models/wsg_50_description/package.xml")))

directives_file = FindResourceOrThrow("drake/sandbox/planar_iiwa_simple_collision_welded_gripper.yaml") \
    if simple_collision else FindResourceOrThrow("drake/sandbox/planar_iiwa_dense_collision_welded_gripper.yaml")
directives = LoadModelDirectives(directives_file)
models = ProcessModelDirectives(directives, plant, parser)

q0 = [-0.2, -1.2, 1.6]
index = 0
for joint_index in plant.GetJointIndices(models[0].model_instance):
    joint = plant.get_mutable_joint(joint_index)
    if isinstance(joint, RevoluteJoint):
        joint.set_default_angle(q0[index])
        index += 1

plant.Finalize()

visualizer = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url, delete_prefix_on_load=False)

diagram = builder.Build()
visualizer.load()
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)
diagram.Publish(context)


def visualize_trajectory(traj):
    builder = DiagramBuilder()

    scene_graph = builder.AddSystem(SceneGraph())
    plant = MultibodyPlant(time_step=0.0)
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    parser = Parser(plant)
    parser.package_map().Add( "wsg_50_description", os.path.dirname(FindResourceOrThrow(
                "drake/manipulation/models/wsg_50_description/package.xml")))

    directives_file = FindResourceOrThrow("drake/sandbox/planar_iiwa_simple_collision_welded_gripper.yaml") \
        if simple_collision else FindResourceOrThrow("drake/sandbox/planar_iiwa_dense_collision_welded_gripper.yaml")
    directives = LoadModelDirectives(directives_file)
    models = ProcessModelDirectives(directives, plant, parser)

    q0 = [-0.2, -1.2, 1.6]
    index = 0
    for joint_index in plant.GetJointIndices(models[0].model_instance):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    plant.Finalize()

    to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant))
    builder.Connect(to_pose.get_output_port(), scene_graph.get_source_pose_port(plant.get_source_id()))

    traj_system = builder.AddSystem(TrajectorySource(traj))
    builder.Connect(traj_system.get_output_port(), to_pose.get_input_port())

    meshcat = ConnectMeshcatVisualizer(builder, scene_graph, zmq_url=zmq_url)

    vis_diagram = builder.Build()

    simulator = Simulator(vis_diagram)
    meshcat.start_recording()
    simulator.AdvanceTo(traj.end_time())
    meshcat.publish_recording()
    with open("/tmp/spp_shelves.html", "w") as f:
        f.write(meshcat.vis.static_html())


seed_points = np.array([[0.0, -2.016, 1.975], # in tight
                        [-1, -2, 0.5],        # neutral pose
                        [0.3, -0.8, 0.5],     # above shelf
                        [0.25, -1.6, -0.25],  # in shelf 1
                        [0.07, -1.8, -0.2],   # leaving shelf 1
                        [-0.1, -2, -0.3]])    # out of shelf 1

# traj = PiecewisePolynomial.FirstOrderHold(np.array([0, 1]), np.array([seed_points[4], seed_points[1]]).T)
# visualize_trajectory(traj)

#%%

#  Now IRIS in configuration space, using dReal to solve for the growth volume
# through the nonconvex kinematics.

from pydrake.all import (
    DrealSolver, eq, SnoptSolver,
    Sphere, Ellipsoid, GeometrySet,
    RigidBody_, AutoDiffXd, initializeAutoDiff,
)

# Maybe this one doesn't actually deserve to be part of the main class,
# or it needs to be renamed.
# It's really the gradient of solution to the GrowthVolume optimization.
def TangentPlane(self, point):
    a = 2 * self.A().T @ self.A() @ (point - self.center())
    a = a / np.linalg.norm(a)
    b = a.dot(point)
    return a, b

Hyperellipsoid.TangentPlane = TangentPlane

# diagram, plant, scene_graph = make_environment(robot=True, gripper=False)
# lb = np.array([0, -.5, 0])
# ub = np.array([1, .5, 1])
# context = diagram.CreateDefaultContext()
# query = scene_graph.get_query_output_port().Eval(scene_graph.GetMyContextFromRoot(context))

sym_plant = plant.ToSymbolic()
sym_context = sym_plant.CreateDefaultContext()
dReal = DrealSolver()

# For SNOPT test.
autodiff_plant = plant.ToAutoDiffXd()
autodiff_context = autodiff_plant.CreateDefaultContext()
snopt = SnoptSolver()

vis.delete()

def MakeFromSceneGraph(query, geom, expressed_in=None):
    shape = query.inspector().GetShape(geom)
    if isinstance(shape, (Sphere, Ellipsoid)):
        return Hyperellipsoid(query, geom, expressed_in)
    return HPolyhedron(query, geom, expressed_in)

dReal_polytope_tol = .1

def CheckVolume(E, bodyA, bodyB, setA, setB, A, b, volume):
    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(plant.num_positions(), "q")
    prog.AddBoundingBoxConstraint(lb, ub, q)
    prog.AddLinearConstraint(A, b-np.inf, b, q)
    p_AA =  prog.NewContinuousVariables(3, "p_AA")
    p_BB = prog.NewContinuousVariables(3, "p_BB")
    if volume < np.inf:
        prog.AddConstraint((q-E.center()).T @ E.A().T @ E.A() @ (q-E.center()) <= volume)
    setA.AddPointInSetConstraints(prog, p_AA)
    setB.AddPointInSetConstraints(prog, p_BB)

    sym_plant.SetPositions(sym_context, q)
    X_WA = sym_plant.EvalBodyPoseInWorld(sym_context, bodyA)
    X_WB = sym_plant.EvalBodyPoseInWorld(sym_context, bodyB)
    # Add +0 pending https://github.com/RobotLocomotion/drake/issues/15216
    p_WA = X_WA.multiply(p_AA+0)
    p_WB = X_WB.multiply(p_BB+0)
    prog.AddConstraint(eq(p_WA, p_WB))
    prog.SetSolverOption(dReal.id(), "precision", .9*dReal_polytope_tol)
    result = dReal.Solve(prog)
    return result.is_success(), result.GetSolution(q)


def GrowthVolume(E, bodyA, bodyB, setA, setB, A, b, guess=None):
    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(plant.num_positions(), "q")

    if guess is not None:
        prog.SetInitialGuess(q, guess)

    prog.AddLinearConstraint(A, b-np.inf, b, q)
    p_AA = prog.NewContinuousVariables(3, "p_AA")
    p_BB = prog.NewContinuousVariables(3, "p_BB")
    setA.AddPointInSetConstraints(prog, p_AA)
    setB.AddPointInSetConstraints(prog, p_BB)
    prog.AddQuadraticErrorCost(E.A().T @ E.A(), E.center(), q)

    # TODO: Remove these.  They're for debugging only.
    #set_meshcat_object(vis['setA'],setA)
    #set_meshcat_object(vis['setB'],setB)

    if isinstance(bodyA, RigidBody_[AutoDiffXd]):
        # TODO: Update this to use MBP<double> and Jacobians if I'm going to keep it.
        def kinematics_constraint(vars):
            p_AA, p_BB, q = np.split(vars,[3,6])
            autodiff_plant.SetPositions(autodiff_context, q)
            X_WA = autodiff_plant.EvalBodyPoseInWorld(autodiff_context, bodyA)
            X_WB = autodiff_plant.EvalBodyPoseInWorld(autodiff_context, bodyB)
            p_WA = X_WA.multiply(p_AA)
            p_WB = X_WB.multiply(p_BB)
            return p_WA - p_WB
        prog.AddConstraint(kinematics_constraint, lb=[0,0,0], ub=[0,0,0],
                           vars=np.concatenate((p_AA,p_BB,q)))
        result = snopt.Solve(prog)

    else:
        # TODO: Construct these symbolic expressions once per body outside this method.
        # But I would have to substitute in the new q each time.
        # Better is to construct the prog once for each pair, and just update the cost?
        sym_plant.SetPositions(sym_context, q)
        X_WA = sym_plant.EvalBodyPoseInWorld(sym_context, bodyA)
        X_WB = sym_plant.EvalBodyPoseInWorld(sym_context, bodyB)
        # Add +0 pending https://github.com/RobotLocomotion/drake/issues/15216
        p_WA = X_WA.multiply(p_AA+0)
        p_WB = X_WB.multiply(p_BB+0)
        prog.AddConstraint(eq(p_WA, p_WB))
        result = snopt.Solve(prog)
        #result = dReal.Solve(prog)

    return result.is_success(), result.get_optimal_cost(), result.GetSolution(q)

def iris_cspace(query, point, require_containment_points=[], termination_threshold=2e-2, iteration_limit=100):
    vis = Visualizer(zmq_url=zmq_url)
    set_meshcat_object(vis['sample'], point, color=0x99dd99)

    ellipsoid_epsilon = 1e-1
    dim = plant.num_positions()
    lb = plant.GetPositionLowerLimits()
    ub = plant.GetPositionUpperLimits()
    assert dim == 3 # need to update the volume once this changes
    volume_of_unit_sphere = 4.0*np.pi/3.0
    E = Hyperellipsoid(np.eye(3)/ellipsoid_epsilon, point)
    set_meshcat_object(vis['ellipse'], E)
    best_volume = ellipsoid_epsilon**dim * volume_of_unit_sphere

    inspector = query.inspector()
    pairs = inspector.GetCollisionCandidates()

    P = HPolyhedron.MakeBox(lb, ub)
    A = np.vstack((P.A(), np.zeros((10*len(pairs),3))))  # allow up to 10 faces per pair.
    b = np.concatenate((P.b(), np.zeros(10*len(pairs))))

    geom_ids = inspector.GetGeometryIds(GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity)
    sets = {geom:MakeFromSceneGraph(query, geom, inspector.GetFrameId(geom)) for geom in geom_ids}

    use_autodiff = True
    use_dReal = True
    ad_bodies = {geom:autodiff_plant.GetBodyFromFrameId(inspector.GetFrameId(geom)) for geom in geom_ids}
    sym_bodies = {geom:sym_plant.GetBodyFromFrameId(inspector.GetFrameId(geom)) for geom in geom_ids}
    if use_autodiff:
        bodies = ad_bodies

    iteration = 0
    num_faces = 2*len(lb)
    while True:
        ## Find separating hyperplanes

        for geomA, geomB in pairs:
            print(f"geomA={inspector.GetName(geomA)}, geomB={inspector.GetName(geomB)}")
            # Run snopt at the beginning
            while True:
                success, growth, qstar = GrowthVolume(E,
                    bodies[geomA], bodies[geomB],
                    sets[geomA], sets[geomB], A[:num_faces,:], b[:num_faces] - dReal_polytope_tol, point)
                if success:
                    print(f"snopt_example={qstar}, growth = {growth}")
                    # Add a face to the polytope
                    A[num_faces,:], b[num_faces] = E.TangentPlane(qstar)
                    num_faces += 1
                else:
                    break

            if use_dReal:
                tries = 0
                while True:
                    reachable, counter_example = CheckVolume(
                        E, sym_bodies[geomA], sym_bodies[geomB], sets[geomA], sets[geomB],
                        A[:num_faces,:], b[:num_faces] - dReal_polytope_tol, np.inf)
                    if not reachable:
                        print("unreachable")
                        break
                    else:
                        z = E.A() @ (counter_example - E.center())
                        dreal_growth = z.dot(z)
                        print(f"counter_example = {counter_example}, growth = {dreal_growth}")
                        success, growth, qstar = GrowthVolume(E,
                            bodies[geomA], bodies[geomB],
                            sets[geomA], sets[geomB], A[:num_faces,:], b[:num_faces] - dReal_polytope_tol, counter_example)
                        if success:
                            print(f"snopt_example={qstar}, growth = {growth}")
                            # Add a face to the polytope
                            A[num_faces,:], b[num_faces] = E.TangentPlane(qstar)
                            num_faces += 1
                        if np.all(A[:num_faces,:] @ counter_example <= b[:num_faces] - dReal_polytope_tol):
                            # Then also add the counter-example
                            A[num_faces,:], b[num_faces] = E.TangentPlane(counter_example)
                            num_faces += 1
                    tries += 1

        if any([np.any(A[:num_faces,:] @ q > b[:num_faces]) for q in require_containment_points]):
            print("terminating because a required containment point would have not been contained")
            break

        P = HPolyhedron(A[:num_faces,:],b[:num_faces])
        set_meshcat_object(vis[f'polytope'], P, wireframe=True)

        E = P.MaximumVolumeInscribedEllipsoid()
        set_meshcat_object(vis[f'ellipse'], E)
        print(iteration)

        iteration += 1
        if iteration >= iteration_limit:
            break

        volume = volume_of_unit_sphere / np.linalg.det(E.A())
        if volume - best_volume <= termination_threshold:
            break
        best_volume = volume

    return P


# q = np.array([0.55, 0, 0.65])
#q0 = plant.GetPositions(plant.GetMyContextFromRoot(context))
# iris_cspace(query, q, require_containment_points=[q], iteration_limit=100);

#%%

print(sym_plant)
link0 = sym_plant.GetBodyByName("iiwa_link_7")
sym_plant.SetPositions(
  context, Variable("theta"))
print(sym_plant.GetAccelerationLowerLimits)

#%%

# Cpp
iris_options = IrisOptions()
iris_options.require_sample_point_is_contained = True
iris_options.iteration_limit = 10
iris_options.enable_ibex = False

regions = []
for i in range(seed_points.shape[0]):
    start_time = time.time()
#     hpoly = IrisInConfigurationSpace(plant, plant_context, seed_points[i,:], iris_options)
    hpoly = iris_cspace(query, seed_points[i,:], require_containment_points=[seed_points[i,:]], iteration_limit=100)
    ellipse = hpoly.MaximumVolumeInscribedEllipsoid()
    print("Time: %6.2f \tVolume: %6.2f \tCenter:" % (time.time() - start_time, ellipse.Volume()),
          ellipse.center(), flush=True)
    regions.append(hpoly)

#%%

# Solve path planning
start_time = time.time()
spp = BsplineTrajectoryThroughUnionOfHPolyhedra(seed_points[2,:], seed_points[3,:], regions)
spp.set_max_velocity([.4, .4, .4])
spp.set_extra_control_points_per_region(5)
# print(spp.num_regions())
traj = spp.Solve()
print(time.time() - start_time)
print(traj.start_time())
print(traj.end_time())

for q in traj.control_points():
    if not any([r.PointInSet(q) for r in regions]):
        print(f"control point {q} in not in any region")


vis.delete()
visualize_trajectory(traj)

#%%

def get_ctrl_plant():
    plant = MultibodyPlant(time_step=0.0)
    parser = Parser(plant)
    parser.package_map().Add( "wsg_50_description", os.path.dirname(FindResourceOrThrow(
                "drake/manipulation/models/wsg_50_description/package.xml")))

    directives_file = FindResourceOrThrow("drake/sandbox/planar_iiwa_simple_collision_welded_gripper.yaml") \
        if simple_collision else FindResourceOrThrow("drake/sandbox/planar_iiwa_dense_collision_welded_gripper.yaml")
    directives = LoadModelDirectives(directives_file)
    models = ProcessModelDirectives(directives, plant, parser)

    q0 = [-0.2, -1.2, 1.6]
    index = 0
    for joint_index in plant.GetJointIndices(models[0].model_instance):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    plant.Finalize()

    return plant

start_time = time.time()
toppra_options = CalcGridPointsOptions()
gridpoints = Toppra.CalcGridPoints(traj, toppra_options)
toppra = Toppra(traj, get_ctrl_plant(), gridpoints)
vel_con = toppra.AddJointVelocityLimit([-0.6, -0.6, -0.6], [0.6, 0.6, 0.6])
acc_con = toppra.AddJointAccelerationLimit([-2, -2, -2], [2, 2, 2])
s_traj = toppra.SolvePathParameterization()

# q(s) & s(t) -> q(t)
print(time.time() - start_time)
print(s_traj.end_time())

opt_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    s_traj.get_segment_times(), traj.vector_values(gridpoints), np.zeros(3), np.zeros(3))
visualize_trajectory(opt_traj)

#%%

visualize_trajectory(traj)

#%%


