import numpy as np
import scipy
from meshcat.servers.zmqserver import start_zmq_server_as_subprocess
from meshcat import Visualizer
import meshcat
from pydrake.all import ConnectMeshcatVisualizer
from t_space_utils import EvaluatePlanePair
from pydrake.all import InverseKinematics
from functools import partial
import mcubes
import visualizations_utils as viz_utils
from pydrake.all import RationalForwardKinematics, GeometrySet
from pydrake.geometry import Role
import pydrake.symbolic as sym
import iris_utils
from IPython.display import display

class IrisPlantVisualizer:
    def __init__(self, plant, builder, scene_graph, **kwargs):
        proc, zmq_url, web_url = start_zmq_server_as_subprocess(server_args=[])
        proc2, zmq_url2, web_url2 = start_zmq_server_as_subprocess(server_args=[])
        self.vis = Visualizer(zmq_url=zmq_url)
        self.vis.delete()
        self.vis2 = Visualizer(zmq_url=zmq_url2)
        self.vis2.delete()

        self.plant = plant

        self.builder = builder
        self.scene_graph = scene_graph

        # Construct Rational Forward Kinematics
        self.forward_kin = RationalForwardKinematics(plant)
        self.t_variables = sym.Variables(self.forward_kin.t())
        self.t_array = self.forward_kin.t()
        self.num_joints = self.plant.num_positions()
        # the point around which we construct the stereographic projection
        self.q_star = kwargs.get('q_star', np.zeros(self.num_joints))
        self.q_lower_limits = plant.GetPositionLowerLimits()
        self.t_lower_limits = self.forward_kin.ComputeTValue(self.q_lower_limits,  self.q_star).squeeze()
        self.q_upper_limits = plant.GetPositionUpperLimits()
        self.t_upper_limits = self.forward_kin.ComputeTValue(self.q_upper_limits,  self.q_star).squeeze()

        self.viz_role = kwargs.get('viz_role', Role.kIllustration)
        visualizer = ConnectMeshcatVisualizer(self.builder, scene_graph, zmq_url=zmq_url,
                                      delete_prefix_on_load=False, role=self.viz_role)
        self.diagram = self.builder.Build()
        visualizer.load()
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = plant.GetMyContextFromRoot(self.diagram_context)
        self.diagram.Publish(self.diagram_context)

        # construct collision pairs
        self.query = self.scene_graph.get_query_output_port().Eval(
            self.scene_graph.GetMyContextFromRoot(self.diagram_context))
        self.inspector = self.query.inspector()
        self.pairs = self.inspector.GetCollisionCandidates()

        # only gets kProximity pairs. Might be more efficient?
        # geom_ids = inspector.GetGeometryIds(GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity)
        pair_set = set()
        for p in self.pairs:
            pair_set.add(p[0])
            pair_set.add(p[1])
        self.geom_ids = self.inspector.GetGeometryIds(GeometrySet(list(pair_set)))
        self.link_poses_by_body_index_rat_pose = self.forward_kin.CalcLinkPoses(self.q_star,
                                                                                self.plant.world_body().index())
        self.X_WA_list = [p.asRigidTransformExpr() for p in self.link_poses_by_body_index_rat_pose]
        self.body_indexes_by_geom_id = {geom:
                                            plant.GetBodyFromFrameId(self.inspector.GetFrameId(geom)).index() for geom
                                        in
                                        self.geom_ids}
        self.hpoly_sets_in_self_frame_by_geom_id = {
            geom: iris_utils.MakeFromHPolyhedronSceneGraph(self.query, geom, self.inspector.GetFrameId(geom))
            for geom in self.geom_ids}
        self.vpoly_sets_in_self_frame_by_geom_id = {
            geom: iris_utils.MakeFromVPolytopeSceneGraph(self.query, geom, self.inspector.GetFrameId(geom))
            for geom in self.geom_ids}

        self.t_space_vertex_world_position_by_geom_id = {}
        for geom in self.geom_ids:
            VPoly = self.vpoly_sets_in_self_frame_by_geom_id[geom]
            num_verts = VPoly.vertices().shape[1]
            X_WA = self.X_WA_list[int(self.body_indexes_by_geom_id[geom])]
            R_WA = X_WA.rotation().matrix()
            p_WA = X_WA.translation()
            vert_pos = R_WA @ (VPoly.vertices()) + np.repeat(p_WA[:, np.newaxis], num_verts, 1)
            self.t_space_vertex_world_position_by_geom_id[geom] = vert_pos

        self.ik = InverseKinematics(plant, self.plant_context)
        self.collision_constraint = self.ik.AddMinimumDistanceConstraint(1e-4, 0.01)
        self.col_func_handle = partial(self.eval_cons, c=self.collision_constraint, tol=0.01)
        self.col_func_handle_rational = self.eval_cons_rational

        #plotting planes setup
        x = np.linspace(-1, 1, 3)
        y = np.linspace(-1, 1, 3)
        verts = []

        for idxx in range(len(x)):
            for idxy in range(len(y)):
                verts.append(np.array([x[idxx], y[idxy]]))
        self.tri = scipy.spatial.Delaunay(verts)
        self.plane_triangles = self.tri.simplices
        self.plane_verts = self.tri.points[:, :]
        self.plane_verts = np.concatenate((self.plane_verts, 0 * self.plane_verts[:, 0].reshape(-1, 1)), axis=1)


        #region -> (collision -> plane dictionary)
        self.region_to_collision_pair_to_plane_dictionary = None

    def jupyter_cell(self,):
        display(self.vis.jupyter_cell())
        display(self.vis2.jupyter_cell())

    def eval_cons(self, q, c, tol):
        return 1 - 1 * float(c.evaluator().CheckSatisfied(q, tol))

    def eval_cons_rational(self, *t):
        t = np.array(t)
        q = self.forward_kin.ComputeQValue(t, self.q_star).squeeze()
        return self.col_func_handle(q)

    def visualize_collision_constraint(self, N = 50):
        """
        :param N: N is density of marchingcubes grid. Runtime scales cubically in N
        :return:
        """
        vertices, triangles = mcubes.marching_cubes_func(tuple(self.t_lower_limits),
                                                         tuple(self.t_upper_limits),
                                                         N, N, N, self.col_func_handle_rational, 0.5)

        self.vis2["collision_constraint"].set_object(
            meshcat.geometry.TriangularMeshGeometry(vertices, triangles),
            meshcat.geometry.MeshLambertMaterial(color=0xff0000, wireframe=True))

    def plot_regions(self, regions, ellipses = None, region_suffix = '', opacity = 0.5):
        viz_utils.plot_regions(self.vis2, regions, ellipses, region_suffix, opacity)

    def plot_seedpoints(self, seed_points):
        for i in range(seed_points.shape[0]):
            self.vis2['iris']['seedpoints']["seedpoint"+str(i)].set_object(
                        meshcat.geometry.Sphere(0.05), meshcat.geometry.MeshLambertMaterial(color=0x0FB900))
            self.vis2['iris']['seedpoints']["seedpoint"+str(i)].set_transform(
                    meshcat.transformations.translation_matrix(seed_points[i,:]))
            
    def plot_vertices(vertices):
        for i in range(vertices.shape[0]):
            visualizer.vis2['iris']['vertices']["vertex"+str(i)].set_object(
                        meshcat.geometry.Sphere(0.05), meshcat.geometry.MeshLambertMaterial(color=0xF9FF33))
            visualizer.vis2['iris']['vertices']["vertex"+str(i)].set_transform(
                    meshcat.transformations.translation_matrix(vertices[i,:]))

    def showres(self,q):
        self.plant.SetPositions(self.plant_context, q)
        col = self.col_func_handle(q)
        t = self.forward_kin.ComputeTValue(q, self.q_star)
        if col:
            self.vis2["t"].set_object(
                meshcat.geometry.Sphere(0.1), meshcat.geometry.MeshLambertMaterial(color=0xFFB900))
            self.vis2["t"].set_transform(
                meshcat.transformations.translation_matrix(t))
        else:
            self.vis2["t"].set_object(
                meshcat.geometry.Sphere(0.1), meshcat.geometry.MeshLambertMaterial(color=0x3EFF00))
            self.vis2["t"].set_transform(
                meshcat.transformations.translation_matrix(t))
        self.diagram.Publish(self.diagram_context)

    def showres_t(self, t):
        q = self.forward_kin.ComputeQValue(t, self.q_star)
        self.showres(q)

    def show_res_with_planes(self, q):
        t = self.forward_kin.ComputeTValue(q, self.q_star)
        self.showres(q)
        if self.region_to_collision_pair_to_plane_dictionary is not None:
            for region_id, (region, collision_pair_to_plane_dictionary) in enumerate(self.region_to_collision_pair_to_plane_dictionary.items()):
                if region.PointInSet(t):
                    colors = viz_utils.n_colors(len(collision_pair_to_plane_dictionary.keys()))
                    for i, (pair, planes) in enumerate(collision_pair_to_plane_dictionary.items()):
                        geomA, geomB = pair[0], pair[1]
                        self.plot_plane_geom_id(geomA, geomB, collision_pair_to_plane_dictionary, t, color=colors[i],
                                                region_name=f"region {region_id}")

    def show_res_with_planes_gid(self, q, gid):
        t = self.forward_kin.ComputeTValue(q, self.q_star)
        self.showres(q)
        if self.region_to_collision_pair_to_plane_dictionary is not None:
            for region_id, (region, collision_pair_to_plane_dictionary) in enumerate(self.region_to_collision_pair_to_plane_dictionary.items()):
                if region.PointInSet(t):
                    colors = viz_utils.n_colors(len(collision_pair_to_plane_dictionary.keys()))
                    for i, (pair, planes) in enumerate(collision_pair_to_plane_dictionary.items()):
                        geomA, geomB = pair[0], pair[1]
                        if geomA == gid or geomB == gid: 
                            self.plot_plane_geom_id(geomA, geomB, collision_pair_to_plane_dictionary, t, color=colors[i],
                                                    region_name=f"region {region_id}")

    def plot_plane_geom_id(self, geomA, geomB, planes_dict, cur_t, color=(0, 0, 0), region_name = ''):
        verts_tf, p1, p2 = self.transform_plane_geom_id(geomA, geomB, planes_dict, cur_t)

        mat = meshcat.geometry.MeshLambertMaterial(color=viz_utils.rgb_to_hex(color), wireframe=False)
        mat.opacity = 0.5
        self.vis[region_name]["plane"][f"{geomA.get_value()}, {geomB.get_value()}"].set_object(
            meshcat.geometry.TriangularMeshGeometry(verts_tf, self.plane_triangles),
            mat)

        mat.opacity = 1.0
        viz_utils.plot_point(loc=p1, radius=0.05, mat=mat, vis=self.vis[region_name]["plane"][f"{geomA.get_value()}, {geomB.get_value()}"],
                         marker_id='p1')
        mat = meshcat.geometry.MeshLambertMaterial(color=viz_utils.rgb_to_hex(color), wireframe=False)
        mat.opacity = 1.0
        viz_utils.plot_point(loc=p2, radius=0.05, mat=mat, vis=self.vis[region_name]["plane"][f"{geomA.get_value()}, {geomB.get_value()}"],
                         marker_id='p2')

    def transform(self, a, b, p1, p2, plane_verts):
        alpha = (-b - a.T @ p1) / (a.T @ (p2 - p1))
        offset = alpha * (p2 - p1) + p1
        z = np.array([0, 0, 1])
        crossprod = np.cross(viz_utils.normalize(a), z)
        if np.linalg.norm(crossprod) <= 1e-4:
            R = np.eye(3)
        else:
            ang = np.arcsin(np.linalg.norm(crossprod))
            axis = viz_utils.normalize(crossprod)
            R = viz_utils.get_rotation_matrix(axis, -ang)

        verts_tf = (R @ plane_verts.T).T + offset
        return verts_tf

    def transform_at_t(self, cur_t, a_poly, b_poly, p1_rat, p2_rat):
        eval_dict = dict(zip(b_poly.indeterminates(), cur_t))
        a, b = EvaluatePlanePair((a_poly, b_poly), eval_dict)
        eval_dict = dict(zip(self.t_variables, cur_t))
        #     print(f"{a}, {b}")
        p1 = np.array([p.Evaluate(eval_dict) for p in p1_rat])
        p2 = np.array([p.Evaluate(eval_dict) for p in p2_rat])
        return self.transform(a, b, p1, p2, self.plane_verts), p1, p2

    def transform_plane_geom_id(self, geomA, geomB, planes_dict, cur_t):
        vA = self.t_space_vertex_world_position_by_geom_id[geomA][:, 0]
        vB = self.t_space_vertex_world_position_by_geom_id[geomB][:, 0]
        a_poly, b_poly = planes_dict[(geomA, geomB)]
        return self.transform_at_t(cur_t, a_poly, b_poly, vA, vB)

    def animate_t(self, traj, steps, runtime):
        # loop
        idx = 0
        going_fwd = True
        time_points = np.linspace(0, traj.end_time(), steps)

        for _ in range(runtime):
            # print(idx)
            q = self.forward_kin.ComputeQValue(time_points[idx], self.q_star).squeeze()
            if self.region_to_collision_pair_to_plane_dictionary is not None:
                self.show_res_with_planes(q)
            else:
                self.showres(q)
            if going_fwd:
                if idx + 1 < steps:
                    idx += 1
                else:
                    going_fwd = False
                    idx -= 1
            else:
                if idx - 1 >= 0:
                    idx -= 1
                else:
                    going_fwd = True
                    idx += 1

    def draw_traj_tspace(self, traj, maxit, name):
        # evals end twice fix later
        for it in range(maxit):
            pt = traj.value(it * traj.end_time() / maxit)
            pt_nxt = traj.value((it + 1) * traj.end_time() / maxit)

            pt_q = self.forward_kin.ComputeQValue(pt.reshape(1, -1), self.q_star).squeeze()

            mat = meshcat.geometry.MeshLambertMaterial(color=0xFFF812)
            mat.reflectivity = 1.0
            self.vis2[name]['traj']['points' + str(it)].set_object(viz_utils.meshcat_line(pt.squeeze(), pt_nxt.squeeze(), width=0.03),
                                                              mat)
            #
            # set_joint_angles(pt_q.reshape(-1, ))
            # tf_l2 = self.plant.EvalBodyPoseInWorld(self.plant_context,
            #                                        self.plant.get_body(pydrake.multibody.tree.BodyIndex(3)))
            # R_l2 = tf_l2.rotation()
            # tl_l2 = R_l2 @ np.array([0, 0, 0.9]) + tf_l2.translation()
            #
            # tf_la = self.plant.EvalBodyPoseInWorld(self.plant_context,
            #                                        self.plant.get_body(pydrake.multibody.tree.BodyIndex(4)))
            # R_la = tf_la.rotation()
            # tl_la = R_la @ np.array([0, 0, 1.2]) + tf_la.translation()
            #
            # mat = meshcat.geometry.MeshLambertMaterial(color=0x0029F1)
            # mat.reflectivity = 1.0
            # self.vis[name]['traj']['link2']['points' + str(it)].set_object(
            #     meshcat.geometry.Sphere(0.02), mat)
            # self.vis[name]['traj']['link2']['points' + str(it)].set_transform(
            #     meshcat.transformations.translation_matrix(tl_l2))
            # mat = meshcat.geometry.MeshLambertMaterial(color=0x07F100)
            # mat.reflectivity = 1.0
            # vis[name]['traj']['linka']['points' + str(it)].set_object(
            #     meshcat.geometry.Sphere(0.02), mat)
            # vis[name]['traj']['linka']['points' + str(it)].set_transform(
            #     meshcat.transformations.translation_matrix(tl_la))