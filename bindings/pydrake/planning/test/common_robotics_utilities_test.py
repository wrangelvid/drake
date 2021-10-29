import unittest
import numpy as np

from pydrake.planning.common_robotics_utilities import (
    MakeKinematicLinearRRTNearestNeighborsFunction,
    MakeRRTTimeoutTerminationFunction,
    PropagatedState,
    RRTPlanSinglePath,
    SimpleRRTPlannerState,
    Graph,
    GrowRoadMap,
    UpdateRoadMapEdges,
)


class TestCommonRoboticsUtilities(unittest.TestCase):
    def test_rrt(self):
        start = np.array([0.5, 0.5])
        goal = np.array([2.5, 0.5])
        goal_bias = 0.05
        step_size = 0.1
        check_step = 0.01
        solve_timeout = 2

        rrt_tree = [SimpleRRTPlannerState(start)]

        def sampling_fn():
            if (np.random.rand() < goal_bias):
                return goal
            return np.random.rand(2) * 3

        def distance_fn(point1, point2):
            return np.linalg.norm(point2 - point1)

        def check_goal_fn(sample):
            return np.linalg.norm(sample - goal) < 1e-6

        def extend_fn(nearest, sample):
            if (sample[0] >= 1 and sample[0] <= 2 and sample[1] <= 2):
                return []

            extend = None
            extend_dist = distance_fn(nearest, sample)
            if extend_dist <= step_size:
                extend = sample
            else:
                extend = nearest + step_size/extend_dist * (sample - nearest)

            check_dist = distance_fn(nearest, extend)
            for ii in range(1, int(check_dist/check_step)):
                check_point = nearest \
                    + ii * check_step / check_dist * (extend - nearest)
                if (check_point[0] >= 1 and check_point[0] <= 2
                        and check_point[1] <= 2):
                    return []
            return [PropagatedState(state=extend, relative_parent_index=-1)]

        nearest_neighbor_fn = MakeKinematicLinearRRTNearestNeighborsFunction(
            distance_fn=distance_fn, use_parallel=False)

        termination_fn = MakeRRTTimeoutTerminationFunction(solve_timeout)

        single_result = RRTPlanSinglePath(
            tree=rrt_tree, sampling_fn=sampling_fn,
            nearest_neighbor_fn=nearest_neighbor_fn,
            forward_propagation_fn=extend_fn, state_added_callback_fn=None,
            check_goal_reached_fn=check_goal_fn, goal_reached_callback_fn=None,
            termination_check_fn=termination_fn)

        print(single_result.Path())

    def test_prm(self):
        np.random.seed(42)
        test_env = np.array(["####################",
                             "#                  #",
                             "#  ####            #",
                             "#  ####    #####   #",
                             "#  ####    #####   #",
                             "#          #####   #",
                             "#          #####   #",
                             "#                  #",
                             "#      #########   #",
                             "#     ##########   #",
                             "#    ###########   #",
                             "#   ############   #",
                             "#                  #",
                             "#                  #",
                             "#    ##            #",
                             "#    ##   ######## #",
                             "#    ##   ######## #",
                             "#    ##   ######## #",
                             "#                  #",
                             "####################"])

        test_env_shape = [len(test_env[0]), len(test_env)]

        K = 5
        roadmap_size = 100

        def roadmap_termination_fn(current_roadmap_size):
            return current_roadmap_size >= roadmap_size

        def state_sampling_fn():
            x = np.random.randint(test_env_shape[0])
            y = np.random.randint(test_env_shape[1])

            return np.array([x, y])

        def distance_fn(start, end):
            return np.linalg.norm(end - start)

        def check_state_validity_fn(point):
            x, y = point
            return test_env[int(y)][int(x)] != '#'

        def check_edge_validity_fn(start, end):
            def checkEdgeCollisionFree(start, end, stepsize):
                num_steps = np.ceil(distance_fn(start,end)/stepsize)

                for step in range(int(num_steps)+1):
                    interpolation_ratio = step / num_steps
                    interpolated_point = start + np.round(interpolation_ratio*(start-end))

                    if not check_state_validity_fn(interpolated_point):
                        return False

                return True

            return checkEdgeCollisionFree(start, end, 0.5 ) and checkEdgeCollisionFree(end, start, 0.5)

        roadmap = Graph()

        GrowRoadMap(roadmap, state_sampling_fn, distance_fn, check_state_validity_fn,
                    check_edge_validity_fn, roadmap_termination_fn, K, False, True, False)
        # self.assertTrue(roadmap.CheckGraphLinkage())

        UpdateRoadMapEdges(roadmap, check_edge_validity_fn, distance_fn, False)
        # self.assertTrue(roadmap.CheckGraphLinkage())

        nodes_to_prune = [10,20,30,40,50,60]
        # serial_pruned_roadmap = roadmap.MakePrunedCopy(nodes_to_prune, False)
        # self.assertTrue(serial_pruned_roadmap.CheckGraphLinkage())

        # parallel_pruned_roadmap = roadmap.MakePrunedCopy(nodes_to_prune, True)
        # self.assertTrue(parallel_pruned_roadmap.CheckGraphLinkage())
