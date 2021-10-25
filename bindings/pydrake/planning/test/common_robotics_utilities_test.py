import unittest
import numpy as np

from pydrake.planning.common_robotics_utilities import (
    MakeKinematicLinearRRTNearestNeighborsFunction,
    MakeRRTTimeoutTerminationFunction,
    PropagatedState,
    RRTPlanSinglePath,
    SimpleRRTPlannerState,
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
            for ii in range(1, check_dist//check_step):
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
        self.assertTrue(False)
