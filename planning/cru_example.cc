#include <gflags/gflags.h>

#include <common_robotics_utilities/simple_rrt_planner.hpp>

#include "drake/common/eigen_types.h"
#include "drake/common/text_logging.h"

namespace drake {
namespace planning {
namespace {

using common_robotics_utilities::simple_rrt_planner::CheckGoalReachedFunction;
using common_robotics_utilities::simple_rrt_planner::ForwardPropagation;
using common_robotics_utilities::simple_rrt_planner::
    MakeKinematicLinearRRTNearestNeighborsFunction;
using common_robotics_utilities::simple_rrt_planner::
    MakeRRTTimeoutTerminationFunction;
using common_robotics_utilities::simple_rrt_planner::
    MakeStateAndGoalsSamplingFunction;
using common_robotics_utilities::simple_rrt_planner::PlanningTree;
using common_robotics_utilities::simple_rrt_planner::
    RRTForwardPropagationFunction;
using common_robotics_utilities::simple_rrt_planner::RRTNearestNeighborFunction;
using common_robotics_utilities::simple_rrt_planner::RRTPlanMultiPath;
using common_robotics_utilities::simple_rrt_planner::
    RRTTerminationCheckFunction;
using common_robotics_utilities::simple_rrt_planner::SamplingFunction;
using common_robotics_utilities::simple_rrt_planner::SimpleRRTPlannerState;
using Eigen::Vector2d;

int do_main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::mt19937_64 prng(0);
  double goal_bias = 0.05;
  double step_size = 0.1;
  double check_step = 0.01;
  double solve_timeout = 2;

  Vector2d start(0.5, 0.5);
  Vector2d goal(2.5, 0.5);

  // +-----------+
  // |           |
  // |           |
  // |   +---+   |
  // |   |xxx|   |
  // |   |xxx|   |
  // | s |xxx| g |
  // +---+---+---+

  PlanningTree<Vector2d> rrt_tree;
  rrt_tree.emplace_back(SimpleRRTPlannerState<Vector2d>(start));

  const std::function<Vector2d(void)> state_sampling_fn = [&](void) {
    std::uniform_real_distribution<double> x_dist(0, 3);
    std::uniform_real_distribution<double> y_dist(0, 3);
    return Vector2d(x_dist(prng), y_dist(prng));
  };

  const std::function<double(const Vector2d&, const Vector2d&)> distance_fn =
      [&](const Vector2d& point1, const Vector2d& point2) {
        return (point2 - point1).norm();
      };

  const RRTForwardPropagationFunction<Vector2d> rrt_extend_fn =
      [&](const Vector2d& nearest, const Vector2d& sampled) {
        ForwardPropagation<Vector2d> forward_propagation;

        // Check if sample in collision
        if (sampled(0) >= 1 && sampled(0) <= 2 && sampled(1) <= 2) {
          return forward_propagation;
        }

        Vector2d extend;
        double extend_dist = distance_fn(nearest, sampled);
        if (extend_dist <= step_size) {
          extend = sampled;
        } else {
          extend = nearest + step_size / extend_dist * (sampled - nearest);
        }

        double check_dist = distance_fn(nearest, extend);
        for (int ii = 1; ii < check_dist / check_step; ii++) {
          Vector2d check_point =
              nearest + ii * check_step / check_dist * (extend - nearest);
          // Check if sample in collision
          if (check_point(0) >= 1 && check_point(0) <= 2 &&
              check_point(1) <= 2) {
            return forward_propagation;
          }
        }

        forward_propagation.emplace_back(extend, -1);
        return forward_propagation;
      };

  const CheckGoalReachedFunction<Vector2d> rrt_goal_reached_fn =
      [&](const Vector2d& state) { return (distance_fn(state, goal) < 1e-6); };

  const auto rrt_sample_fn = MakeStateAndGoalsSamplingFunction<Vector2d>(
      state_sampling_fn, {goal}, goal_bias, prng);
  auto rrt_nearest_neighbors_fn =
      MakeKinematicLinearRRTNearestNeighborsFunction<Vector2d>(distance_fn,
                                                               false);

  auto rrt_result = RRTPlanMultiPath<Vector2d>(
      rrt_tree, rrt_sample_fn, rrt_nearest_neighbors_fn, rrt_extend_fn, {},
      rrt_goal_reached_fn, {},
      MakeRRTTimeoutTerminationFunction(solve_timeout));

  auto paths = rrt_result.Paths();
  drake::log()->info("Number of paths: {}", paths.size());
  drake::log()->info("Waypoints for path 1");
  for (size_t jj = 0; jj < paths[0].size(); jj++) {
    drake::log()->info(paths[0][jj].transpose());
  }

  return 0;
}

}  // namespace
}  // namespace planning
}  // namespace drake

int main(int argc, char* argv[]) {
  return drake::planning::do_main(argc, argv);
}
