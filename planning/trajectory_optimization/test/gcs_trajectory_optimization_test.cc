#include "drake/planning/trajectory_optimization/gcs_trajectory_optimization.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/point.h"

namespace drake {
namespace planning {
namespace trajectory_optimization {
namespace {

using Eigen::Vector2d;
using geometry::optimization::ConvexSets;
using geometry::optimization::HPolyhedron;
using geometry::optimization::Point;

GTEST_TEST(GCSTrajectoryOptimizationTest, Basic) {
  const int kDimension = 2;
  GCSTrajectoryOptimization gcs(kDimension);
  EXPECT_EQ(gcs.num_positions(), kDimension);

  // Add a single region (the unit box), and plan a line segment inside that
  // box.
  Vector2d start(-0.5, -0.5), goal(0.5, 0.5);
  ConvexSets region_sets, source_sets, target_sets;
  region_sets.emplace_back(HPolyhedron::MakeUnitBox(kDimension));
  source_sets.emplace_back(Point(start));
  target_sets.emplace_back(Point(goal));
  auto* regions = gcs.AddRegions(region_sets, 1);
  auto* source = gcs.AddRegions(source_sets, 0);
  auto* target = gcs.AddRegions(target_sets, 0);

  gcs.AddEdges(source, regions);
  gcs.AddEdges(regions, target);

  auto [traj, result] = gcs.SolvePath(*source, *target);
  EXPECT_TRUE(result.is_success());
  EXPECT_EQ(traj.rows(), 2);
  EXPECT_EQ(traj.cols(), 1);
  EXPECT_TRUE(CompareMatrices(traj.value(traj.start_time()), start, 1e-6));
  EXPECT_TRUE(CompareMatrices(traj.value(traj.end_time()), goal, 1e-6));
}

}  // namespace
}  // namespace trajectory_optimization
}  // namespace planning
}  // namespace drake
