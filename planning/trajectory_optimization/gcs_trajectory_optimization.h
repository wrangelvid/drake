#pragma once

#include <map>
#include <vector>

#include "drake/common/trajectories/bspline_trajectory.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/graph_of_convex_sets.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/point.h"

namespace drake {
namespace planning {
namespace trajectory_optimization {

using geometry::optimization::GraphOfConvexSets;
using geometry::optimization::GraphOfConvexSetsOptions;
using geometry::optimization::HPolyhedron;
using geometry::optimization::Point;
using Vertex = geometry::optimization::GraphOfConvexSets::Vertex;
using solvers::Constraint;
using solvers::Cost;

struct GCSTrajectoryOptimizationConstructor {
  GCSTrajectoryOptimizationConstructor() = default;

  /** The order of the Bezier trajectory within a region.
  It will have order + 1 control points.
  The order must be at least 1.
  */
  int order{3};

  /** The maximum duration spend in a region (seconds).
  Some solvers struggle numerically with large values.
  */
  double d_max{20};

  /** Some cost/constraints are only convex for d > 0.
  For example the perspective quadratic cost of the path energy ||r'(s)||^2 / d
  becomes non-convex for d = 0. Otherwise d_min can be set to 0.
  */
  double d_min{1e-6};

  /**
  TODO(wrangelvid): Allow for custom edge variables.
  TODO(wrangelvid): Add full-dimension overlap graph generation.
  */
};

/**
GCSTrajectoryOptimization implements a simplified motion planning optimization
problem introduced in the paper "Motion Planning around Obstacles with Convex
Optimization."

"Motion Planning around Obstacles with Convex Optimization" by Tobia Marcucci,
Mark Petersen, David von Wrangel, Russ Tedrake. https://arxiv.org/abs/2205.04422

Instead of using the full time scaling curve, this problem uses a single time
scaling variable for each region. This makes enforcing the continuity
constraints more difficult, but significantly simplifies higher order derivative
constraints.
*/
class GCSTrajectoryOptimization {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GCSTrajectoryOptimization);

  /**
  Constructs the motion planning problem.

  @param regions is a list of collision free polytopes in configuration space.
  @param gcs_constructor includes settings to control the construction of the
  construction of the graph of convex sets.
  */
  GCSTrajectoryOptimization(
      const std::vector<HPolyhedron>& regions,
      const GCSTrajectoryOptimizationConstructor& constructor =
          GCSTrajectoryOptimizationConstructor());

  /** Returns the number of position variables. */
  int num_positions() const { return num_positions_; };

  /**
  @param show_slacks determines whether the values of the intermediate
  (slack) variables are also displayed in the graph.
  @param precision sets the floating point precision (how many digits are
  generated) of the annotations.
  @param scientific sets the floating point formatting to scientific (if true)
  or fixed (if false).
  */
  std::string GetGraphvizString(bool show_slack = true, int precision = 3,
                                bool scientific = false) const {
    return gcs_.GetGraphvizString(std::nullopt, show_slack, precision,
                                  scientific);
  }

  /** Adds a linear constraint on the second derivative of the path,
  `lb` ≤ r̈(s) ≤ `ub`. Note that this does NOT directly constrain q̈(t).
  */
  void AddTimeCost(double weight = 1.0);

  /** Adds multiple L2Norm Costs on the upper bound of the path length.
  We upper bound the path integral by the sum of the distances between
  control points. For Bezier curves, this is equivalent to the sum
  of the L2Norm of the derivative control points of the curve divided by the
  order.
  */
  void AddPathLengthCost(double weight = 1.0);

  /**
  TODO(wrangelvid): write a doc string.
  */
  void AddPathEnergyCost(double weight = 1.0);

  /** Adds a linear velocity constraints to the entire graph `lb` ≤ q̈(t) ≤ `ub`.
   */
  void AddVelocityBounds(const Eigen::Ref<const Eigen::VectorXd>& lb,
                         const Eigen::Ref<const Eigen::VectorXd>& ub);

  /**
  TODO(wrangelvid): write a doc string.
  */
  void AddSourceTarget(const Eigen::Ref<const Eigen::VectorXd>& source,
                       const Eigen::Ref<const Eigen::VectorXd>& target);

  trajectories::BsplineTrajectory<double> SolvePath(
      const GraphOfConvexSetsOptions& options);

 private:
  std::vector<HPolyhedron> regions_{};
  GCSTrajectoryOptimizationConstructor constructor_;
  int num_positions_{};

  Vertex* source_{};
  Vertex* target_{};

  std::vector<std::shared_ptr<Cost>> edge_cost_{};
  std::vector<std::shared_ptr<Constraint>> edge_constraint_{};

  Eigen::VectorX<symbolic::Variable> u_duration_;
  Eigen::VectorX<symbolic::Variable> u_vars_;

  // r(s)
  trajectories::BsplineTrajectory<symbolic::Expression> u_r_trajectory_;
  GraphOfConvexSets gcs_{GraphOfConvexSets()};
};

}  // namespace trajectory_optimization
}  // namespace planning
}  // namespace drake
