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
using VertexId = geometry::optimization::GraphOfConvexSets::VertexId;
using Vertex = geometry::optimization::GraphOfConvexSets::Vertex;
using solvers::Constraint;
using solvers::Cost;

struct GCSTrajectoryOptimizationConstructor {
  GCSTrajectoryOptimizationConstructor() = default;
  /** The order of the Bézier trajectory within a region.
  It will have order + 1 control points.
  The order must be at least 1.
  */
  int order{3};

  /** The maximum duration spend in a region (seconds).
  Some solvers struggle numerically with large values.
  */
  double d_max{20};

  /** Some cost and constraints are only convex for d > 0.
  For example the perspective quadratic cost of the path energy ||ṙ(s)||² / d
  becomes non-convex for d = 0. Otherwise d_min can be set to 0.
  */
  double d_min{1e-6};

  /** Dimension of the configuration space.*/
  int dimension;
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

  @param gcs_constructor includes settings to control the construction of the
  construction of the graph of convex sets.
  */
  GCSTrajectoryOptimization(
      const GCSTrajectoryOptimizationConstructor& constructor =
          GCSTrajectoryOptimizationConstructor());

  /** Returns the number of position variables. */
  int num_positions() const { return constructor_.dimension; };

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

  /** Add a subgraph to the graph of convex sets.
  @param regions is a list of collision free polytopes in configuration space.
  @param name must be a unique name of the subgraph.
  */
  void AddSubgraph(const std::vector<HPolyhedron>& regions,
                   const std::string& name);

  /** Add a point to the graph of convex sets.
  @param x is the position of the point.
  @param from_subgraph is the name of the subgraph the incoming edges of 
    the point will be added to. For example, if the point is a goal point
    one might leave the to_subgraph empty.
  @param to_subgraph is the name of the subgraph the outgoing edges of 
    the point will be added to. For example, if the point is a start point
    one might leave the from_subgraph empty.

  To add a midpoint, one would set both to_subgraph and from_subgraph so that 
  the point is connected to the subgraphs on both sides.

  @return the id of the vertex in the graph of convex sets.
  */
  VertexId AddPoint(const Eigen::Ref<const Eigen::VectorXd>& x,
                const std::string& from_subgraph = "",
                const std::string& to_subgraph = "");

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

  /** Adds multiple Perspective Quadratic Costs on the upper bound of the path
  energy. We upper bound the path integral by the sum of the distances between
  the control points divided by the duration. For Bezier curves,
  this is equivalent to the sum of the L2Norm of the derivative control points
  of the curve divided by the order and the duration.

  Note that for the perspective quadratic cost to be convex, the d_min must be
  greater than 0.
  */
  void AddPathEnergyCost(double weight = 1.0);

  /** Adds a linear velocity constraints to the entire graph `lb` ≤ q̈(t) ≤ `ub`.
   */
  void AddVelocityBounds(const Eigen::Ref<const Eigen::VectorXd>& lb,
                         const Eigen::Ref<const Eigen::VectorXd>& ub);

  trajectories::BsplineTrajectory<double> SolvePath(VertexId source_id, VertexId target_id,
      const GraphOfConvexSetsOptions& options);

 private:
  HPolyhedron time_scaling_set_{};
  std::map<std::string, std::vector<Vertex*>> subgraphs_{};
  std::map<std::string, std::vector<HPolyhedron>> subgraph_regions_{};
  GCSTrajectoryOptimizationConstructor constructor_;

  Vertex* source_{};
  Vertex* target_{};

  std::vector<std::shared_ptr<Cost>> edge_cost_{};
  std::vector<std::shared_ptr<Constraint>> edge_constraint_{};
  std::vector<std::shared_ptr<Constraint>> continuity_constraint_{};

  std::map<VertexId, bool> is_source_{};

  Eigen::VectorX<symbolic::Variable> u_duration_;
  Eigen::VectorX<symbolic::Variable> u_vars_;

  // r(s)
  trajectories::BsplineTrajectory<symbolic::Expression> u_r_trajectory_;
  GraphOfConvexSets gcs_{GraphOfConvexSets()};
};

}  // namespace trajectory_optimization
}  // namespace planning
}  // namespace drake
