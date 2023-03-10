#include "drake/planning/trajectory_optimization/gcs_trajectory_optimization.h"

#include "iostream"

#include "drake/common/pointer_cast.h"
#include "drake/common/symbolic/decompose.h"
#include "drake/geometry/optimization/cartesian_product.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/point.h"
#include "drake/math/matrix_util.h"

namespace drake {
namespace planning {
namespace trajectory_optimization {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using geometry::optimization::CartesianProduct;
using geometry::optimization::HPolyhedron;
using geometry::optimization::Point;
using math::BsplineBasis;
using math::EigenToStdVector;
using solvers::Binding;
using solvers::L2NormCost;
using solvers::LinearConstraint;
using solvers::LinearCost;
using solvers::LinearEqualityConstraint;
using solvers::PerspectiveQuadraticCost;
using std::numeric_limits;
using symbolic::DecomposeLinearExpressions;
using symbolic::Expression;
using symbolic::MakeMatrixContinuousVariable;
using symbolic::MakeVectorContinuousVariable;
using trajectories::BsplineTrajectory;
using Edge = geometry::optimization::GraphOfConvexSets::Edge;
using VertexId = geometry::optimization::GraphOfConvexSets::VertexId;
using EdgeId = geometry::optimization::GraphOfConvexSets::EdgeId;
using geometry::optimization::GraphOfConvexSetsOptions;

const double inf = std::numeric_limits<double>::infinity();

GCSTrajectoryOptimization::GCSTrajectoryOptimization(
    const GCSTrajectoryOptimizationConstructor& constructor)
    : constructor_(constructor) {
  DRAKE_DEMAND(constructor_.order > 0);
  // Make time scaling set.
  time_scaling_set_ =
      HPolyhedron::MakeBox(constructor_.d_min * Eigen::VectorXd::Ones(1),
                           constructor_.d_max * Eigen::VectorXd::Ones(1));

  // Formulate edge costs and constraints.
  auto u_control = MakeMatrixContinuousVariable(num_positions(),
                                                constructor_.order + 1, "xu");
  auto v_control = MakeMatrixContinuousVariable(num_positions(),
                                                constructor_.order + 1, "xv");

  auto u_control_vars = Eigen::Map<Eigen::VectorX<symbolic::Variable>>(
      u_control.data(), u_control.size());
  auto v_control_vars = Eigen::Map<Eigen::VectorX<symbolic::Variable>>(
      v_control.data(), v_control.size());

  u_duration_ = MakeVectorContinuousVariable(1, "Tu");
  auto v_duration = MakeVectorContinuousVariable(1, "Tv");

  u_vars_ = solvers::ConcatenateVariableRefList({u_control_vars, u_duration_});
  auto edge_vars = solvers::ConcatenateVariableRefList(
      {u_control_vars, u_duration_, v_control_vars, v_duration});

  u_r_trajectory_ = BsplineTrajectory<Expression>(
      BsplineBasis<Expression>(constructor_.order + 1, constructor_.order + 1,
                               math::KnotVectorType::kClampedUniform, 0, 1),
      EigenToStdVector<Expression>(u_control.cast<Expression>()));

  auto v_r_trajectory = BsplineTrajectory<Expression>(
      BsplineBasis<Expression>(constructor_.order + 1, constructor_.order + 1,
                               math::KnotVectorType::kClampedUniform, 0, 1),
      EigenToStdVector<Expression>(v_control.cast<Expression>()));

  // Zeroth order continuity constraints.
  const Eigen::VectorX<Expression> path_continuity_error =
      v_r_trajectory.control_points().front() -
      u_r_trajectory_.control_points().back();
  Eigen::MatrixXd M(num_positions(), edge_vars.size());
  DecomposeLinearExpressions(path_continuity_error, edge_vars, &M);

  auto path_continuity_constraint = std::make_shared<LinearEqualityConstraint>(
      M, VectorXd::Zero(num_positions()));
  continuity_constraint_.push_back(path_continuity_constraint);
}

void GCSTrajectoryOptimization::AddSubgraph(
    const std::vector<HPolyhedron>& regions, const std::string& name) {
  // TODO(wrangelvid): Allow for custom edge variables.
  // TODO(wrangelvid): Add full-dimension overlap graph generation.
  //  Check if region with name already exists.
  DRAKE_DEMAND(subgraphs_.find(name) == subgraphs_.end());

  // Make sure all regions have the same ambient dimension.
  for (const auto& region : regions) {
    DRAKE_DEMAND(region.ambient_dimension() == num_positions());
  }

  // Add Regions with time scaling set.
  std::vector<Vertex*> subgraph_vertices;
  int vertex_count = 0;
  for (const auto& region : regions) {
    subgraph_vertices.push_back(
        gcs_.AddVertex(region.CartesianPower(constructor_.order + 1)
                           .CartesianProduct(time_scaling_set_),
                       name + ": " + std::to_string(vertex_count)));
    vertex_count++;
  }

  // Check for overlap between regions.
  // TODO(wrangelvid): This is O(n^2) and can be improved.
  for (size_t i = 0; i < regions.size(); i++) {
    for (size_t j = i + 1; j < regions.size(); j++) {
      if (regions[i].IntersectsWith(regions[j])) {
        // Regions are overlapping, add edge.
        auto u = subgraph_vertices[i];
        auto v = subgraph_vertices[j];
        // Add path continuity constraints.
        auto uv_edge =
            gcs_.AddEdge(u->id(), v->id(), u->name() + " -> " + v->name());
        auto vu_edge =
            gcs_.AddEdge(v->id(), u->id(), v->name() + " -> " + u->name());

        // Add path continuity constraints.
        for (const auto& path_continuity_constraint : continuity_constraint_) {
          uv_edge->AddConstraint(Binding<Constraint>(path_continuity_constraint,
                                                     {u->x(), v->x()}));
          vu_edge->AddConstraint(Binding<Constraint>(path_continuity_constraint,
                                                     {v->x(), u->x()}));
        }
      }
    }
  }

  // Add subgraph to subgraphs.
  subgraphs_[name] = subgraph_vertices;
  subgraph_regions_[name] = regions;
}

void GCSTrajectoryOptimization::AddSubspace(const ConvexSet& region,
                                            const std::string& name,
                                            const std::string& from_subgraph,
                                            const std::string& to_subgraph,
                                            double delay) {
  //  Check if region with name already exists.
  DRAKE_DEMAND(subspaces_.find(name) == subspaces_.end());
  DRAKE_DEMAND(region.ambient_dimension() == num_positions());

  if (to_subgraph.empty() && from_subgraph.empty()) {
    throw std::runtime_error(
        "Either to_subgraph or from_subgraph must be specified.");
  }

  if (delay < 0) {
    throw std::runtime_error("Delay must be non-negative.");
  }

  std::vector<Vertex*> to_subgraph_vertices;
  std::vector<Vertex*> from_subgraph_vertices;
  if (!to_subgraph.empty()) {
    if (subgraphs_.find(to_subgraph) == subgraphs_.end()) {
      throw std::runtime_error("Subgraph " + to_subgraph + " does not exist.");
    } else {
      // Check if point is in any of the subgraph regions.
      for (size_t i = 0; i < subgraph_regions_[to_subgraph].size(); i++) {
        // TODO(wrangelvid): Maybe there is a better way of doing this without
        //  the need to store the regions.
        if (typeid(region) == typeid(Point)) {
          // Checking if a point is in a region is faster than checking for
          // intersection.
          if (subgraph_regions_[to_subgraph][i].PointInSet(
                  static_cast<const Point&>(region).x())) {
            to_subgraph_vertices.push_back(subgraphs_[to_subgraph][i]);
          }
        } else {
          if (subgraph_regions_[to_subgraph][i].IntersectsWith(region)) {
            to_subgraph_vertices.push_back(subgraphs_[to_subgraph][i]);
          }
        }
      }
      if (to_subgraph_vertices.empty()) {
        throw std::runtime_error("Set does not intersect with subgraph " +
                                 to_subgraph);
      }
    }
  }

  if (!from_subgraph.empty()) {
    if (subgraphs_.find(from_subgraph) == subgraphs_.end()) {
      throw std::runtime_error("Subgraph " + from_subgraph +
                               " does not exist.");
    } else {
      // Check if point is in any of the subgraph regions.
      for (size_t i = 0; i < subgraph_regions_[from_subgraph].size(); i++) {
        if (typeid(region) == typeid(Point)) {
          // Checking if a point is in a region is faster than checking for
          // intersection.
          if (subgraph_regions_[from_subgraph][i].PointInSet(
                  static_cast<const Point&>(region).x())) {
            from_subgraph_vertices.push_back(subgraphs_[from_subgraph][i]);
          }
        } else {
          if (subgraph_regions_[from_subgraph][i].IntersectsWith(region)) {
            from_subgraph_vertices.push_back(subgraphs_[from_subgraph][i]);
          }
        }
      }
      if (from_subgraph_vertices.empty()) {
        throw std::runtime_error("Set does not intersect with subgraph " +
                                 from_subgraph);
      }
    }
  }
  // Make time scaling set.
  auto time_scaling_set = HPolyhedron::MakeBox(delay * VectorXd::Ones(1),
                                               delay * VectorXd::Ones(1));

  // Add point to graph.
  auto region_vertex =
      gcs_.AddVertex(CartesianProduct(region, time_scaling_set), name);

  // Add edges to subgraph vertices.
  for (const auto& vertex : to_subgraph_vertices) {
    auto edge = gcs_.AddEdge(region_vertex->id(), vertex->id(),
                             region_vertex->name() + " -> " + vertex->name());

    // Match the position of the source and the first control point.
    for (int j = 0; j < num_positions(); j++) {
      edge->AddConstraint(edge->xu()[j] == edge->xv()[j]);
    }
  }
  for (const auto& vertex : from_subgraph_vertices) {
    auto edge = gcs_.AddEdge(vertex->id(), region_vertex->id(),
                             vertex->name() + " -> " + region_vertex->name());

    // Match the position of the target and the last control point.
    for (int j = 0; j < num_positions(); j++) {
      edge->AddConstraint(
          edge->xu()[num_positions() * constructor_.order + j] ==
          edge->xv()[j]);
    }
  }

  subspaces_[name] = region_vertex;
}

void GCSTrajectoryOptimization::AddTimeCost(double weight,
                                            const std::string& subgraph) {
  // The time cost is the sum of duration variables ∑ dᵢ
  auto time_cost =
      std::make_shared<LinearCost>(weight * Eigen::VectorXd::Ones(1), 0.0);

  for (const auto& [name, vertices] : subgraphs_) {
    if (!subgraph.empty() && name != subgraph) {
      // If subgraph is not empty, we only add the cost to the specified
      // subgraph.
      continue;
    }
    for (const auto& v : vertices) {
      // The duration variable is the last element of the vertex.
      v->AddCost(Binding<LinearCost>(time_cost, v->x().tail(1)));
    }
  }
}

void GCSTrajectoryOptimization::AddPathLengthCost(double weight,
                                                  const std::string& subgraph) {
  /*
    We will upper bound the path integral by the sum of the distances between
    the control points. ∑ ||rᵢ − rᵢ₊₁||₂

    In the case of a Bezier curve, the path length is given by the integral of
    the norm of the derivative of the curve.

    So the path length cost becomes: ∑ ||ṙᵢ||₂ / order
  */
  auto weight_matrix =
      weight * MatrixXd::Identity(num_positions(), num_positions());

  auto u_rdot_control =
      dynamic_pointer_cast_or_throw<BsplineTrajectory<symbolic::Expression>>(
          u_r_trajectory_.MakeDerivative())
          ->control_points();

  for (size_t i = 0; i < u_rdot_control.size(); i++) {
    Eigen::MatrixXd M(u_rdot_control[i].rows(), u_vars_.size());
    DecomposeLinearExpressions(u_rdot_control[i] / constructor_.order, u_vars_,
                               &M);

    auto path_length_cost = std::make_shared<L2NormCost>(
        weight_matrix * M, Eigen::VectorXd::Zero(num_positions()));

    for (const auto& [name, vertices] : subgraphs_) {
      if (!subgraph.empty() && name != subgraph) {
        // If subgraph is not empty, we only add the cost to the specified
        // subgraph.
        continue;
      }
      for (const auto& v : vertices) {
        // The duration variable is the last element of the vertex.
        v->AddCost(Binding<L2NormCost>(path_length_cost, v->x()));
      }
    }
  }
}

void GCSTrajectoryOptimization::AddPathEnergyCost(double weight,
                                                  const std::string& subgraph) {
  /*
    We will upper bound the path integral by the sum of the distances between
    the control points. ∑ ||rᵢ − rᵢ₊₁||₂

    In the case of a Bezier curve, the path length is given by the integral of
    the norm of the derivative of the curve.  ∑ ||ṙᵢ||₂ / order

    So the path energy cost becomes: ∑ ||ṙᵢ||₂ / (order * dᵢ)
  */
  auto sqrt_weight_matrix =
      weight * MatrixXd::Identity(num_positions(), num_positions()).cwiseSqrt();

  auto u_rdot_control =
      dynamic_pointer_cast_or_throw<BsplineTrajectory<symbolic::Expression>>(
          u_r_trajectory_.MakeDerivative())
          ->control_points();

  Eigen::MatrixXd b_ctrl(u_duration_.rows(), u_vars_.size());
  DecomposeLinearExpressions(u_duration_.cast<Expression>(), u_vars_, &b_ctrl);

  for (size_t i = 0; i < u_rdot_control.size(); i++) {
    Eigen::MatrixXd A_ctrl(u_rdot_control[i].rows(), u_vars_.size());
    DecomposeLinearExpressions(u_rdot_control[i], u_vars_, &A_ctrl);
    Eigen::MatrixXd A(1 + num_positions(), A_ctrl.cols());
    A << constructor_.order * b_ctrl, sqrt_weight_matrix * A_ctrl;

    auto energy_cost = std::make_shared<PerspectiveQuadraticCost>(
        A, Eigen::VectorXd::Zero(1 + num_positions()));

    for (const auto& [name, vertices] : subgraphs_) {
      if (!subgraph.empty() && name != subgraph) {
        // If subgraph is not empty, we only add the cost to the specified
        // subgraph.
        continue;
      }
      for (const auto& v : vertices) {
        // The duration variable is the last element of the vertex.
        v->AddCost(Binding<PerspectiveQuadraticCost>(energy_cost, v->x()));
      }
    }
  }
}

void GCSTrajectoryOptimization::AddVelocityBounds(
    const Eigen::Ref<const Eigen::VectorXd>& lb,
    const Eigen::Ref<const Eigen::VectorXd>& ub, const std::string& subgraph) {
  DRAKE_DEMAND(lb.size() == num_positions());
  DRAKE_DEMAND(ub.size() == num_positions());

  // We have q̇(t) = drds * dsdt = ṙ(s) / duration, and duration >= 0, so we
  // use duration * lb <= ṙ(s) <= duration * ub. But will be reformulated as:
  // - inf <=   duration * lb - ṙ(s) <= 0
  // - inf <= - duration * ub + ṙ(s) <= 0

  // This also leverages the convex hull property of the B-splines: if all of
  // the control points satisfy these convex constraints and the curve is
  // inside the convex hull of these constraints, then the curve satisfies the
  // constraints for all t.

  auto u_rdot_control =
      dynamic_pointer_cast_or_throw<BsplineTrajectory<symbolic::Expression>>(
          u_r_trajectory_.MakeDerivative())
          ->control_points();

  Eigen::MatrixXd b_ctrl(u_duration_.rows(), u_vars_.size());
  DecomposeLinearExpressions(u_duration_.cast<Expression>(), u_vars_, &b_ctrl);

  for (size_t i = 0; i < u_rdot_control.size(); i++) {
    Eigen::MatrixXd A_ctrl(u_rdot_control[i].rows(), u_vars_.size());
    DecomposeLinearExpressions(u_rdot_control[i], u_vars_, &A_ctrl);
    Eigen::MatrixXd A(2 * num_positions(), A_ctrl.cols());
    A << A_ctrl - ub * b_ctrl, -A_ctrl + lb * b_ctrl;

    auto velocity_constraint = std::make_shared<LinearConstraint>(
        A, Eigen::VectorXd::Constant(2 * num_positions(), -inf),
        Eigen::VectorXd::Zero(2 * num_positions()));

    for (const auto& [name, vertices] : subgraphs_) {
      if (!subgraph.empty() && name != subgraph) {
        // If subgraph is not empty, we only add the cost to the specified
        // subgraph.
        continue;
      }
      for (const auto& v : vertices) {
        // The duration variable is the last element of the vertex.
        v->AddConstraint(
            Binding<LinearConstraint>(velocity_constraint, v->x()));
      }
    }
  }
}

BsplineTrajectory<double> GCSTrajectoryOptimization::SolvePath(
    std::string& source_subspace, std::string& target_subspace,
    const GraphOfConvexSetsOptions& options) {
  // Check if the source and target subspaces are valid.
  if (subspaces_.find(source_subspace) == subspaces_.end()) {
    throw std::runtime_error("Source subspace does not exist.");
  }
  if (subspaces_.find(target_subspace) == subspaces_.end()) {
    throw std::runtime_error("Target subspace does not exist.");
  }

  VertexId source_id = subspaces_[source_subspace]->id();
  VertexId target_id = subspaces_[target_subspace]->id();
  auto result = gcs_.SolveShortestPath(source_id, target_id, options);

  if (!result.is_success()) {
    throw std::runtime_error("GCS failed to find a path.");
  }

  // Extract the flow from the solution.
  std::map<VertexId, std::vector<Edge*>> outgoing_edges;
  std::map<EdgeId, double> flows;
  for (auto& edge : gcs_.Edges()) {
    outgoing_edges[edge->u().id()].push_back(edge);
    flows[edge->id()] = result.GetSolution(edge->phi());
  }

  // Extract the path by traversing the graph with a depth first search.
  std::vector<VertexId> visited_vertex_ids{source_id};
  std::vector<VertexId> path_vertex_ids{source_id};
  std::vector<Edge*> path_edges{};
  while (path_vertex_ids.back() != target_id) {
    // Find the edge with the maximum flow from the current node.
    double maximum_flow = 0;
    VertexId max_flow_vertex_id;
    Edge* max_flow_edge = nullptr;
    for (Edge* e : outgoing_edges[path_vertex_ids.back()]) {
      double next_flow = flows[e->id()];
      VertexId next_vertex_id = e->v().id();

      // If the edge has not been visited and has a flow greater than the
      // current maximum, update the maximum flow and the vertex id.
      if (std::find(visited_vertex_ids.begin(), visited_vertex_ids.end(),
                    e->v().id()) == visited_vertex_ids.end() &&
          next_flow > maximum_flow && next_flow > options.flow_tolerance) {
        maximum_flow = next_flow;
        max_flow_vertex_id = next_vertex_id;
        max_flow_edge = e;
      }
    }

    if (max_flow_edge == nullptr) {
      // If no candidate edges are found, backtrack to the previous node and
      // continue the search.
      path_vertex_ids.pop_back();
      continue;
    } else {
      // If the maximum flow is non-zero, add the vertex to the path and
      // continue the search.
      visited_vertex_ids.push_back(max_flow_vertex_id);
      path_vertex_ids.push_back(max_flow_vertex_id);
      path_edges.push_back(max_flow_edge);
    }
  }

  // Extract the path from the edges.
  std::vector<double> path_times(constructor_.order + 1, 0.0);
  std::vector<Eigen::MatrixX<double>> control_points;
  for (auto& edge : path_edges) {
    // Extract the control points from the solution.
    // Sometimes the solution goes through a point which has a single control
    // point.
    int num_control_points = (edge->xv().size() - 1) / num_positions();
    Eigen::MatrixX<double> edge_path_points =
        Eigen::Map<Eigen::MatrixX<double>>(
            result.GetSolution(edge->xv()).data(), num_positions(),
            num_control_points);
    for (int i = 0; i < constructor_.order + 1; ++i) {
      if (i < edge_path_points.cols()) {
        control_points.push_back(edge_path_points.col(i));
      } else {
        control_points.push_back(
            edge_path_points.col(edge_path_points.cols() - 1));
      }
    }

    if (edge->v().id() == target_id) {
      std::vector<double> next_path_times(constructor_.order + 1,
                                          path_times.back());
      path_times.insert(path_times.end(), next_path_times.begin(),
                        next_path_times.end());
      break;
    }

    // Extract the duration from the solution.
    double duration = result.GetSolution(edge->xv()).tail<1>().value();
    std::vector<double> next_path_times(constructor_.order + 1,
                                        path_times.back() + duration);
    path_times.insert(path_times.end(), next_path_times.begin(),
                      next_path_times.end());
  }
  return BsplineTrajectory<double>(
      BsplineBasis(constructor_.order + 1, path_times), control_points);
}

}  // namespace trajectory_optimization
}  // namespace planning
}  // namespace drake
