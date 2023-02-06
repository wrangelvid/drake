#include "drake/planning/trajectory_optimization/gcs_trajectory_optimization.h"

#include "drake/common/pointer_cast.h"
#include "drake/common/symbolic/decompose.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/point.h"
#include "drake/math/matrix_util.h"

namespace drake {
namespace planning {
namespace trajectory_optimization {

using Eigen::MatrixXd;
using Eigen::VectorXd;
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
    const std::vector<HPolyhedron>& regions,
    const GCSTrajectoryOptimizationConstructor& constructor)
    : regions_(regions),
      constructor_(constructor),
      num_positions_(regions[0].ambient_dimension()) {
  DRAKE_DEMAND(constructor_.order > 0);

  for (const auto& region : regions_) {
    DRAKE_DEMAND(region.ambient_dimension() == num_positions());
  }

  // Add Regions with time scaling set.
  MatrixXd A_time(2, 1);
  A_time.row(0) = MatrixXd::Identity(1, 1);
  A_time.row(1) = -MatrixXd::Identity(1, 1);
  VectorXd b_time = -constructor_.d_min * VectorXd::Ones(2);
  b_time(0) = constructor_.d_max;

  const HPolyhedron time_scaling_set = HPolyhedron(A_time, b_time);

  for (const auto& region : regions_) {
    gcs_.AddVertex(region.CartesianPower(constructor_.order + 1)
                       .CartesianProduct(time_scaling_set));
  };

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
  // Check for overlap between regions.
  auto vertices = gcs_.Vertices();
  for (size_t i = 0; i < regions_.size(); i++) {
    for (size_t j = i + 1; j < regions_.size(); j++) {
      if (regions_[i].IntersectsWith(regions_[j])) {
        // Regions are overlapping, add edge.
        auto u = vertices[i];
        auto v = vertices[j];
        // Add path continuity constraints.
        gcs_.AddEdge(u->id(), v->id(), u->name() + " -> " + v->name())
            ->AddConstraint(Binding<Constraint>(path_continuity_constraint,
                                                {u->x(), v->x()}));
        gcs_.AddEdge(v->id(), u->id(), v->name() + " -> " + u->name())
            ->AddConstraint(Binding<Constraint>(path_continuity_constraint,
                                                {v->x(), u->x()}));
      }
    }
  }
}

void GCSTrajectoryOptimization::AddTimeCost(double weight) {
  // Add time cost to all edges.
  Eigen::MatrixXd M(1, u_vars_.size());
  DecomposeLinearExpressions(u_duration_.cast<Expression>(), u_vars_, &M);
  auto time_cost = std::make_shared<LinearCost>(weight * M.row(0), 0.0);

  edge_cost_.push_back(time_cost);

  for (const auto& e : gcs_.Edges()) {
    if (source_ != nullptr && e->u().id() == source_->id()) {
      continue;
    }
    e->AddCost(Binding<LinearCost>(time_cost, e->xu()));
  }
  // TODO(wrangelvid): Re add source and target.
}

void GCSTrajectoryOptimization::AddPathLengthCost(double weight) {
  /*
    We will upper bound the path integral by the sum of the distances between
    the control points. \sum_{i=0}^{n-1} ||r_i - r_{i+1}||_2,
    where n is the order of the curve.

    In the case of a Bezier curve, the path length is given by the integral of
    the norm of the derivative of the curve.
    We can formulate the cost as \sum_{i=0}^{n-1} ||dr_i||_2 / order.
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
    edge_cost_.push_back(path_length_cost);

    // Add path length cost to all edges.
    for (auto& edge : gcs_.Edges()) {
      if (source_ != nullptr && edge->u().id() == source_->id()) {
        continue;
      }
      edge->AddCost(Binding<L2NormCost>(path_length_cost, edge->xu()));
    }
  }
}

void GCSTrajectoryOptimization::AddPathEnergyCost(double weight) {
  /*
  TODO(wrangelvid): write some simple doc string here.
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
    edge_cost_.push_back(energy_cost);

    // Add path length cost to all edges.
    for (auto& edge : gcs_.Edges()) {
      if (source_ != nullptr && edge->u().id() == source_->id()) {
        continue;
      }
      edge->AddCost(Binding<PerspectiveQuadraticCost>(energy_cost, edge->xu()));
    }
  }
}

void GCSTrajectoryOptimization::AddSourceTarget(
    const Eigen::Ref<const Eigen::VectorXd>& source,
    const Eigen::Ref<const Eigen::VectorXd>& target) {
  DRAKE_DEMAND(source.size() == num_positions());
  DRAKE_DEMAND(target.size() == num_positions());

  // TODO(wrangelvid): add inital/final velocity constraints.

  // Remove existing source and target constraints.
  if (source_ != nullptr) {
    gcs_.RemoveVertex(source_->id());
  }
  if (target_ != nullptr) {
    gcs_.RemoveVertex(target_->id());
  }

  // Add source and target vertices.
  auto vertices = gcs_.Vertices();
  source_ = gcs_.AddVertex(Point(source), "source");
  target_ = gcs_.AddVertex(Point(target), "target");

  // Check if source and target can be connected to the graph.
  std::vector<int> source_edges_index;
  std::vector<int> target_edges_index;
  for (size_t i = 0; i < regions_.size(); i++) {
    if (regions_[i].PointInSet(source)) {
      source_edges_index.push_back(i);
    }
    if (regions_[i].PointInSet(target)) {
      target_edges_index.push_back(i);
    }
  }

  if (source_edges_index.empty()) {
    throw std::runtime_error("Source vertex is not connected.");
  }

  if (target_edges_index.empty()) {
    throw std::runtime_error("Target vertex is not connected.");
  }

  // Connect source and target to the graph.
  for (const auto& i : source_edges_index) {
    auto edge = gcs_.AddEdge(source_->id(), vertices[i]->id(),
                             "source -> " + vertices[i]->name());

    // Match the position of the source and the first control point.
    for (int j = 0; j < num_positions(); j++) {
      edge->AddConstraint(edge->xu()[j] == edge->xv()[j]);
    }
  }

  for (const auto& i : target_edges_index) {
    auto edge = gcs_.AddEdge(vertices[i]->id(), target_->id(),
                             vertices[i]->name() + " -> target");

    // Match the position of the target and the last control point.
    for (int j = 0; j < num_positions(); j++) {
      edge->AddConstraint(
          edge->xu()[num_positions() * constructor_.order + j] ==
          edge->xv()[j]);
    }

    // Add edge costs.
    for (const auto& cost : edge_cost_) {
      edge->AddCost(Binding<Cost>(cost, edge->xu()));
    }

    // Add edge Constraints.
    for (const auto& constraint : edge_constraint_) {
      edge->AddConstraint(Binding<Constraint>(constraint, edge->xu()));
    }
  }
}

void GCSTrajectoryOptimization::AddVelocityBounds(
    const Eigen::Ref<const Eigen::VectorXd>& lb,
    const Eigen::Ref<const Eigen::VectorXd>& ub) {
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
    edge_constraint_.push_back(velocity_constraint);

    // Add velocity bounds to all edges.
    for (auto& edge : gcs_.Edges()) {
      if (source_ != nullptr && edge->u().id() == source_->id()) {
        continue;
      }
      edge->AddConstraint(
          Binding<LinearConstraint>(velocity_constraint, edge->xu()));
    }
  }

  // TODO(wrangelvid): Re add source and target.
}

BsplineTrajectory<double> GCSTrajectoryOptimization::SolvePath(
    const GraphOfConvexSetsOptions& options) {
  auto result = gcs_.SolveShortestPath(source_->id(), target_->id(), options);

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
  std::vector<VertexId> visited_vertex_ids{source_->id()};
  std::vector<VertexId> path_vertex_ids{source_->id()};
  std::vector<Edge*> path_edges{};
  VertexId target_id = target_->id();
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
    Eigen::MatrixX<double> edge_path_points =
        Eigen::Map<Eigen::MatrixX<double>>(
            result.GetSolution(edge->xv()).data(), num_positions(),
            constructor_.order + 1);
    for (int i = 0; i < edge_path_points.cols(); ++i) {
      control_points.push_back(edge_path_points.col(i));
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
