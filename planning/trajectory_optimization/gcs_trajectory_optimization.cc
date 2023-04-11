#include "drake/planning/trajectory_optimization/gcs_trajectory_optimization.h"

#include "iostream"

#include "drake/common/pointer_cast.h"
#include "drake/common/symbolic/decompose.h"
#include "drake/geometry/optimization/cartesian_product.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/point.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace planning {
namespace trajectory_optimization {

using Subgraph = GCSTrajectoryOptimization::Subgraph;
using SubgraphEdges = GCSTrajectoryOptimization::SubgraphEdges;

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
using solvers::LorentzConeConstraint;
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
using drake::geometry::optimization::ConvexSets;
using geometry::optimization::GraphOfConvexSetsOptions;

const double inf = std::numeric_limits<double>::infinity();

Subgraph::Subgraph(const ConvexSets& regions, int order,
                   std::vector<std::pair<int, int>>& edges_between_regions,
                   const std::string& name, GCSTrajectoryOptimization* gcs)
    : regions_(regions), order_(order), name_(name), gcs_(gcs) {
  if (order_ < 0) {
    throw std::runtime_error("Order must be positive.");
  }

  // Make sure all regions have the same ambient dimension.
  for (const auto& region : regions_) {
    DRAKE_DEMAND(region->ambient_dimension() == gcs->num_positions());
  }

  // Formulate edge costs and constraints.
  auto u_control =
      MakeMatrixContinuousVariable(gcs_->num_positions(), order_ + 1, "xu");
  auto v_control =
      MakeMatrixContinuousVariable(gcs_->num_positions(), order_ + 1, "xv");

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
      BsplineBasis<Expression>(order_ + 1, order_ + 1,
                               math::KnotVectorType::kClampedUniform, 0, 1),
      EigenToStdVector<Expression>(u_control.cast<Expression>()));

  auto v_r_trajectory = BsplineTrajectory<Expression>(
      BsplineBasis<Expression>(order_ + 1, order_ + 1,
                               math::KnotVectorType::kClampedUniform, 0, 1),
      EigenToStdVector<Expression>(v_control.cast<Expression>()));

  // Zeroth order continuity constraints.
  const Eigen::VectorX<Expression> path_continuity_error =
      v_r_trajectory.control_points().front() -
      u_r_trajectory_.control_points().back();
  Eigen::MatrixXd M(gcs_->num_positions(), edge_vars.size());
  DecomposeLinearExpressions(path_continuity_error, edge_vars, &M);

  auto path_continuity_constraint = std::make_shared<LinearEqualityConstraint>(
      M, VectorXd::Zero(gcs_->num_positions()));

  // Add Regions with time scaling set.
  for (size_t i = 0; i < regions_.size(); i++) {
    // Assign each control point to a separate set.
    ConvexSets vertex_set;
    for (int j = 0; j < order + 1; j++) {
      vertex_set.emplace_back(*regions_[i]);
    }
    // Add time scaling set.
    vertex_set.emplace_back(gcs->time_scaling_set_);

    vertices_.emplace_back(gcs_->gcs_.AddVertex(
        CartesianProduct(vertex_set), name + ": " + std::to_string(i)));
  }

  // Connect vertices with edges.
  for (const auto& [u_idx, v_idx] : edges_between_regions) {
    // Add edge.
    auto u = vertices_[u_idx];
    auto v = vertices_[v_idx];
    auto uv_edge = gcs_->gcs_.AddEdge(*u, *v, u->name() + " -> " + v->name());

    edges_.emplace_back(uv_edge);

    // Add path continuity constraints.
    uv_edge->AddConstraint(
        Binding<Constraint>(path_continuity_constraint, {u->x(), v->x()}));
  }

  // Apply global graph constraints and costs
  for (auto weight : gcs_->global_time_costs_) {
    Subgraph::AddTimeCost(weight);
  }

  if (order_ > 0) {
    // These cost rely on the derivative of the trajectory.
    for (auto weight_matrix : gcs_->global_path_length_costs_) {
      Subgraph::AddPathLengthCost(weight_matrix);
    }

    for (auto weight_matrix : gcs_->global_path_energy_costs_) {
      Subgraph::AddPathEnergyCost(weight_matrix);
    }

    for (auto& [lb, ub] : gcs_->global_velocity_bounds_) {
      Subgraph::AddVelocityBounds(lb, ub);
    }
  }
}

void Subgraph::AddTimeCost(double weight) {
  // The time cost is the sum of duration variables ∑ dᵢ
  auto time_cost =
      std::make_shared<LinearCost>(weight * Eigen::VectorXd::Ones(1), 0.0);

  for (const auto& v : vertices_) {
    // The duration variable is the last element of the vertex.
    v->AddCost(Binding<LinearCost>(time_cost, v->x().tail(1)));
  }
}

void Subgraph::AddPathLengthCost(const Eigen::MatrixXd& weight_matrix) {
  /*
    We will upper bound the path integral by the sum of the distances between
    the control points. ∑ ||rᵢ − rᵢ₊₁||₂

    In the case of a Bezier curve, the path length is given by the integral of
    the norm of the derivative of the curve.

    So the path length cost becomes: ∑ ||ṙᵢ||₂ / order
  */
  DRAKE_DEMAND(weight_matrix.rows() == gcs_->num_positions());
  DRAKE_DEMAND(weight_matrix.cols() == gcs_->num_positions());

  if (order() == 0) {
    throw std::runtime_error(
        "Path length cost is not defined for a set of order 0.");
  }

  auto u_rdot_control =
      dynamic_pointer_cast_or_throw<BsplineTrajectory<symbolic::Expression>>(
          u_r_trajectory_.MakeDerivative())
          ->control_points();

  for (size_t i = 0; i < u_rdot_control.size(); i++) {
    Eigen::MatrixXd M(u_rdot_control[i].rows(), u_vars_.size());
    DecomposeLinearExpressions(u_rdot_control[i] / order(), u_vars_, &M);

    auto path_length_cost = std::make_shared<L2NormCost>(
        weight_matrix * M, Eigen::VectorXd::Zero(gcs_->num_positions()));

    for (const auto& v : vertices_) {
      // The duration variable is the last element of the vertex.
      v->AddCost(Binding<L2NormCost>(path_length_cost, v->x()));
    }
  }
}

void Subgraph::AddPathEnergyCost(const Eigen::MatrixXd& weight_matrix) {
  /*
    We will upper bound the path integral by the sum of the distances between
    the control points. ∑ ||rᵢ − rᵢ₊₁||₂

    In the case of a Bezier curve, the path length is given by the integral of
    the norm of the derivative of the curve.  ∑ ||ṙᵢ||₂ / order

    So the path energy cost becomes: ∑ ||ṙᵢ||₂ / (order * dᵢ)
  */
  DRAKE_DEMAND(weight_matrix.rows() == gcs_->num_positions());
  DRAKE_DEMAND(weight_matrix.cols() == gcs_->num_positions());

  if (order() == 0) {
    throw std::runtime_error(
        "Path energy cost is not defined for a set of order 0.");
  }

  auto sqrt_weight_matrix = weight_matrix.cwiseSqrt();

  auto u_rdot_control =
      dynamic_pointer_cast_or_throw<BsplineTrajectory<symbolic::Expression>>(
          u_r_trajectory_.MakeDerivative())
          ->control_points();

  Eigen::MatrixXd b_ctrl(u_duration_.rows(), u_vars_.size());
  DecomposeLinearExpressions(u_duration_.cast<Expression>(), u_vars_, &b_ctrl);

  for (size_t i = 0; i < u_rdot_control.size(); i++) {
    Eigen::MatrixXd A_ctrl(u_rdot_control[i].rows(), u_vars_.size());
    DecomposeLinearExpressions(u_rdot_control[i], u_vars_, &A_ctrl);
    Eigen::MatrixXd A(1 + gcs_->num_positions(), A_ctrl.cols());
    A << order() * b_ctrl, sqrt_weight_matrix * A_ctrl;

    auto energy_cost = std::make_shared<PerspectiveQuadraticCost>(
        A, Eigen::VectorXd::Zero(1 + gcs_->num_positions()));

    for (const auto& v : vertices_) {
      // The duration variable is the last element of the vertex.
      v->AddCost(Binding<PerspectiveQuadraticCost>(energy_cost, v->x()));
    }
  }
}

void Subgraph::AddVelocityBounds(const Eigen::Ref<const Eigen::VectorXd>& lb,
                                 const Eigen::Ref<const Eigen::VectorXd>& ub) {
  DRAKE_DEMAND(lb.size() == gcs_->num_positions());
  DRAKE_DEMAND(ub.size() == gcs_->num_positions());

  if (order() == 0) {
    throw std::runtime_error(
        "Velocity Bounds are not defined for a set of order 0.");
  }

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
    Eigen::MatrixXd A(2 * gcs_->num_positions(), A_ctrl.cols());
    A << A_ctrl - ub * b_ctrl, -A_ctrl + lb * b_ctrl;

    auto velocity_constraint = std::make_shared<LinearConstraint>(
        A, Eigen::VectorXd::Constant(2 * gcs_->num_positions(), -inf),
        Eigen::VectorXd::Zero(2 * gcs_->num_positions()));

    for (const auto& v : vertices_) {
      // The duration variable is the last element of the vertex.
      v->AddConstraint(Binding<LinearConstraint>(velocity_constraint, v->x()));
    }
  }
}

SubgraphEdges::SubgraphEdges(const Subgraph* from, const Subgraph* to,
                             const ConvexSet* subspace,
                             GCSTrajectoryOptimization* gcs)
    : gcs_(gcs) {
  // Formulate edge costs and constraints.
  if (subspace != nullptr) {
    if (subspace->ambient_dimension() != gcs_->num_positions()) {
      throw std::runtime_error(
          "Subspace dimension must match the number of positions.");
    }
    if (typeid(*subspace) != typeid(Point) &&
        typeid(*subspace) != typeid(HPolyhedron)) {
      throw std::runtime_error("Subspace must be a Point or HPolyhedron.");
    }
  }

  auto u_control = MakeMatrixContinuousVariable(gcs_->num_positions(),
                                                from->order() + 1, "xu");
  auto v_control = MakeMatrixContinuousVariable(gcs_->num_positions(),
                                                to->order() + 1, "xv");

  auto u_control_vars = Eigen::Map<Eigen::VectorX<symbolic::Variable>>(
      u_control.data(), u_control.size());
  auto v_control_vars = Eigen::Map<Eigen::VectorX<symbolic::Variable>>(
      v_control.data(), v_control.size());

  auto u_duration = MakeVectorContinuousVariable(1, "Tu");
  auto v_duration = MakeVectorContinuousVariable(1, "Tv");

  auto edge_vars = solvers::ConcatenateVariableRefList(
      {u_control_vars, u_duration, v_control_vars, v_duration});

  auto u_r_trajectory = BsplineTrajectory<Expression>(
      BsplineBasis<Expression>(from->order() + 1, from->order() + 1,
                               math::KnotVectorType::kClampedUniform, 0, 1),
      EigenToStdVector<Expression>(u_control.cast<Expression>()));

  auto v_r_trajectory = BsplineTrajectory<Expression>(
      BsplineBasis<Expression>(to->order() + 1, to->order() + 1,
                               math::KnotVectorType::kClampedUniform, 0, 1),
      EigenToStdVector<Expression>(v_control.cast<Expression>()));

  // Zeroth order continuity constraints.
  const Eigen::VectorX<Expression> path_continuity_error =
      v_r_trajectory.control_points().front() -
      u_r_trajectory.control_points().back();
  Eigen::MatrixXd M(gcs_->num_positions(), edge_vars.size());
  DecomposeLinearExpressions(path_continuity_error, edge_vars, &M);

  auto path_continuity_constraint = std::make_shared<LinearEqualityConstraint>(
      M, VectorXd::Zero(gcs_->num_positions()));

  // TODO(wrangelvid) this can be parallelized.
  for (int i = 0; i < from->size(); i++) {
    for (int j = 0; j < to->size(); j++) {
      if (from->regions()[i]->IntersectsWith(*to->regions()[j])) {
        if (subspace != nullptr) {
          // Check if the regions are connected through the subspace.
          if (!RegionsConnectThroughSubspace(*from->regions()[i],
                                             *to->regions()[j], *subspace)) {
            continue;
          }
        }
        // Add edge.
        auto u = from->vertices()[i];
        auto v = to->vertices()[j];
        auto uv_edge =
            gcs_->gcs_.AddEdge(*u, *v, u->name() + " -> " + v->name());

        edges_.emplace_back(uv_edge);

        // Add path continuity constraints.
        uv_edge->AddConstraint(
            Binding<Constraint>(path_continuity_constraint, {u->x(), v->x()}));

        if (subspace != nullptr) {
          // Add subspace constraints to the first control point of the v
          // vertex. Since we are using zeroth order continuity, the last
          // control point
          auto vars = v->x().segment(0, gcs_->num_positions());
          solvers::MathematicalProgram prog{};
          const auto& x =
              prog.NewContinuousVariables(gcs_->num_positions(), "x");
          subspace->AddPointInSetConstraints(&prog, x);
          for (const auto& binding : prog.GetAllConstraints()) {
            const std::shared_ptr<Constraint>& constraint = binding.evaluator();
            uv_edge->AddConstraint(Binding<Constraint>(constraint, vars));
          }
        }
      }
    }
  }
}

bool SubgraphEdges::RegionsConnectThroughSubspace(const ConvexSet& A,
                                                  const ConvexSet& B,
                                                  const ConvexSet& subspace) {
  DRAKE_THROW_UNLESS(A.ambient_dimension() == B.ambient_dimension() &&
                     A.ambient_dimension() == subspace.ambient_dimension());
  if (typeid(subspace) == typeid(Point)) {
    // If the subspace is a point, then the point must be in both A and B.
    return A.PointInSet(static_cast<const Point&>(subspace).x()) &&
           B.PointInSet(static_cast<const Point&>(subspace).x());
  } else {
    // Otherwise, we can formulate a problem to check if a point is contained in
    // A, B and the subspace.
    solvers::MathematicalProgram prog{};
    const auto& x = prog.NewContinuousVariables(gcs_->num_positions(), "x");
    A.AddPointInSetConstraints(&prog, x);
    B.AddPointInSetConstraints(&prog, x);
    subspace.AddPointInSetConstraints(&prog, x);
    solvers::MathematicalProgramResult result = solvers::Solve(prog);
    return result.is_success();
  }
}

GCSTrajectoryOptimization::GCSTrajectoryOptimization(
    const GCSTrajectoryOptimizationOptions& options)
    : options_(options) {
  // Make time scaling set.
  time_scaling_set_ =
      HPolyhedron::MakeBox(options_.d_min * Eigen::VectorXd::Ones(1),
                           options_.d_max * Eigen::VectorXd::Ones(1));
}

void GCSTrajectoryOptimization::AddTimeCost(double weight) {
  // Add time cost to each subgraph.
  for (auto& subgraph : subgraphs_) {
    subgraph->AddTimeCost(weight);
  }
  global_time_costs_.push_back(weight);
}

void GCSTrajectoryOptimization::AddPathLengthCost(
    const Eigen::MatrixXd& weight_matrix) {
  // Add path length cost to each subgraph.
  for (auto& subgraph : subgraphs_) {
    if (subgraph->order() > 0) {
      subgraph->AddPathLengthCost(weight_matrix);
    }
  }
  global_path_length_costs_.push_back(weight_matrix);
}

void GCSTrajectoryOptimization::AddPathEnergyCost(
    const Eigen::MatrixXd& weight_matrix) {
  for (auto& subgraph : subgraphs_) {
    if (subgraph->order() > 0) {
      subgraph->AddPathEnergyCost(weight_matrix);
    }
  }
  global_path_energy_costs_.push_back(weight_matrix);
}

void GCSTrajectoryOptimization::AddVelocityBounds(
    const Eigen::Ref<const Eigen::VectorXd>& lb,
    const Eigen::Ref<const Eigen::VectorXd>& ub) {
  for (auto& subgraph : subgraphs_) {
    if (subgraph->order() > 0) {
      subgraph->AddVelocityBounds(lb, ub);
    }
  }
  global_velocity_bounds_.push_back({lb, ub});
}

BsplineTrajectory<double> GCSTrajectoryOptimization::SolvePath(
    Subgraph& source, Subgraph& target,
    const GraphOfConvexSetsOptions& options) {
  // Check if the source and target subgraphs have exactly one region.
  // TODO(wrangelvid) allow multiple regions.
  if (source.size() != 1) {
    throw std::runtime_error("Source subgraph has more than one region.");
  }
  if (target.size() != 1) {
    throw std::runtime_error("Target subgraph has more than one region.");
  }

  VertexId source_id = source.vertices()[0]->id();
  VertexId target_id = target.vertices()[0]->id();
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
  std::vector<double> path_times(max_order_ + 1, 0.0);
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
    for (int i = 0; i < max_order_ + 1; ++i) {
      if (i < edge_path_points.cols()) {
        control_points.push_back(edge_path_points.col(i));
      } else {
        control_points.push_back(
            edge_path_points.col(edge_path_points.cols() - 1));
      }
    }

    if (edge->v().id() == target_id) {
      std::vector<double> next_path_times(max_order_ + 1, path_times.back());
      path_times.insert(path_times.end(), next_path_times.begin(),
                        next_path_times.end());
      break;
    }

    // Extract the duration from the solution.
    double duration = result.GetSolution(edge->xv()).tail<1>().value();
    std::vector<double> next_path_times(max_order_ + 1,
                                        path_times.back() + duration);
    path_times.insert(path_times.end(), next_path_times.begin(),
                      next_path_times.end());
  }
  return BsplineTrajectory<double>(BsplineBasis(max_order_ + 1, path_times),
                                   control_points);
}

}  // namespace trajectory_optimization
}  // namespace planning
}  // namespace drake
