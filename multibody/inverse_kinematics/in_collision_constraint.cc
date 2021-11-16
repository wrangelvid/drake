#include "drake/multibody/inverse_kinematics/in_collision_constraint.h"

#include <limits>

#include "drake/multibody/inverse_kinematics/kinematic_constraint_utilities.h"

namespace drake {
namespace multibody {
using internal::RefFromPtrOrThrow;

const double kInf = std::numeric_limits<double>::infinity();

InCollisionConstraint::InCollisionConstraint(
    const multibody::MultibodyPlant<double>* const plant,
    systems::Context<double>* plant_context)
    : solvers::Constraint(1, RefFromPtrOrThrow(plant).num_positions(),
                          Vector1d(-kInf), Vector1d(0)),
      plant_{RefFromPtrOrThrow(plant)},
      plant_context_{plant_context} {
  if (!plant_.geometry_source_is_registered()) {
    throw std::invalid_argument(
        "InCollisionConstrain: MultibodyPlant has not registered its "
        "geometry source with SceneGraph yet. Please refer to "
        "AddMultibodyPlantSceneGraph on how to connect MultibodyPlant to "
        "SceneGraph.");
  }
}
namespace {
void InitializeY(const Eigen::Ref<const Eigen::VectorXd>&, Eigen::VectorXd* y) {
  (*y)(0) = 0;
}

void InitializeY(const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y) {
  (*y) = math::InitializeAutoDiff(Vector1d(0), Eigen::RowVectorXd::Zero(x(0).derivatives().size()));
}

void ComputeDistanceJacobian(
    const MultibodyPlant<double>&, const systems::Context<double>&,
    const geometry::QueryObject<double>&,
    const geometry::SignedDistancePair<double>& signed_distance_pair,
    const Eigen::Ref<const Eigen::VectorXd>&, Eigen::VectorXd* y) {
  (*y)(0) = signed_distance_pair.distance;
}

void ComputeDistanceJacobian(
    const MultibodyPlant<double>& plant,
    const systems::Context<double>& context,
    const geometry::QueryObject<double>& query_object,
    const geometry::SignedDistancePair<double>& signed_distance_pair,
    const Eigen::Ref<const AutoDiffVecXd>& x, AutoDiffVecXd* y) {
  const geometry::SceneGraphInspector<double>& inspector =
      query_object.inspector();
  const geometry::FrameId frame_A_id =
      inspector.GetFrameId(signed_distance_pair.id_A);
  const geometry::FrameId frame_B_id =
      inspector.GetFrameId(signed_distance_pair.id_B);
  const Frame<double>& frameA =
      plant.GetBodyFromFrameId(frame_A_id)->body_frame();
  const Frame<double>& frameB =
      plant.GetBodyFromFrameId(frame_B_id)->body_frame();
  Vector3<double> p_WCa, p_WCb;
  plant.CalcPointsPositions(
      context, frameA,
      inspector.X_FG(signed_distance_pair.id_A) * signed_distance_pair.p_ACa,
      plant.world_frame(), &p_WCa);
  plant.CalcPointsPositions(
      context, frameB,
      inspector.X_FG(signed_distance_pair.id_B) * signed_distance_pair.p_BCb,
      plant.world_frame(), &p_WCb);
  Eigen::Matrix<double, 6, Eigen::Dynamic> Jq_V_BCa_W(6, plant.num_positions());
  plant.CalcJacobianSpatialVelocity(context, JacobianWrtVariable::kQDot, frameA,
                                    signed_distance_pair.p_ACa, frameB,
                                    plant.world_frame(), &Jq_V_BCa_W);
  const Eigen::RowVectorXd ddistance_dq = (p_WCa - p_WCb).transpose() *
                                          Jq_V_BCa_W.bottomRows<3>() /
                                          signed_distance_pair.distance;

  *y = math::initializeAutoDiffGivenGradientMatrix(
      Vector1d(signed_distance_pair.distance),
      ddistance_dq * math::autoDiffToGradientMatrix(x));
}
}  // namespace

template <typename T>
void InCollisionConstraint::DoEvalGeneric(const Eigen::Ref<const VectorX<T>>& x,
                                          VectorX<T>* y) const {
  y->resize(1);
  internal::UpdateContextConfiguration(plant_context_, plant_, x);
  const auto& query_port = plant_.get_geometry_query_input_port();
  if (!query_port.HasValue(*plant_context_)) {
    throw std::invalid_argument(
        "MinimumDistanceConstraint: Cannot get a valid geometry::QueryObject. "
        "Either the plant geometry_query_input_port() is not properly "
        "connected to the SceneGraph's output port, or the plant_context_ is "
        "incorrect. Please refer to AddMultibodyPlantSceneGraph on connecting "
        "MultibodyPlant to SceneGraph.");
  }
  const auto& query_object =
      query_port.Eval<geometry::QueryObject<double>>(*plant_context_);

  const std::vector<geometry::SignedDistancePair<double>>
      signed_distance_pairs =
          query_object.ComputeSignedDistancePairwiseClosestPoints();

  InitializeY(x, y);

  double minimum_distance = kInf;
  const geometry::SignedDistancePair<double>* minimum_distance_pair{nullptr};
  for (const auto& signed_distance_pair : signed_distance_pairs) {
    const double distance = signed_distance_pair.distance;
    if (distance < minimum_distance) {
      minimum_distance = distance;
      minimum_distance_pair = &signed_distance_pair;
    }
  }
  ComputeDistanceJacobian(plant_, *plant_context_, query_object,
                          *minimum_distance_pair, x, y);
}

void InCollisionConstraint::DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
                                   Eigen::VectorXd* y) const {
  DoEvalGeneric(x, y);
}

void InCollisionConstraint::DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
                                   AutoDiffVecXd* y) const {
  DoEvalGeneric(x, y);
}
}  // namespace multibody
}  // namespace drake
