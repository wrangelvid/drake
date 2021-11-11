#include <fstream>
#include <iomanip>
#include <iostream>

#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/scs_solver.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace multibody {
using drake::VectorX;
using drake::multibody::BodyIndex;

ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairsForBox
ConfigurationSpaceCollisionFreeRegion::FindFilteredCollisionPairsForBox(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const Eigen::Ref<const Eigen::VectorXd>& q_lower,
    const Eigen::Ref<const Eigen::VectorXd>& q_upper,
    const FilteredCollisionPairs& existing_filtered_collision_pairs) const {
  FilteredCollisionPairs filtered_collision_pairs;
  // We will form many SOS problems. In each SOS problem, we check if the link
  // obstacle is collision free from the obstacle for bounded box on q.

  // First clap the bounds on q with the joint limits.
  const Eigen::VectorXd q_lower_joint_limit =
      rational_forward_kinematics_.plant().GetPositionLowerLimits();
  const Eigen::VectorXd q_upper_joint_limit =
      rational_forward_kinematics_.plant().GetPositionUpperLimits();
  const int nq = rational_forward_kinematics_.plant().num_positions();
  Eigen::VectorXd q_lower_combined(nq);
  Eigen::VectorXd q_upper_combined(nq);
  for (int i = 0; i < nq; ++i) {
    q_lower_combined(i) = std::max(q_lower(i), q_lower_joint_limit(i));
    q_upper_combined(i) = std::min(q_upper(i), q_upper_joint_limit(i));
  }
  DRAKE_DEMAND((q_lower_combined.array() <= q_star.array()).all());
  DRAKE_DEMAND((q_upper_combined.array() >= q_star.array()).all());
  const Eigen::VectorXd t_upper = rational_forward_kinematics_.ComputeTValue(
      q_upper_combined, q_star, true);
  const Eigen::VectorXd t_lower = rational_forward_kinematics_.ComputeTValue(
      q_lower_combined, q_star, true);
  auto context = rational_forward_kinematics_.plant().CreateDefaultContext();
  rational_forward_kinematics_.plant().SetPositions(context.get(), q_star);

  const BodyIndex world_index =
      rational_forward_kinematics_.plant().world_body().index();
  for (const auto& body_to_polytopes : link_polytopes_) {
    const drake::VectorX<drake::symbolic::Variable> t_on_chain =
        rational_forward_kinematics_.FindTOnPath(body_to_polytopes.first,
                                                 world_index);
    const BodyIndex expressed_body_index = internal::FindBodyInTheMiddleOfChain(
        rational_forward_kinematics_.plant(), world_index,
        body_to_polytopes.first);
    const drake::symbolic::Variables middle_to_link_variables(
        rational_forward_kinematics_.FindTOnPath(expressed_body_index,
                                                 body_to_polytopes.first));
    const drake::symbolic::Variables world_to_middle_variables(
        rational_forward_kinematics_.FindTOnPath(world_index,
                                                 expressed_body_index));
    // Compute the pose of the link (B) and the world (W) in the expressed link
    // (A).
    // TODO(hongkai.dai): save the poses to an unordered set.
    const RationalForwardKinematics::Pose<drake::symbolic::Polynomial> X_AB =
        rational_forward_kinematics_.CalcLinkPoseAsMultilinearPolynomial(
            q_star, body_to_polytopes.first, expressed_body_index);
    const RationalForwardKinematics::Pose<drake::symbolic::Polynomial> X_AW =
        rational_forward_kinematics_.CalcLinkPoseAsMultilinearPolynomial(
            q_star, world_index, expressed_body_index);
    for (const auto& link_polytope : body_to_polytopes.second) {
      for (const auto& obstacle : obstacles_) {
        if (existing_filtered_collision_pairs.count(
                drake::SortedPair<ConvexGeometry::Id>(
                    link_polytope->get_id(), obstacle->get_id())) == 0) {
          const SeparationPlane* separation_plane =
              map_polytopes_to_separation_planes_
                  .find(std::make_pair(link_polytope->get_id(),
                                       obstacle->get_id()))
                  ->second;
          const auto& a_A = separation_plane->a;
          Eigen::Vector3d p_AC;
          rational_forward_kinematics_.plant().CalcPointsPositions(
              *context,
              rational_forward_kinematics_.plant()
                  .get_body(obstacle->body_index())
                  .body_frame(),
              obstacle->p_BC(),
              rational_forward_kinematics_.plant()
                  .get_body(expressed_body_index)
                  .body_frame(),
              &p_AC);
          const std::vector<LinkVertexOnPlaneSideRational>
              positive_side_rationals =
                  GenerateLinkOnOneSideOfPlaneRationalFunction(
                      rational_forward_kinematics_, link_polytope, obstacle,
                      X_AB, a_A, p_AC, PlaneSide::kPositive, a_order_);
          const std::vector<LinkVertexOnPlaneSideRational>
              negative_side_rationals =
                  GenerateLinkOnOneSideOfPlaneRationalFunction(
                      rational_forward_kinematics_, obstacle, link_polytope,
                      X_AW, a_A, p_AC, PlaneSide::kNegative, a_order_);

          VerificationOption option;
          std::vector<LinkVertexOnPlaneSideRational> rationals =
              positive_side_rationals;
          std::copy(negative_side_rationals.begin(),
                    negative_side_rationals.end(),
                    std::back_inserter(rationals));
          auto prog = ConstructProgramToVerifyCollisionFreeBox(
              rationals, t_lower, t_upper, {}, option);

          drake::solvers::MosekSolver mosek_solver;
          drake::solvers::ScsSolver scs_solver;
          prog->SetSolverOption(scs_solver.id(), "verbose", 0);
          const auto result = mosek_solver.Solve(*prog, {}, {});
          if (result.is_success()) {
            filtered_collision_pairs.emplace(link_polytope->get_id(),
                                             obstacle->get_id());
          }
        }
      }
    }
  }
  return FilteredCollisionPairsForBox(q_lower, q_upper,
                                      filtered_collision_pairs);
}

enum class BoxFileState {
  kStart,
  kLower,
  kUpper,
  kPair,
};
std::vector<ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairsForBox>
ReadFilteredCollisionPairsForBoxesFromFile(const std::string& file_name,
                                           int q_size) {
  std::ifstream file;
  file.open(file_name, std::ios::in);
  std::vector<
      ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairsForBox>
      filtered_collision_pairs_for_boxes;
  if (file.is_open()) {
    std::string line;
    BoxFileState box_file_state = BoxFileState::kStart;
    Eigen::VectorXd q_lower(q_size);
    Eigen::VectorXd q_upper(q_size);
    ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairs pairs;
    while (std::getline(file, line)) {
      switch (box_file_state) {
        case BoxFileState::kStart: {
          if (line != "box") {
            throw std::runtime_error(
                "Cannot read filtered collision pairs from this file.");
          }
          pairs.clear();
          box_file_state = BoxFileState::kLower;
          break;
        }
        case BoxFileState::kLower: {
          std::istringstream in_lower(line);
          q_lower.resize(q_size);
          for (int i = 0; i < q_size; ++i) {
            in_lower >> q_lower(i);
          }
          box_file_state = BoxFileState::kUpper;
          break;
        }
        case BoxFileState::kUpper: {
          std::istringstream in_upper(line);
          q_upper.resize(q_size);
          for (int i = 0; i < q_size; ++i) {
            in_upper >> q_upper(i);
          }
          box_file_state = BoxFileState::kPair;
          break;
        }
        case BoxFileState::kPair: {
          if (line != "") {
            int id1, id2;
            std::istringstream id_line(line);
            id_line >> id1 >> id2;
            pairs.emplace(ConvexGeometry::Id(id1), ConvexGeometry::Id(id2));
          } else {
            filtered_collision_pairs_for_boxes.emplace_back(q_lower, q_upper,
                                                            pairs);
            box_file_state = BoxFileState::kStart;
          }
          break;
        }
      }
    }
  }

  file.close();
  return filtered_collision_pairs_for_boxes;
}

void ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairsForBox::
    WriteToFile(const std::string& file_name) const {
  std::ofstream file;
  file.open(file_name, std::ios::app);
  if (file.is_open()) {
    file << "box\n";
    int precision(15);
    file << std::setprecision(precision) << q_lower.transpose() << "\n";
    file << std::setprecision(precision) << q_upper.transpose() << "\n";
    for (const auto& pair : filtered_collision_pairs) {
      file << pair.first() << " " << pair.second() << "\n";
    }
    file << "\n";
  }
  file.close();
}

ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairs
ExtractFilteredCollisionPairsForBox(
    const Eigen::Ref<const Eigen::VectorXd>& q_lower,
    const Eigen::Ref<const Eigen::VectorXd>& q_upper,
    const std::vector<
        ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairsForBox>&
        filtered_collision_pairs_for_boxes) {
  DRAKE_DEMAND((q_lower.array() <= q_upper.array()).all());
  ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairs pairs;
  for (const auto& filtered_collision_pairs_for_box :
       filtered_collision_pairs_for_boxes) {
    if ((q_lower.array() >= filtered_collision_pairs_for_box.q_lower.array())
            .all() &&
        (q_upper.array() <= filtered_collision_pairs_for_box.q_upper.array())
            .all()) {
      pairs.insert(
          filtered_collision_pairs_for_box.filtered_collision_pairs.begin(),
          filtered_collision_pairs_for_box.filtered_collision_pairs.end());
    }
  }
  return pairs;
}
}  // namespace multibody
}  // namespace drake
