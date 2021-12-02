#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"

#include <algorithm>
#include <limits>

#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"
#include "drake/solvers/csdp_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/scs_solver.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace multibody {
using drake::Matrix3;
using drake::Matrix3X;
using drake::Vector3;
using drake::VectorX;
using drake::multibody::BodyIndex;
using drake::multibody::MultibodyPlant;
using drake::symbolic::RationalFunction;

const double kInf = std::numeric_limits<double>::infinity();

namespace {
struct OrderedKinematicsChain {
  OrderedKinematicsChain(BodyIndex m_end1, BodyIndex m_end2) {
    if (m_end1 < m_end2) {
      end1 = m_end1;
      end2 = m_end2;
    } else {
      end1 = m_end2;
      end2 = m_end1;
    }
  }

  bool operator==(const OrderedKinematicsChain& other) const {
    return end1 == other.end1 && end2 == other.end2;
  }
  BodyIndex end1;
  BodyIndex end2;
};

struct OrderedKinematicsChainHash {
  size_t operator()(const OrderedKinematicsChain& c) const {
    return c.end1 * 100 + c.end2;
  }
};
}  // namespace

ConfigurationSpaceCollisionFreeRegion::ConfigurationSpaceCollisionFreeRegion(
    const MultibodyPlant<double>& plant,
    const std::vector<std::shared_ptr<const ConvexPolytope>>& link_polytopes,
    const std::vector<std::shared_ptr<const ConvexPolytope>>& obstacles,
    SeparatingPlaneOrder a_order)
    : rational_forward_kinematics_(plant),
      obstacles_{obstacles},
      a_order_{a_order} {
  // First group the link polytopes by the attached link.
  for (const auto& link_polytope : link_polytopes) {
    DRAKE_DEMAND(link_polytope->body_index() != plant.world_body().index());
    const auto it = link_polytopes_.find(link_polytope->body_index());
    if (it == link_polytopes_.end()) {
      link_polytopes_.emplace_hint(
          it, std::make_pair(link_polytope->body_index(),
                             std::vector<std::shared_ptr<const ConvexPolytope>>(
                                 {link_polytope})));
    } else {
      it->second.push_back(link_polytope);
    }
  }
  // Now create the separation planes.
  // By default, we only consider the pairs between a link polytope and a world
  // obstacle.
  separation_planes_.reserve(link_polytopes.size() * obstacles.size());
  // Create a map from the pair of obstacle and link to the vector containing
  // all t on the kinematics chain from the obstacle to the link. These t are
  // used for the separating plane normal a, when a is an affine function of t.
  std::unordered_map<OrderedKinematicsChain, VectorX<drake::symbolic::Variable>,
                     OrderedKinematicsChainHash>
      map_link_obstacle_to_t;
  for (const auto& obstacle : obstacles_) {
    DRAKE_DEMAND(obstacle->body_index() == plant.world_body().index());
    for (const auto& link_polytope_pairs : link_polytopes_) {
      for (const auto& link_polytope : link_polytope_pairs.second) {
        Vector3<drake::symbolic::Expression> a;
        VectorX<drake::symbolic::Variable> a_decision_vars;
        if (a_order_ == SeparatingPlaneOrder::kConstant) {
          a_decision_vars.resize(3);
          for (int i = 0; i < 3; ++i) {
            a_decision_vars(i) = drake::symbolic::Variable(
                "a" + std::to_string(separation_planes_.size() * 3 + i));
            a(i) = a_decision_vars(i);
          }
        } else if (a_order_ == SeparatingPlaneOrder::kAffine) {
          // Get t on the kinematics chain.
          const OrderedKinematicsChain link_obstacle(
              link_polytope->body_index(), obstacle->body_index());
          auto it = map_link_obstacle_to_t.find(link_obstacle);
          VectorX<drake::symbolic::Variable> t_on_chain;
          if (it == map_link_obstacle_to_t.end()) {
            t_on_chain = rational_forward_kinematics_.FindTOnPath(
                obstacle->body_index(), link_polytope->body_index());
            map_link_obstacle_to_t.emplace_hint(it, link_obstacle, t_on_chain);
          } else {
            t_on_chain = it->second;
          }
          // Now create the variable A and b, such that a = A*t_on_chain + b
          // A has size 3 * t_on_chain.rows().
          // The first 3 * t_on_chain.rows() in a_decision_vars are for A, the
          // bottom 3 entries are for b.
          a_decision_vars.resize(3 * t_on_chain.rows() + 3);
          Matrix3X<drake::symbolic::Variable> A(3, t_on_chain.rows());
          Vector3<drake::symbolic::Variable> b;
          for (int j = 0; j < t_on_chain.rows(); ++j) {
            for (int i = 0; i < 3; ++i) {
              A(i, j) = drake::symbolic::Variable(
                  "A" + std::to_string(separation_planes_.size()) + "[" +
                  std::to_string(3 * j + i) + "]");
              a_decision_vars(3 * j + i) = A(i, j);
            }
          }
          for (int i = 0; i < 3; ++i) {
            b(i) = drake::symbolic::Variable(
                "b" + std::to_string(separation_planes_.size()) + "[" +
                std::to_string(i) + "]");
            a_decision_vars(3 * t_on_chain.rows() + i) = b(i);
          }
          a = A * t_on_chain + b;
        } else {
          throw std::runtime_error("Unknown order for a.");
        }
        // Expressed body is the middle link in the chain from the world to
        // the link_polytope.
        separation_planes_.emplace_back(
            a, link_polytope, obstacle,
            internal::FindBodyInTheMiddleOfChain(plant, obstacle->body_index(),
                                                 link_polytope->body_index()),
            a_order, a_decision_vars);
        map_polytopes_to_separation_planes_.emplace(
            std::make_pair(link_polytope->get_id(), obstacle->get_id()),
            &(separation_planes_[separation_planes_.size() - 1]));
      }
    }
  }
}

bool ConfigurationSpaceCollisionFreeRegion::IsLinkPairCollisionIgnored(
    ConvexGeometry::Id id1, ConvexGeometry::Id id2,
    const FilteredCollisionPairs& filtered_collision_pairs) const {
  return filtered_collision_pairs.count(
             drake::SortedPair<ConvexGeometry::Id>(id1, id2)) > 0;
}

void ConfigurationSpaceCollisionFreeRegion::ComputeBoundsOnT(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    Eigen::VectorXd* t_lower_limit, Eigen::VectorXd* t_upper_limit) const {
  const int nq = rational_forward_kinematics_.plant().num_positions();
  Eigen::VectorXd q_upper(nq);
  Eigen::VectorXd q_lower(nq);
  for (drake::multibody::JointIndex i{0};
       i < rational_forward_kinematics_.plant().num_joints(); ++i) {
    const auto& joint = rational_forward_kinematics_.plant().get_joint(i);
    q_upper.segment(joint.position_start(), joint.num_positions()) =
        joint.position_upper_limits();
    q_lower.segment(joint.position_start(), joint.num_positions()) =
        joint.position_lower_limits();
  }
  *t_upper_limit =
      rational_forward_kinematics_.ComputeTValue(q_upper, q_star, true);
  *t_lower_limit =
      rational_forward_kinematics_.ComputeTValue(q_lower, q_star, true);
}

std::vector<LinkVertexOnPlaneSideRational>
// Generate t space sos conditions
ConfigurationSpaceCollisionFreeRegion::GenerateLinkOnOneSideOfPlaneRationals(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const FilteredCollisionPairs& filtered_collision_pairs) const {
  auto context = rational_forward_kinematics_.plant().CreateDefaultContext();
  rational_forward_kinematics_.plant().SetPositions(context.get(), q_star);

  const BodyIndex world_index =
      rational_forward_kinematics_.plant().world_body().index();
  std::vector<LinkVertexOnPlaneSideRational> rationals;
  for (const auto& body_to_polytopes : link_polytopes_) {
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
        if (!IsLinkPairCollisionIgnored(link_polytope->get_id(),
                                        obstacle->get_id(),
                                        filtered_collision_pairs)) {
          const auto& a_A = map_polytopes_to_separation_planes_
                                .find(std::make_pair(link_polytope->get_id(),
                                                     obstacle->get_id()))
                                ->second->a;
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
          // I cannot use "insert" function to append vectors, since
          // LinkVertexOnPlaneSideRational contains const members, hence it does
          // not have an assignment operator.
          std::copy(positive_side_rationals.begin(),
                    positive_side_rationals.end(),
                    std::back_inserter(rationals));
          std::copy(negative_side_rationals.begin(),
                    negative_side_rationals.end(),
                    std::back_inserter(rationals));
        }
      }
    }
  }
  return rationals;
}

namespace {
// This struct is only used in ConstructProgramToVerifyCollisionFreeBox.
struct UnorderedKinematicsChain {
  UnorderedKinematicsChain(BodyIndex m_start, BodyIndex m_end)
      : start(m_start), end(m_end) {}

  bool operator==(const UnorderedKinematicsChain& other) const {
    return start == other.start && end == other.end;
  }

  BodyIndex start;
  BodyIndex end;
};

struct UnorderedKinematicsChainHash {
  size_t operator()(const UnorderedKinematicsChain& p) const {
    return p.start * 100 + p.end;
  }
};

void FindMonomialBasisForConstantSeparatingPlane(
    const RationalForwardKinematics& rational_forward_kinematics,
    const LinkVertexOnPlaneSideRational& rational,
    std::unordered_map<OrderedKinematicsChain,
                       std::pair<VectorX<drake::symbolic::Variable>,
                                 VectorX<drake::symbolic::Monomial>>,
                       OrderedKinematicsChainHash>*
        map_ordered_kinematics_chain_to_monomial_basis,
    VectorX<drake::symbolic::Variable>* t_chain,
    VectorX<drake::symbolic::Monomial>* monomial_basis_chain) {
  DRAKE_DEMAND(rational.a_order == SeparatingPlaneOrder::kConstant);
  // First check if the monomial basis for this kinematics chain has been
  // computed.
  const OrderedKinematicsChain ordered_kinematics_chain(
      rational.link_polytope->body_index(), rational.expressed_body_index);
  const auto it = map_ordered_kinematics_chain_to_monomial_basis->find(
      ordered_kinematics_chain);
  if (it == map_ordered_kinematics_chain_to_monomial_basis->end()) {
    *t_chain = rational_forward_kinematics.FindTOnPath(
        rational.link_polytope->body_index(), rational.expressed_body_index);
    *monomial_basis_chain = GenerateMonomialBasisWithOrderUpToOne(
        drake::symbolic::Variables(*t_chain));
    map_ordered_kinematics_chain_to_monomial_basis->emplace_hint(
        it, std::make_pair(ordered_kinematics_chain,
                           std::make_pair(*t_chain, *monomial_basis_chain)));
  } else {
    *t_chain = it->second.first;
    *monomial_basis_chain = it->second.second;
  }
}

/**
 * The major difference from FindMonomialBasisForConstantSeparatingPlane is that
 * t_chain includes all t from this link polytope to the other link polytope,
 * while FindMonomialBasisForConstaintSeparatingPlane's t_chain only include
 * t from this link polytope to the expressed body.
 */
void FindMonomialBasisForAffineSeparatingPlane(
    const RationalForwardKinematics& rational_forward_kinematics,
    const LinkVertexOnPlaneSideRational& rational,
    std::unordered_map<UnorderedKinematicsChain,
                       std::pair<VectorX<drake::symbolic::Variable>,
                                 VectorX<drake::symbolic::Monomial>>,
                       UnorderedKinematicsChainHash>*
        map_unordered_kinematics_chain_to_monomial_basis,
    VectorX<drake::symbolic::Variable>* t_chain,
    VectorX<drake::symbolic::Monomial>* monomial_basis_chain) {
  DRAKE_DEMAND(rational.a_order == SeparatingPlaneOrder::kAffine);
  // First check if the monomial basis for this kinematics chain has been
  // computed.
  const UnorderedKinematicsChain unordered_kinematics_chain(
      rational.link_polytope->body_index(),
      rational.other_side_link_polytope->body_index());
  const auto it = map_unordered_kinematics_chain_to_monomial_basis->find(
      unordered_kinematics_chain);
  if (it == map_unordered_kinematics_chain_to_monomial_basis->end()) {
    *t_chain = rational_forward_kinematics.FindTOnPath(
        rational.link_polytope->body_index(),
        rational.other_side_link_polytope->body_index());
    const VectorX<drake::symbolic::Variable> t_monomial =
        rational_forward_kinematics.FindTOnPath(
            rational.link_polytope->body_index(),
            rational.expressed_body_index);
    *monomial_basis_chain = GenerateMonomialBasisWithOrderUpToOne(
        drake::symbolic::Variables(t_monomial));
    map_unordered_kinematics_chain_to_monomial_basis->emplace_hint(
        it, std::make_pair(unordered_kinematics_chain,
                           std::make_pair(*t_chain, *monomial_basis_chain)));
  } else {
    *t_chain = it->second.first;
    *monomial_basis_chain = it->second.second;
  }
}
}  // namespace

std::unique_ptr<drake::solvers::MathematicalProgram>
// Actually constructs the programs
ConfigurationSpaceCollisionFreeRegion::ConstructProgramToVerifyCollisionFreeBox(
    const std::vector<LinkVertexOnPlaneSideRational>& rationals,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper,
    const FilteredCollisionPairs& filtered_collision_pairs,
    const VerificationOption& verification_option) const {
  auto prog = std::make_unique<drake::solvers::MathematicalProgram>();
  // Add t as indeterminates
  const auto& t = rational_forward_kinematics_.t();
  prog->AddIndeterminates(t);
  // Add separation planes as decision variables
  for (const auto& separation_plane : separation_planes_) {
    if (!IsLinkPairCollisionIgnored(
            separation_plane.positive_side_polytope->get_id(),
            separation_plane.negative_side_polytope->get_id(),
            filtered_collision_pairs)) {
      prog->AddDecisionVariables(separation_plane.decision_variables);
    }
  }

  // Now build the polynomials t - t_lower and t_upper - t
  DRAKE_DEMAND(t_lower.size() == t_upper.size());
  // maps t(i) to (t(i) - t_lower(i), t_upper(i) - t(i))
  std::unordered_map<
      drake::symbolic::Variable::Id,
      std::pair<drake::symbolic::Polynomial, drake::symbolic::Polynomial>>
      map_t_to_box_bounds;
  map_t_to_box_bounds.reserve(t.size());
  const drake::symbolic::Monomial monomial_one{};
  for (int i = 0; i < t.size(); ++i) {
    map_t_to_box_bounds.emplace(
        rational_forward_kinematics_.t()(i).get_id(),
        std::make_pair(
            drake::symbolic::Polynomial({{drake::symbolic::Monomial(t(i)), 1},
                                         {monomial_one, -t_lower(i)}}),
            drake::symbolic::Polynomial({{drake::symbolic::Monomial(t(i)), -1},
                                         {monomial_one, t_upper(i)}})));
  }

  // map the kinematics chain to (t_chain, monomial_basis), where t_chain are
  // t on the kinematics chain.
  //
  // Case 1. when we use constant separating plane
  // (SeparationPlaneOrder::kConstraint), we consider the chain from the
  // expressed body to the link (or the obstacle). In this case, t - t_lower and
  // t_upper - t contains the t on this chain, also the monomial basis are
  // generated from t on this chain.
  //
  // Case 2. when we use affine separating plane (a = A * t + b). In this case
  // t - t_lower and t_upper - t contains the t on the chain from the link to
  // the obstacle, while the monomial basis are generated from t on the "half
  // chain" from the expressed body to the link (or the obstacle).

  // This is for Constant separating plane.
  std::unordered_map<OrderedKinematicsChain,
                     std::pair<VectorX<drake::symbolic::Variable>,
                               VectorX<drake::symbolic::Monomial>>,
                     OrderedKinematicsChainHash>
      map_ordered_kinematics_chain_to_monomial_basis;

  // This is for affine separating plane.
  std::unordered_map<UnorderedKinematicsChain,
                     std::pair<VectorX<drake::symbolic::Variable>,
                               VectorX<drake::symbolic::Monomial>>,
                     UnorderedKinematicsChainHash>
      map_unordered_kinematics_chain_to_monomial_basis;

  for (const auto& rational : rationals) {
    VectorX<drake::symbolic::Variable> t_chain;
    VectorX<drake::symbolic::Monomial> monomial_basis_chain;
    if (a_order_ == SeparatingPlaneOrder::kConstant) {
      FindMonomialBasisForConstantSeparatingPlane(
          rational_forward_kinematics_, rational,
          &map_ordered_kinematics_chain_to_monomial_basis, &t_chain,
          &monomial_basis_chain);
    } else if (a_order_ == SeparatingPlaneOrder::kAffine) {
      FindMonomialBasisForAffineSeparatingPlane(
          rational_forward_kinematics_, rational,
          &map_unordered_kinematics_chain_to_monomial_basis, &t_chain,
          &monomial_basis_chain);
    }
    VectorX<drake::symbolic::Polynomial> t_minus_t_lower(t_chain.size());
    VectorX<drake::symbolic::Polynomial> t_upper_minus_t(t_chain.size());
    for (int i = 0; i < t_chain.size(); ++i) {
      auto it_t = map_t_to_box_bounds.find(t_chain(i).get_id());
      t_minus_t_lower(i) = it_t->second.first;
      t_upper_minus_t(i) = it_t->second.second;
    }
    // Now add the constraint that t_lower <= t <= t_upper implies the rational
    // being nonnegative.
    AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
        prog.get(), rational.rational, t_minus_t_lower, t_upper_minus_t,
        monomial_basis_chain, verification_option);
  }

  return prog;
}

double ConfigurationSpaceCollisionFreeRegion::FindLargestBoxThroughBinarySearch(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const FilteredCollisionPairs& filtered_collision_pairs,
    const Eigen::Ref<const Eigen::VectorXd>& negative_delta_q,
    const Eigen::Ref<const Eigen::VectorXd>& positive_delta_q,
    double rho_lower_initial, double rho_upper_initial,
    const std::vector<FilteredCollisionPairsForBox>&
        filtered_collision_pairs_for_boxes,
    const BinarySearchOption& option) const {
  std::vector<BoxVerificationTuple> box_verification_tuples;
  VectorX<drake::symbolic::Variable> t_lower_vars, t_upper_vars;
  GenerateVerificationConstraintForBilinearAlternation(
      q_star, filtered_collision_pairs, &box_verification_tuples, &t_lower_vars,
      &t_upper_vars);
  const int nq = rational_forward_kinematics_.plant().num_positions();
  DRAKE_DEMAND(q_star.rows() == nq);

  // Compute t_lower and t_upper from the joint limits.
  Eigen::VectorXd t_lower_limit, t_upper_limit;
  ComputeBoundsOnT(q_star, &t_lower_limit, &t_upper_limit);

  return this->FindLargestBoxThroughBinarySearch(
      box_verification_tuples, negative_delta_q, positive_delta_q,
      t_lower_limit, t_upper_limit, q_star, rho_lower_initial,
      rho_upper_initial, t_lower_vars, t_upper_vars,
      filtered_collision_pairs_for_boxes, option);
}

template <typename T>
drake::symbolic::Polynomial ComputePolynomialFromGramian(
    const VectorX<drake::symbolic::Monomial>& monomial_basis,
    const drake::MatrixX<T>& gramian) {
  const int gramian_rows = monomial_basis.rows();
  DRAKE_DEMAND(gramian.rows() == gramian_rows);
  DRAKE_DEMAND(gramian.cols() == gramian_rows);
  drake::symbolic::Polynomial poly{};
  for (int j = 0; j < gramian_rows; ++j) {
    poly.AddProduct(gramian(j, j), pow(monomial_basis(j), 2));
    for (int i = j + 1; i < gramian_rows; ++i) {
      poly.AddProduct(2 * gramian(i, j), monomial_basis(i) * monomial_basis(j));
    }
  }
  return poly;
}

drake::solvers::MathematicalProgramResult
ConfigurationSpaceCollisionFreeRegion::FindLargestBoxThroughBilinearAlternation(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const FilteredCollisionPairs& filtered_collision_pairs,
    const BilinearAlternationOption& options,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_init,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_init,
    Eigen::VectorXd* t_lower_sol, Eigen::VectorXd* t_upper_sol,
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>* q_box_sols)
    const {
  std::vector<BoxVerificationTuple> bilinear_alternation_tuples;
  VectorX<drake::symbolic::Variable> t_lower_vars, t_upper_vars;
  GenerateVerificationConstraintForBilinearAlternation(
      q_star, filtered_collision_pairs, &bilinear_alternation_tuples,
      &t_lower_vars, &t_upper_vars);

  // Compute t_lower and t_upper from the joint limits.
  Eigen::VectorXd t_lower_joint_limit, t_upper_joint_limit;
  ComputeBoundsOnT(q_star, &t_lower_joint_limit, &t_upper_joint_limit);

  return FindLargestBoxThroughBilinearAlternation(
      q_star, bilinear_alternation_tuples, options, t_lower_init, t_upper_init,
      t_lower_joint_limit, t_upper_joint_limit, t_lower_vars, t_upper_vars,
      t_lower_sol, t_upper_sol, q_box_sols);
}

std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> RemoveIncludedBox(
    const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>&
        q_box_sols) {
  std::vector<bool> remaining_boxes_flag(q_box_sols.size(), true);
  for (int i = 0; i < static_cast<int>(q_box_sols.size()); ++i) {
    for (int j = i + 1; j < static_cast<int>(q_box_sols.size()); ++j) {
      if ((q_box_sols[i].first.array() >= q_box_sols[j].first.array()).all() &&
          (q_box_sols[i].second.array() <= q_box_sols[j].second.array())
              .all()) {
        remaining_boxes_flag[i] = false;
      }
      if ((q_box_sols[i].first.array() <= q_box_sols[j].first.array()).all() &&
          (q_box_sols[i].second.array() >= q_box_sols[j].second.array())
              .all()) {
        remaining_boxes_flag[j] = false;
      }
    }
  }
  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> q_remaining_boxes;
  q_remaining_boxes.reserve(q_box_sols.size());
  for (int i = 0; i < static_cast<int>(q_box_sols.size()); ++i) {
    if (remaining_boxes_flag[i]) {
      q_remaining_boxes.push_back(q_box_sols[i]);
    }
  }
  return q_remaining_boxes;
}

bool ConfigurationSpaceCollisionFreeRegion::FindLargestBox(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const FilteredCollisionPairsForBox& filtered_collision_pairs_base,
    const std::vector<FilteredCollisionPairsForBox>&
        filtered_collision_pairs_for_boxes,
    const Eigen::Ref<const Eigen::VectorXd>& negative_delta_q_init,
    const Eigen::Ref<const Eigen::VectorXd>& positive_delta_q_init,
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>* q_box_sols,
    const FindLargestBoxOption& option) const {
  DRAKE_DEMAND(option.grow_all_dimension == false);
  DRAKE_DEMAND((negative_delta_q_init.array() <= 0).all());
  DRAKE_DEMAND((positive_delta_q_init.array() >= 0).all());
  DRAKE_DEMAND(
      (positive_delta_q_init.array() > negative_delta_q_init.array()).all());
  // Compute the verification tuples.
  std::vector<BoxVerificationTuple> bilinear_alternation_tuples;
  VectorX<drake::symbolic::Variable> t_lower_vars, t_upper_vars;
  GenerateVerificationConstraintForBilinearAlternation(
      q_star, filtered_collision_pairs_base.filtered_collision_pairs,
      &bilinear_alternation_tuples, &t_lower_vars, &t_upper_vars);

  // Compute t_lower and t_upper from the joint limits.
  Eigen::VectorXd t_lower_joint_limit, t_upper_joint_limit;
  ComputeBoundsOnT(q_star, &t_lower_joint_limit, &t_upper_joint_limit);

  double rho_init = SequentiallyDoubleBoxSize(
      bilinear_alternation_tuples, negative_delta_q_init, positive_delta_q_init,
      t_lower_joint_limit, t_upper_joint_limit, q_star, t_lower_vars,
      t_upper_vars, filtered_collision_pairs_for_boxes,
      option.binary_search_option());
  if (rho_init > 1) {
    Eigen::VectorXd q_lower_init, q_upper_init;
    CalcConfigurationBoundsFromRho(negative_delta_q_init, positive_delta_q_init,
                                   rho_init, q_star, &q_lower_init,
                                   &q_upper_init);
    q_box_sols->emplace_back(q_lower_init, q_upper_init);
  }

  //// Now do a binary search between rho_init and rho_init * 2.
  // const double rho_binary1 = FindLargestBoxThroughBinarySearch(
  //    bilinear_alternation_tuples, negative_delta_q_init,
  //    positive_delta_q_init, t_lower_joint_limit, t_upper_joint_limit, q_star,
  //    rho_init, rho_init * 2, t_lower_vars, t_upper_vars,
  //    option.binary_search_option());

  // const double rho_init = 1;
  const Eigen::VectorXd t_lower_bilinear_init =
      rational_forward_kinematics_.ComputeTValue(
          q_star + rho_init * negative_delta_q_init, q_star, true);
  const Eigen::VectorXd t_upper_bilinear_init =
      rational_forward_kinematics_.ComputeTValue(
          q_star + rho_init * positive_delta_q_init, q_star, true);
  std::cout << "t_lower_bilinear_init: " << t_lower_bilinear_init.transpose()
            << "\n";
  std::cout << "t_upper_bilinear_init: " << t_upper_bilinear_init.transpose()
            << "\n";

  // Find the shape of the box through bilinear alternation.
  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> q_box_sols_bilinear;
  Eigen::VectorXd t_lower_sol_bilinear, t_upper_sol_bilinear;
  auto result = FindLargestBoxThroughBilinearAlternation(
      q_star, bilinear_alternation_tuples, option, t_lower_bilinear_init,
      t_upper_bilinear_init, t_lower_joint_limit, t_upper_joint_limit,
      t_lower_vars, t_upper_vars, &t_lower_sol_bilinear, &t_upper_sol_bilinear,
      &q_box_sols_bilinear);
  q_box_sols->insert(q_box_sols->end(), q_box_sols_bilinear.begin(),
                     q_box_sols_bilinear.end());
  if (!result.is_success()) {
    if (rho_init > 1) {
      // Bilinear alternation fails, but we can still use the result from the
      // previous step, when we sequentially enlarge the box by a factor of 2.
      return true;
    } else {
      return false;
    }
  }

  const Eigen::VectorXd positive_delta_q =
      (t_upper_sol_bilinear.array().atan() * 2).matrix();
  const Eigen::VectorXd negative_delta_q =
      (t_lower_sol_bilinear.array().atan() * 2).matrix();
  std::cout << "positive_delta_q (deg): "
            << positive_delta_q.transpose() / M_PI * 180 << "\n";
  std::cout << "negative_delta_q (deg): "
            << negative_delta_q.transpose() / M_PI * 180 << "\n";

  // Find the scaling factor of the box through binary search.
  // First find an upper bound that is infeasible.
  const double rho_lower_initial = SequentiallyDoubleBoxSize(
      bilinear_alternation_tuples, negative_delta_q, positive_delta_q,
      t_lower_joint_limit, t_upper_joint_limit, q_star, t_lower_vars,
      t_upper_vars, filtered_collision_pairs_for_boxes,
      option.binary_search_option());
  double rho_upper_initial = rho_lower_initial * 2;
  std::cout << "rho_upper_initial: " << rho_upper_initial << "\n";
  // Now do a binary search between rho_lower_initial and rho_upper_initial.
  const double rho_sol = FindLargestBoxThroughBinarySearch(
      bilinear_alternation_tuples, negative_delta_q, positive_delta_q,
      t_lower_joint_limit, t_upper_joint_limit, q_star, rho_lower_initial,
      rho_upper_initial, t_lower_vars, t_upper_vars,
      filtered_collision_pairs_for_boxes, option.binary_search_option());
  Eigen::VectorXd q_lower_sol, q_upper_sol;
  CalcConfigurationBoundsFromRho(negative_delta_q, positive_delta_q, rho_sol,
                                 q_star, &q_lower_sol, &q_upper_sol);
  q_box_sols->emplace_back(q_lower_sol, q_upper_sol);
  std::cout << "q_lower_sol: " << q_lower_sol.transpose() << "\n";
  std::cout << "q_upper_sol: " << q_upper_sol.transpose() << "\n";
  const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>
      q_remaining_boxes = RemoveIncludedBox(*q_box_sols);
  *q_box_sols = q_remaining_boxes;
  return true;
}

bool ConfigurationSpaceCollisionFreeRegion::IsPostureCollisionFree(
    const drake::systems::Context<double>& context) const {
  const auto& plant = rational_forward_kinematics_.plant();
  drake::math::RigidTransform<double> X_WW;
  X_WW.SetIdentity();
  for (const auto& link_and_polytopes : link_polytopes_) {
    const drake::math::RigidTransform<double> X_WB = plant.EvalBodyPoseInWorld(
        context, plant.get_body(link_and_polytopes.first));
    for (const auto& link_polytope : link_and_polytopes.second) {
      for (const auto& obstacle : obstacles_) {
        if (link_polytope->IsInCollision(*obstacle, X_WB, X_WW)) {
          return true;
        }
      }
    }
  }
  return false;
}

void ConfigurationSpaceCollisionFreeRegion::CalcConfigurationBoundsFromRho(
    const Eigen::Ref<const Eigen::VectorXd>& negative_delta_q,
    const Eigen::Ref<const Eigen::VectorXd>& positive_delta_q, double rho,
    const Eigen::Ref<const Eigen::VectorXd>& q_star, Eigen::VectorXd* q_lower,
    Eigen::VectorXd* q_upper) const {
  Eigen::VectorXd delta_q_plus = rho * positive_delta_q;
  Eigen::VectorXd delta_q_minus = rho * negative_delta_q;
  const int nq = rational_forward_kinematics_.plant().num_positions();
  q_lower->resize(nq);
  q_upper->resize(nq);
  const Eigen::VectorXd q_lower_joint_limit =
      rational_forward_kinematics_.plant().GetPositionLowerLimits();
  const Eigen::VectorXd q_upper_joint_limit =
      rational_forward_kinematics_.plant().GetPositionUpperLimits();
  for (int i = 0; i < nq; ++i) {
    // First clamp delta_q_plus and delta_q_minus to within [-pi, pi].
    // Note this only works for robots with revolute joints. If the robot has
    // prismatic joints, this method is not applicable.
    delta_q_plus(i) = std::min(M_PI, delta_q_plus(i));
    delta_q_minus(i) = std::max(-M_PI, delta_q_minus(i));
    // now clamp q_upper and q_lower to be within joint limits.
    (*q_lower)(i) =
        std::max(q_star(i) + delta_q_minus(i), q_lower_joint_limit(i));
    (*q_upper)(i) =
        std::min(q_star(i) + delta_q_plus(i), q_upper_joint_limit(i));
  }
}

// The box is t_lower <= t <= t_upper
// Hence we need the following polynomials
// t(i) - t_lower(i) >= 0
// t_upper(i) - t(i) >= 0
// @param[out] box_lower_bound_polynomials t(i) - t_lower(i)
// @param[out] box_upper_bound_polynomials t_upper(i) - t(i)
template <typename T>
void ConstructBoxBoundPolynomials(
    const drake::VectorX<drake::symbolic::Variable>& t,
    const VectorX<T>& t_lower, const VectorX<T>& t_upper,
    std::vector<drake::symbolic::Polynomial>* box_lower_bound_polynomials,
    std::vector<drake::symbolic::Polynomial>* box_upper_bound_polynomials) {
  const int t_size = t.size();
  DRAKE_DEMAND(t_lower.rows() == t_size);
  DRAKE_DEMAND(t_upper.rows() == t_size);
  box_lower_bound_polynomials->clear();
  box_lower_bound_polynomials->reserve(t_size);
  box_upper_bound_polynomials->clear();
  box_upper_bound_polynomials->reserve(t_size);
  const drake::symbolic::Monomial monomial_one{};
  for (int i = 0; i < t_size; ++i) {
    const drake::symbolic::Monomial ti_monomial(t(i));
    box_lower_bound_polynomials->emplace_back(
        drake::symbolic::Polynomial::MapType{{ti_monomial, 1},
                                             {monomial_one, -t_lower(i)}});
    box_upper_bound_polynomials->emplace_back(
        drake::symbolic::Polynomial::MapType{{ti_monomial, -1},
                                             {monomial_one, t_upper(i)}});
  }
}

void ConfigurationSpaceCollisionFreeRegion::
    GenerateVerificationConstraintForBilinearAlternation(
        const Eigen::Ref<const Eigen::VectorXd>& q_star,
        const FilteredCollisionPairs& filtered_collision_pairs,
        std::vector<BoxVerificationTuple>* bilinear_alternation_tuples,
        VectorX<drake::symbolic::Variable>* t_lower,
        VectorX<drake::symbolic::Variable>* t_upper) const {
  // First declare the variables c and d
  const int num_t = rational_forward_kinematics_.t().size();
  t_lower->resize(num_t);
  t_upper->resize(num_t);
  for (int i = 0; i < num_t; ++i) {
    (*t_lower)(i) =
        drake::symbolic::Variable("t_lower(" + std::to_string(i) + ")");
    (*t_upper)(i) =
        drake::symbolic::Variable("t_upper(" + std::to_string(i) + ")");
  }
  // Now build the box as t_lower <= t <= t_upper
  std::vector<drake::symbolic::Polynomial> box_lower_bound_polynomials,
      box_upper_bound_polynomials;
  ConstructBoxBoundPolynomials(rational_forward_kinematics_.t(), *t_lower,
                               *t_upper, &box_lower_bound_polynomials,
                               &box_upper_bound_polynomials);
  // Loop through each rational, and get the BoxVerificationTuple.
  const std::vector<LinkVertexOnPlaneSideRational> rationals =
      GenerateLinkOnOneSideOfPlaneRationals(q_star, filtered_collision_pairs);
  bilinear_alternation_tuples->resize(rationals.size());
  // This is for Constant separating plane.
  std::unordered_map<OrderedKinematicsChain,
                     std::pair<VectorX<drake::symbolic::Variable>,
                               VectorX<drake::symbolic::Monomial>>,
                     OrderedKinematicsChainHash>
      map_ordered_kinematics_chain_to_monomial_basis;

  // This is for affine separating plane.
  std::unordered_map<UnorderedKinematicsChain,
                     std::pair<VectorX<drake::symbolic::Variable>,
                               VectorX<drake::symbolic::Monomial>>,
                     UnorderedKinematicsChainHash>
      map_unordered_kinematics_chain_to_monomial_basis;
  for (int i = 0; i < static_cast<int>(rationals.size()); ++i) {
    const auto& rational = rationals[i];
    bilinear_alternation_tuples->at(i).p = rational.rational.numerator();
    bilinear_alternation_tuples->at(i).geometry_pair =
        drake::SortedPair<ConvexGeometry::Id>(
            rational.link_polytope->get_id(),
            rational.other_side_link_polytope->get_id());
    // First obtain the t used on the kinematics chain, and the monomial basis
    // for the lagrangian multiplier.
    if (a_order_ == SeparatingPlaneOrder::kConstant) {
      FindMonomialBasisForConstantSeparatingPlane(
          rational_forward_kinematics_, rational,
          &map_ordered_kinematics_chain_to_monomial_basis,
          &(bilinear_alternation_tuples->at(i).t_chain),
          &(bilinear_alternation_tuples->at(i).lagrangian_monomial_basis));
    } else if (a_order_ == SeparatingPlaneOrder::kAffine) {
      FindMonomialBasisForAffineSeparatingPlane(
          rational_forward_kinematics_, rational,
          &map_unordered_kinematics_chain_to_monomial_basis,
          &(bilinear_alternation_tuples->at(i).t_chain),
          &(bilinear_alternation_tuples->at(i).lagrangian_monomial_basis));
    }
    // Each of t in t_chain will correspond to two constraints, one for box
    // lower bound and one for upper bound.
    const int t_chain_size = bilinear_alternation_tuples->at(i).t_chain.rows();
    bilinear_alternation_tuples->at(i).constraints.reserve(2 * t_chain_size);
    bilinear_alternation_tuples->at(i).lagrangians.reserve(2 * t_chain_size);
    bilinear_alternation_tuples->at(i).lagrangian_gramians.reserve(
        2 * t_chain_size);
    bilinear_alternation_tuples->at(i).t_bound_vars.reserve(2 * t_chain_size);
    const auto& t_chain = bilinear_alternation_tuples->at(i).t_chain;
    const int gramian_rows =
        bilinear_alternation_tuples->at(i).lagrangian_monomial_basis.rows();
    for (int j = 0; j < t_chain_size; ++j) {
      // For each t in t_chain, add the constraint, the lagrangian corresponding
      // to the lower and upper bound of the box on t.
      const int t_index =
          rational_forward_kinematics_.t_id_to_index().at(t_chain(j).get_id());
      bilinear_alternation_tuples->at(i).constraints.push_back(
          box_lower_bound_polynomials[t_index]);
      bilinear_alternation_tuples->at(i).constraints.push_back(
          box_upper_bound_polynomials[t_index]);
      bilinear_alternation_tuples->at(i).t_bound_vars.push_back(
          (*t_lower)(t_index));
      bilinear_alternation_tuples->at(i).t_bound_vars.push_back(
          (*t_upper)(t_index));
      // Now create symmetric Gramian matrices for Lagrangians.
      drake::solvers::MatrixXDecisionVariable lagrangian_gramian_lower(
          gramian_rows, gramian_rows);
      drake::solvers::MatrixXDecisionVariable lagrangian_gramian_upper(
          gramian_rows, gramian_rows);
      for (int n = 0; n < gramian_rows; ++n) {
        for (int m = n; m < gramian_rows; ++m) {
          const std::string name =
              "lagrangian(" + std::to_string(m) + "," + std::to_string(n) + ")";
          lagrangian_gramian_lower(m, n) = drake::symbolic::Variable(name);
          lagrangian_gramian_upper(m, n) = drake::symbolic::Variable(name);
        }
        for (int m = 0; m < n; ++m) {
          lagrangian_gramian_lower(m, n) = lagrangian_gramian_lower(n, m);
          lagrangian_gramian_upper(m, n) = lagrangian_gramian_upper(n, m);
        }
      }
      bilinear_alternation_tuples->at(i).lagrangian_gramians.push_back(
          lagrangian_gramian_lower);
      bilinear_alternation_tuples->at(i).lagrangian_gramians.push_back(
          lagrangian_gramian_upper);
      // Now compute lagrangian_monomial_basis' * lagrangian_gramian *
      // lagrangian_monomial_basis, and add this product as the new lagrangian
      // to lagrangians.
      auto add_lagrangian =
          [bilinear_alternation_tuples, i](
              const VectorX<drake::symbolic::Monomial>& monomial_basis,
              const drake::solvers::MatrixXDecisionVariable& gramian) {
            bilinear_alternation_tuples->at(i).lagrangians.push_back(
                ComputePolynomialFromGramian(monomial_basis, gramian));
          };
      add_lagrangian(
          bilinear_alternation_tuples->at(i).lagrangian_monomial_basis,
          lagrangian_gramian_lower);
      add_lagrangian(
          bilinear_alternation_tuples->at(i).lagrangian_monomial_basis,
          lagrangian_gramian_upper);
    }
  }
}

void ConfigurationSpaceCollisionFreeRegion::
    ConstrainVerifiedPolynomialsNonnegative(
        const std::vector<int>& verified_polynomials_expected_gram_rows,
        const std::vector<BoxVerificationTuple>& bilinear_alternation_tuples,
        const std::vector<drake::symbolic::Polynomial>& verified_polynomials,
        drake::solvers::MathematicalProgram::NonnegativePolynomial
            nonnegative_polynomial_type,
        drake::solvers::MathematicalProgram* prog) const {
  // Now assign the memory for all the Grammian variables.
  const int gram_variables_size =
      std::accumulate(verified_polynomials_expected_gram_rows.begin(),
                      verified_polynomials_expected_gram_rows.end(), 0,
                      [](int x, int y) { return x + (y + 1) * y / 2; });
  // Now create empty matrices to store Grammian matrix of different size.
  std::unordered_map<int, drake::MatrixX<drake::symbolic::Variable>>
      verified_polynomial_expected_grams;
  for (int verified_polynomials_expected_gram_row :
       verified_polynomials_expected_gram_rows) {
    if (verified_polynomial_expected_grams.count(
            verified_polynomials_expected_gram_row) == 0) {
      verified_polynomial_expected_grams.emplace_hint(
          verified_polynomial_expected_grams.end(),
          verified_polynomials_expected_gram_row,
          drake::MatrixX<drake::symbolic::Variable>(
              verified_polynomials_expected_gram_row,
              verified_polynomials_expected_gram_row));
    }
  }
  // Now create all the variables in the gram matrices.
  auto verified_polynomials_expected_gram_vars =
      prog->NewContinuousVariables(gram_variables_size);
  // Now compute verified_polynomials_expected
  int gram_variables_count = 0;
  for (int k = 0; k < static_cast<int>(bilinear_alternation_tuples.size());
       ++k) {
    auto& verified_polynomial_gram = verified_polynomial_expected_grams
        [verified_polynomials_expected_gram_rows[k]];
    for (int j = 0; j < verified_polynomial_gram.rows(); ++j) {
      for (int i = j; i < verified_polynomial_gram.rows(); ++i) {
        verified_polynomial_gram(i, j) =
            verified_polynomials_expected_gram_vars(gram_variables_count++);
        if (i != j) {
          verified_polynomial_gram(j, i) = verified_polynomial_gram(i, j);
        }
      }
    }
    const drake::symbolic::Polynomial verified_polynomial_expected =
        prog->NewNonnegativePolynomial(
            verified_polynomial_gram,
            bilinear_alternation_tuples[k].lagrangian_monomial_basis,
            nonnegative_polynomial_type);
    prog->AddEqualityConstraintBetweenPolynomials(verified_polynomials[k],
                                                  verified_polynomial_expected);
  }
  DRAKE_DEMAND(gram_variables_count == gram_variables_size);
}

drake::solvers::MathematicalProgramResult
ConfigurationSpaceCollisionFreeRegion::SearchLagrangianMultiplier(
    const std::vector<BoxVerificationTuple>& bilinear_alternation_tuples,
    const drake::VectorX<drake::symbolic::Variable>& t_lower_vars,
    const drake::VectorX<drake::symbolic::Variable>& t_upper_vars,
    const Eigen::VectorXd& t_lower_sol, const Eigen::VectorXd& t_upper_sol,
    const SearchLagrangianOption& option) const {
  DRAKE_DEMAND(t_lower_vars.rows() == t_lower_sol.rows());
  DRAKE_DEMAND(t_upper_vars.rows() == t_upper_sol.rows());
  drake::solvers::MathematicalProgram prog;
  // Add the separation hyperplane parameters as decision variables.
  for (const auto& separation_plane : separation_planes_) {
    prog.AddDecisionVariables(separation_plane.decision_variables);
    prog.AddBoundingBoxConstraint(-100, kInf,
                                  separation_plane.decision_variables);
  }
  drake::symbolic::Variable residue;
  if (option.maximize_residue) {
    residue = prog.NewContinuousVariables<1>()(0);
  }
  prog.AddIndeterminates(rational_forward_kinematics_.t());

  // Construct the environment that maps t_lower_vars -> t_lower_sol,
  // t_upper_vars -> t_upper_sol.
  const int t_size = rational_forward_kinematics_.t().rows();
  drake::symbolic::Environment env;
  for (int i = 0; i < t_size; ++i) {
    env.insert(t_lower_vars(i), t_lower_sol(i));
    env.insert(t_upper_vars(i), t_upper_sol(i));
  }
  const drake::symbolic::Monomial monomial_one{};
  // We will eventually add the constraint that verified_polynomials[i] are
  // non-negative. To do so, we introduce verified_polynomials_expected, such
  // that verified_polynomials[i] = verified_polynomials_expected[i], and
  // verified_polynomials_expected[i] is constructed to be non-negative.
  // We want to avoid constructing verified_polynomials_expected inside the for
  // loop, since it would cause dynamic memory allocation (for the Hessian
  // matrix of the polynomial) in each iteration of the for loop. This becomes
  // the major speed bottleneck. Instead, we first compute the total size of the
  // new variables for all the Hessians, and then allocate the memory for once.
  std::vector<drake::symbolic::Polynomial> verified_polynomials;
  verified_polynomials.reserve(bilinear_alternation_tuples.size());
  std::vector<int> verified_polynomials_expected_gram_rows;
  verified_polynomials_expected_gram_rows.reserve(
      bilinear_alternation_tuples.size());

  int verified_polynomial_count = 0;
  for (const auto& bilinear_alternation_tuple : bilinear_alternation_tuples) {
    verified_polynomials.push_back(bilinear_alternation_tuple.p);

    if (option.maximize_residue) {
      verified_polynomials.back().AddProduct(-residue, monomial_one);
    }
    const int gramian_rows =
        bilinear_alternation_tuple.lagrangian_monomial_basis.rows();
    int gramian_lower_entry_count = 0;
    const int num_constraints =
        static_cast<int>(bilinear_alternation_tuple.constraints.size());
    // We need to add the lower diagonal of the gramian matrices as decision
    // variables.
    VectorX<drake::symbolic::Variable> gramians_lower(
        gramian_rows * (gramian_rows + 1) / 2 * num_constraints);
    for (int i = 0; i < num_constraints; ++i) {
      for (int n = 0; n < gramian_rows; ++n) {
        for (int m = n; m < gramian_rows; ++m) {
          gramians_lower(gramian_lower_entry_count++) =
              bilinear_alternation_tuple.lagrangian_gramians[i](m, n);
        }
      }
    }
    prog.AddDecisionVariables(gramians_lower);
    for (int i = 0; i < num_constraints; ++i) {
      if (!std::isinf(env[bilinear_alternation_tuple.t_bound_vars[i]])) {
        // Now add the PSD/SDDP/DDP constraints on the gramian.
        switch (option.lagrangian_type) {
          case drake::solvers::MathematicalProgram::NonnegativePolynomial::
              kSos: {
            prog.AddPositiveSemidefiniteConstraint(
                bilinear_alternation_tuple.lagrangian_gramians[i]);
            break;
          }
          case drake::solvers::MathematicalProgram::NonnegativePolynomial::
              kSdsos: {
            prog.AddScaledDiagonallyDominantMatrixConstraint(
                bilinear_alternation_tuple.lagrangian_gramians[i]);
            break;
          }
          case drake::solvers::MathematicalProgram::NonnegativePolynomial::
              kDsos: {
            prog.AddPositiveDiagonallyDominantMatrixConstraint(
                bilinear_alternation_tuple.lagrangian_gramians[i]
                    .cast<drake::symbolic::Expression>());
            break;
          }
        }
        // Now evaluate the i'th constraint with t_lower_vars -> t_lower and
        // t_upper_vars -> t_upper.
        const drake::symbolic::Polynomial constraint_i =
            bilinear_alternation_tuple.constraints[i].EvaluatePartial(env);
        // Subtract lagrangians[i] * constraint_i
        verified_polynomials.back() -=
            bilinear_alternation_tuple.lagrangians[i] * constraint_i;
      }
    }
    verified_polynomials_expected_gram_rows.push_back(
        bilinear_alternation_tuple.lagrangian_monomial_basis.size());
    verified_polynomial_count++;
  }
  DRAKE_DEMAND(verified_polynomial_count ==
               static_cast<int>(bilinear_alternation_tuples.size()));
  ConstrainVerifiedPolynomialsNonnegative(
      verified_polynomials_expected_gram_rows, bilinear_alternation_tuples,
      verified_polynomials, option.link_polynomial_type, &prog);

  // Maximize the residue.
  if (option.maximize_residue) {
    prog.AddLinearCost(-residue);
  }

  drake::solvers::MathematicalProgramResult result;
  drake::solvers::MosekSolver mosek_solver;
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 0);
  prog.SetSolverOption(mosek_solver.id(), "MSK_DPAR_INTPNT_CO_TOL_INFEAS",
                       1e-10);
  mosek_solver.Solve(prog, {}, solver_options, &result);
  std::cout
      << "Find Lagrangian given box Mosek time: "
      << result.get_solver_details<drake::solvers::MosekSolver>().optimizer_time
      << "\n";
  drake::solvers::GenerateSDPA(
      prog, "find_lagrangian",
      drake::solvers::RemoveFreeVariableMethod::kTwoSlackVariables);
  // drake::solvers::ScsSolver scs_solver;
  // drake::solvers::MathematicalProgramResult result_scs;
  // scs_solver.Solve(prog, {}, {}, &result_scs);
  return result;
}

drake::solvers::MathematicalProgramResult
ConfigurationSpaceCollisionFreeRegion::SearchBoxGivenLagrangian(
    const std::vector<BoxVerificationTuple>& bilinear_alternation_tuples,
    const drake::VectorX<drake::symbolic::Variable>& t_lower_vars,
    const drake::VectorX<drake::symbolic::Variable>& t_upper_vars,
    const Eigen::VectorXd& t_lower_joint_limit,
    const Eigen::VectorXd& t_upper_joint_limit,
    const std::vector<std::vector<drake::symbolic::Polynomial>>&
        lagrangians_sol,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_init,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_init,
    const BilinearAlternationOption& option) const {
  DRAKE_DEMAND(bilinear_alternation_tuples.size() == lagrangians_sol.size());
  drake::solvers::MathematicalProgram prog;
  prog.AddIndeterminates(rational_forward_kinematics_.t());
  prog.AddDecisionVariables(t_lower_vars);
  prog.AddDecisionVariables(t_upper_vars);
  const int t_size = rational_forward_kinematics_.t().size();
  Eigen::VectorXd t_upper_vars_lower_bound, t_lower_vars_upper_bound;
  if (option.grow_all_dimension) {
    t_upper_vars_lower_bound = t_upper_init;
    t_lower_vars_upper_bound = t_lower_init;
  } else {
    t_upper_vars_lower_bound = Eigen::VectorXd::Zero(t_size);
    t_lower_vars_upper_bound = Eigen::VectorXd::Zero(t_size);
  }
  // Add constraint t_lower_joint_limit <= t_lower <= 0.
  prog.AddBoundingBoxConstraint(t_lower_joint_limit, t_lower_vars_upper_bound,
                                t_lower_vars);
  // Add constraint 0 <= t_upper <= t_upper_joint_limit.
  prog.AddBoundingBoxConstraint(t_upper_vars_lower_bound, t_upper_joint_limit,
                                t_upper_vars);
  DRAKE_DEMAND(option.box_offcenter_ratio >= 0);
  DRAKE_DEMAND(option.box_offcenter_ratio <= 1);
  for (int i = 0; i < t_size; ++i) {
    // |t_lower| >= box_offcenter_ratio * t_upper
    // t_upper >= box_offcenter_ratio * |t_lower|
    Eigen::Matrix2d A_box_offcenter;
    A_box_offcenter << -1, -option.box_offcenter_ratio,
        option.box_offcenter_ratio, 1;
    prog.AddLinearConstraint(A_box_offcenter, Eigen::Vector2d::Zero(),
                             Eigen::Vector2d::Constant(kInf),
                             drake::Vector2<drake::symbolic::Variable>(
                                 t_lower_vars(i), t_upper_vars(i)));
  }

  // Add the separation hyperplane parameters as decision variables.
  for (const auto& separation_plane : separation_planes_) {
    prog.AddDecisionVariables(separation_plane.decision_variables);
    prog.AddBoundingBoxConstraint(-100, kInf,
                                  separation_plane.decision_variables);
  }
  // We will impose the constraint that verified_polynomials[i] >= 0. To do so,
  // we need to introduce another polynomial that is non-negative by
  // construction, and equalize verified_polynomials[i] to this new polynomial.
  // Each new polynomial requires adding new decision variables. If we create
  // these new decision variables inside the for loop below, it would incur a
  // lot of dynamic memory allocation, which is very slow. Hence we first
  // compute the total size of the variables, and then allocate them all at
  // once.
  std::vector<drake::symbolic::Polynomial> verified_polynomials;
  std::vector<int> verified_polynomials_expected_gram_rows;
  verified_polynomials.reserve(
      static_cast<int>(bilinear_alternation_tuples.size()));
  verified_polynomials_expected_gram_rows.reserve(
      static_cast<int>(bilinear_alternation_tuples.size()));
  int verified_polynomials_size = 0;
  for (int i = 0; i < static_cast<int>(bilinear_alternation_tuples.size());
       ++i) {
    verified_polynomials.push_back(bilinear_alternation_tuples[i].p);
    DRAKE_DEMAND(bilinear_alternation_tuples[i].lagrangians.size() ==
                 lagrangians_sol[i].size());
    for (int j = 0; j < static_cast<int>(
                            bilinear_alternation_tuples[i].lagrangians.size());
         ++j) {
      verified_polynomials.back() -=
          lagrangians_sol[i][j] * bilinear_alternation_tuples[i].constraints[j];
    }
    verified_polynomials_expected_gram_rows.push_back(
        bilinear_alternation_tuples[i].lagrangian_monomial_basis.size());
    ++verified_polynomials_size;
  }
  DRAKE_DEMAND(verified_polynomials_size ==
               static_cast<int>(bilinear_alternation_tuples.size()));
  ConstrainVerifiedPolynomialsNonnegative(
      verified_polynomials_expected_gram_rows, bilinear_alternation_tuples,
      verified_polynomials, option.link_polynomial_type, &prog);

  switch (option.cost_type) {
    case BilinearAlternationCost::kBoxEdgeLengthSum: {
      // max(∑ᵢ t_upper(i) - t_lower(i))
      prog.AddLinearCost(
          -(t_upper_vars.cast<drake::symbolic::Expression>().sum() -
            t_lower_vars.cast<drake::symbolic::Expression>().sum()));
      break;
    }
    case BilinearAlternationCost::kVolume: {
      // max ∏ᵢ (t_upper(i) - t_lower(i)).
      Eigen::MatrixXd A(t_size, 2 * t_size);
      A.leftCols(t_size) = Eigen::MatrixXd::Identity(t_size, t_size);
      A.rightCols(t_size) = -Eigen::MatrixXd::Identity(t_size, t_size);
      VectorX<drake::symbolic::Variable> t_bound_vars(2 * t_size);
      t_bound_vars.head(t_size) = t_upper_vars;
      t_bound_vars.tail(t_size) = t_lower_vars;
      prog.AddMaximizeGeometricMeanCost(A, Eigen::VectorXd::Zero(t_size),
                                        t_bound_vars);
    }
  }

  drake::solvers::MosekSolver mosek_solver;

  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 0);
  prog.SetSolverOption(mosek_solver.id(), "MSK_DPAR_INTPNT_CO_TOL_INFEAS",
                       1e-10);
  const drake::solvers::MathematicalProgramResult result =
      mosek_solver.Solve(prog, {}, solver_options);
  std::cout
      << "Find box given Lagrangian mosek time: "
      << result.get_solver_details<drake::solvers::MosekSolver>().optimizer_time
      << "\n";
  // drake::solvers::ScsSolver scs_solver;
  // auto result_scs = scs_solver.Solve(prog, {}, {});

  return result;
}

double ConfigurationSpaceCollisionFreeRegion::FindLargestBoxThroughBinarySearch(
    const std::vector<BoxVerificationTuple>& box_verification_tuples,
    const Eigen::Ref<const Eigen::VectorXd>& negative_delta_q,
    const Eigen::Ref<const Eigen::VectorXd>& positive_delta_q,
    const Eigen::VectorXd& t_lower_joint_limit,
    const Eigen::VectorXd& t_upper_joint_limit,
    const Eigen::Ref<const Eigen::VectorXd>& q_star, double rho_lower_initial,
    double rho_upper_initial,
    const drake::VectorX<drake::symbolic::Variable>& t_lower_vars,
    const drake::VectorX<drake::symbolic::Variable>& t_upper_vars,
    const std::vector<FilteredCollisionPairsForBox>&
        filtered_collision_pairs_for_boxes,
    const BinarySearchOption& option) const {
  DRAKE_DEMAND(negative_delta_q.size() == positive_delta_q.size());
  DRAKE_DEMAND((negative_delta_q.array() <= 0).all());
  DRAKE_DEMAND((positive_delta_q.array() >= 0).all());
  DRAKE_DEMAND(rho_lower_initial >= 0);
  DRAKE_DEMAND(rho_lower_initial <= rho_upper_initial);
  DRAKE_DEMAND(option.delta_q_tol.size() ==
               rational_forward_kinematics_.plant().num_positions());
  double rho_upper = rho_upper_initial;
  double rho_lower = rho_lower_initial;

  while (
      ((rho_upper - rho_lower) * (positive_delta_q - negative_delta_q).array() >
       option.delta_q_tol.array())
          .any()) {
    const double rho = (rho_upper + rho_lower) / 2;
    const bool is_rho_feasible = IsBoxScalingFactorFeasible(
        box_verification_tuples, t_lower_joint_limit, t_upper_joint_limit,
        q_star, negative_delta_q, positive_delta_q, rho, t_lower_vars,
        t_upper_vars, filtered_collision_pairs_for_boxes, option);
    if (is_rho_feasible) {
      // rho is feasible.
      std::cout << "rho = " << rho << " is feasible.\n";
      rho_lower = rho;
    } else {
      // rho is infeasible.
      std::cout << "rho = " << rho << " is infeasible.\n";
      rho_upper = rho;
    }
  }
  return rho_lower;
}

drake::solvers::MathematicalProgramResult
ConfigurationSpaceCollisionFreeRegion::FindLargestBoxThroughBilinearAlternation(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const std::vector<BoxVerificationTuple>& bilinear_alternation_tuples,
    const BilinearAlternationOption& options,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower_init,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper_init,
    const Eigen::VectorXd& t_lower_joint_limit,
    const Eigen::VectorXd& t_upper_joint_limit,
    const drake::VectorX<drake::symbolic::Variable>& t_lower_vars,
    const drake::VectorX<drake::symbolic::Variable>& t_upper_vars,
    Eigen::VectorXd* t_lower_sol, Eigen::VectorXd* t_upper_sol,
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>* q_box_sols)
    const {
  drake::solvers::MathematicalProgramResult result;
  int iter = 0;
  *t_lower_sol = t_lower_init;
  *t_upper_sol = t_upper_init;
  std::vector<double> box_volume_cost;
  box_volume_cost.reserve(options.max_iteration);
  while (iter < options.max_iteration) {
    // Search for Lagrangian multiplier.
    result = SearchLagrangianMultiplier(bilinear_alternation_tuples,
                                        t_lower_vars, t_upper_vars,
                                        *t_lower_sol, *t_upper_sol, options);
    if (!result.is_success()) {
      break;
    }
    // Now retrieve the solution on the Lagrangian multiplier.
    std::vector<std::vector<drake::symbolic::Polynomial>> lagrangians(
        bilinear_alternation_tuples.size());
    for (int i = 0; i < static_cast<int>(bilinear_alternation_tuples.size());
         ++i) {
      lagrangians[i].reserve(bilinear_alternation_tuples[i].lagrangians.size());
      for (int j = 0;
           j <
           static_cast<int>(bilinear_alternation_tuples[i].lagrangians.size());
           ++j) {
        const Eigen::MatrixXd lagrangian_gramian_sol = result.GetSolution(
            bilinear_alternation_tuples[i].lagrangian_gramians[j]);
        lagrangians[i].push_back(ComputePolynomialFromGramian(
            bilinear_alternation_tuples[i].lagrangian_monomial_basis,
            lagrangian_gramian_sol));
      }
    }

    // Search for the new box.
    result = SearchBoxGivenLagrangian(bilinear_alternation_tuples, t_lower_vars,
                                      t_upper_vars, t_lower_joint_limit,
                                      t_upper_joint_limit, lagrangians,
                                      t_lower_init, t_upper_init, options);
    if (!result.is_success()) {
      break;
    }
    std::cout << "iteration: " << iter
              << ", max box cost: " << -result.get_optimal_cost() << "\n";
    *t_lower_sol = result.GetSolution(t_lower_vars);
    *t_upper_sol = result.GetSolution(t_upper_vars);
    // Now compute the new box.
    const Eigen::VectorXd positive_delta_q =
        (t_upper_sol->array().atan() * 2).matrix();
    const Eigen::VectorXd negative_delta_q =
        (t_lower_sol->array().atan() * 2).matrix();
    Eigen::VectorXd q_upper_sol = q_star + positive_delta_q;
    Eigen::VectorXd q_lower_sol = q_star + negative_delta_q;
    q_box_sols->emplace_back(q_lower_sol, q_upper_sol);
    box_volume_cost.push_back(result.get_optimal_cost());
    if (iter > 0 && box_volume_cost[iter - 1] - box_volume_cost[iter] <=
                        options.optimal_threshold) {
      // If the change of cost is below the threshold.
      break;
    }

    iter++;
  }

  return result;
}

bool ConfigurationSpaceCollisionFreeRegion::IsBoxScalingFactorFeasible(
    const std::vector<BoxVerificationTuple>& box_verification_tuples,
    const Eigen::VectorXd& t_lower_joint_limit,
    const Eigen::VectorXd& t_upper_joint_limit,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const Eigen::Ref<const Eigen::VectorXd>& negative_delta_q,
    const Eigen::Ref<const Eigen::VectorXd>& positive_delta_q, double rho,
    const drake::VectorX<drake::symbolic::Variable>& t_lower_vars,
    const drake::VectorX<drake::symbolic::Variable>& t_upper_vars,
    const std::vector<FilteredCollisionPairsForBox>&
        filtered_collision_pairs_for_boxes,
    const VerificationOption& option) const {
  DRAKE_DEMAND((negative_delta_q.array() <= 0).all());
  DRAKE_DEMAND((positive_delta_q.array() >= 0).all());
  Eigen::VectorXd t_lower(rational_forward_kinematics_.t().size());
  Eigen::VectorXd t_upper(rational_forward_kinematics_.t().size());
  t_lower = rational_forward_kinematics_.ComputeTValue(
      q_star + rho * negative_delta_q, q_star, true);
  t_upper = rational_forward_kinematics_.ComputeTValue(
      q_star + rho * positive_delta_q, q_star, true);
  for (int i = 0; i < rational_forward_kinematics_.t().size(); ++i) {
    t_lower(i) = std::max(t_lower(i), t_lower_joint_limit(i));
    t_upper(i) = std::min(t_upper(i), t_upper_joint_limit(i));
  }
  const Eigen::VectorXd q_lower_joint_limit =
      rational_forward_kinematics_.plant().GetPositionLowerLimits();
  const Eigen::VectorXd q_upper_joint_limit =
      rational_forward_kinematics_.plant().GetPositionUpperLimits();
  Eigen::VectorXd q_lower = q_star + rho * negative_delta_q;
  Eigen::VectorXd q_upper = q_star + rho * positive_delta_q;
  for (int i = 0; i < q_star.rows(); ++i) {
    q_lower(i) = std::max(q_lower(i), q_lower_joint_limit(i));
    q_upper(i) = std::min(q_upper(i), q_upper_joint_limit(i));
  }
  const FilteredCollisionPairs filtered_collision_pairs =
      ExtractFilteredCollisionPairsForBox(q_lower, q_upper,
                                          filtered_collision_pairs_for_boxes);
  std::vector<BoxVerificationTuple> box_verification_tuples_filtered;
  box_verification_tuples_filtered.reserve(box_verification_tuples.size());
  for (const auto& box_verification_tuple : box_verification_tuples) {
    if (filtered_collision_pairs.count(box_verification_tuple.geometry_pair) ==
        0) {
      box_verification_tuples_filtered.push_back(box_verification_tuple);
    }
  }

  const SearchLagrangianOption search_lagrangian_option(option, false);
  const auto result = SearchLagrangianMultiplier(
      box_verification_tuples_filtered, t_lower_vars, t_upper_vars, t_lower,
      t_upper, search_lagrangian_option);
  return result.is_success();
}

int ConfigurationSpaceCollisionFreeRegion::SequentiallyDoubleBoxSize(
    const std::vector<BoxVerificationTuple>& box_verification_tuples,
    const Eigen::Ref<const Eigen::VectorXd>& negative_delta_q,
    const Eigen::Ref<const Eigen::VectorXd>& positive_delta_q,
    const Eigen::VectorXd& t_lower_joint_limit,
    const Eigen::VectorXd& t_upper_joint_limit,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const drake::VectorX<drake::symbolic::Variable>& t_lower_vars,
    const drake::VectorX<drake::symbolic::Variable>& t_upper_vars,
    const std::vector<FilteredCollisionPairsForBox>&
        filtered_collision_pairs_for_boxes,
    const BinarySearchOption& option) const {
  // First find an upper bound that is infeasible.
  int rho_lower_initial = 1;
  int rho_upper_initial = 2;
  bool find_rho_upper_bound = false;
  while (!find_rho_upper_bound) {
    rho_upper_initial = 2 * rho_lower_initial;
    if (IsBoxScalingFactorFeasible(
            box_verification_tuples, t_lower_joint_limit, t_upper_joint_limit,
            q_star, negative_delta_q, positive_delta_q, rho_upper_initial,
            t_lower_vars, t_upper_vars, filtered_collision_pairs_for_boxes,
            option)) {
      rho_lower_initial = rho_upper_initial;
    } else {
      find_rho_upper_bound = true;
    }
  }
  return rho_lower_initial;
}

std::vector<LinkVertexOnPlaneSideRational>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    std::shared_ptr<const ConvexPolytope> link_polytope,
    std::shared_ptr<const ConvexPolytope> other_side_link_polytope,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    BodyIndex expressed_body_index,
    const Eigen::Ref<const Vector3<drake::symbolic::Expression>>& a_A,
    const Eigen::Ref<const Eigen::Vector3d>& p_AC, PlaneSide plane_side,
    SeparatingPlaneOrder a_order) {
  // Compute the link pose
  const auto X_AB =
      rational_forward_kinematics.CalcLinkPoseAsMultilinearPolynomial(
          q_star, link_polytope->body_index(), expressed_body_index);

  return GenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link_polytope, other_side_link_polytope,
      X_AB, a_A, p_AC, plane_side, a_order);
}

std::vector<LinkVertexOnPlaneSideRational>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    std::shared_ptr<const ConvexPolytope> link_polytope,
    std::shared_ptr<const ConvexPolytope> other_side_link_polytope,
    const RationalForwardKinematics::Pose<drake::symbolic::Polynomial>&
        X_AB_multilinear,
    const Eigen::Ref<const Vector3<drake::symbolic::Expression>>& a_A,
    const Eigen::Ref<const Eigen::Vector3d>& p_AC, PlaneSide plane_side,
    SeparatingPlaneOrder a_order) {
  std::vector<LinkVertexOnPlaneSideRational> rational_fun;
  rational_fun.reserve(link_polytope->p_BV().cols());
  const drake::symbolic::Monomial monomial_one{};
  Vector3<drake::symbolic::Polynomial> a_A_poly;
  for (int i = 0; i < 3; ++i) {
    a_A_poly(i) = drake::symbolic::Polynomial({{monomial_one, a_A(i)}});
  }
  for (int i = 0; i < link_polytope->p_BV().cols(); ++i) {
    // Step 1: Compute vertex position.
    const Vector3<drake::symbolic::Polynomial> p_AVi =
        X_AB_multilinear.p_AB +
        X_AB_multilinear.R_AB * link_polytope->p_BV().col(i);

    // Step 2: Compute a_A.dot(p_AVi - p_AC)
    const drake::symbolic::Polynomial point_on_hyperplane_side =
        a_A_poly.dot(p_AVi - p_AC);

    // Step 3: Convert the multilinear polynomial to rational function.
    rational_fun.emplace_back(
        rational_forward_kinematics
            .ConvertMultilinearPolynomialToRationalFunction(
                plane_side == PlaneSide::kPositive
                    ? point_on_hyperplane_side - 1
                    : 1 - point_on_hyperplane_side),
        link_polytope, X_AB_multilinear.frame_A_index, other_side_link_polytope,
        a_A, plane_side, a_order);
  }

  for (int i = 0; i < link_polytope->r_B().cols(); ++i) {
    // Step 1: Compute ray.
    const Vector3<drake::symbolic::Polynomial> ri_A =
        X_AB_multilinear.R_AB * link_polytope->r_B().col(i);

    // Step 2: Compute a_A.dot(ri_A)
    const drake::symbolic::Polynomial point_on_hyperplane_side =
        a_A_poly.dot(ri_A);

    // Step 3: Convert the multilinear polynomial to rational function.
    rational_fun.emplace_back(
        rational_forward_kinematics
            .ConvertMultilinearPolynomialToRationalFunction(
                plane_side == PlaneSide::kPositive ? point_on_hyperplane_side
                                                   : -point_on_hyperplane_side),
        link_polytope, X_AB_multilinear.frame_A_index, other_side_link_polytope,
        a_A, plane_side, a_order);
  }
  return rational_fun;
}

void AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
    drake::solvers::MathematicalProgram* prog,
    const drake::symbolic::RationalFunction& polytope_on_one_side_rational,
    const Eigen::Ref<const VectorX<drake::symbolic::Polynomial>>&
        t_minus_t_lower,
    const Eigen::Ref<const VectorX<drake::symbolic::Polynomial>>&
        t_upper_minus_t,
    const Eigen::Ref<const VectorX<drake::symbolic::Monomial>>& monomial_basis,
    const VerificationOption& verification_option) {
  DRAKE_DEMAND(t_minus_t_lower.size() == t_upper_minus_t.size());
  drake::symbolic::Polynomial verified_polynomial =
      polytope_on_one_side_rational.numerator();
  for (int i = 0; i < t_minus_t_lower.size(); ++i) {
    const auto l_lower =
        prog->NewNonnegativePolynomial(monomial_basis,
                                       verification_option.lagrangian_type)
            .first;
    const auto l_upper =
        prog->NewNonnegativePolynomial(monomial_basis,
                                       verification_option.lagrangian_type)
            .first;
    verified_polynomial -= l_lower * t_minus_t_lower(i);
    verified_polynomial -= l_upper * t_upper_minus_t(i);
  }
  // Replace the following lines with prog->AddSosConstraint when we resolve
  // the speed issue.
  const drake::symbolic::Polynomial verified_polynomial_expected =
      prog->NewNonnegativePolynomial(monomial_basis,
                                     verification_option.link_polynomial_type)
          .first;
  const drake::symbolic::Polynomial poly_diff{verified_polynomial -
                                              verified_polynomial_expected};
  for (const auto& item : poly_diff.monomial_to_coefficient_map()) {
    prog->AddLinearEqualityConstraint(item.second, 0);
  }
}
}  // namespace multibody
}  // namespace drake
