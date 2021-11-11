#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"

#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"
#include "drake/solvers/mosek_solver.h"

namespace drake {
namespace multibody {
using symbolic::RationalFunction;

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
  std::unordered_map<OrderedKinematicsChain, VectorX<symbolic::Variable>,
                     OrderedKinematicsChainHash>
      map_link_obstacle_to_t;
  for (const auto& obstacle : obstacles_) {
    DRAKE_DEMAND(obstacle->body_index() == plant.world_body().index());
    for (const auto& link_polytope_pairs : link_polytopes_) {
      for (const auto& link_polytope : link_polytope_pairs.second) {
        Vector3<symbolic::Expression> a;
        VectorX<symbolic::Variable> a_decision_vars;
        if (a_order_ == SeparatingPlaneOrder::kConstant) {
          a_decision_vars.resize(3);
          for (int i = 0; i < 3; ++i) {
            a_decision_vars(i) = symbolic::Variable(
                "a" + std::to_string(separation_planes_.size() * 3 + i));
            a(i) = a_decision_vars(i);
          }
        } else if (a_order_ == SeparatingPlaneOrder::kAffine) {
          // Get t on the kinematics chain.
          const OrderedKinematicsChain link_obstacle(
              link_polytope->body_index(), obstacle->body_index());
          auto it = map_link_obstacle_to_t.find(link_obstacle);
          VectorX<symbolic::Variable> t_on_chain;
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
          Matrix3X<symbolic::Variable> A(3, t_on_chain.rows());
          Vector3<symbolic::Variable> b;
          for (int j = 0; j < t_on_chain.rows(); ++j) {
            for (int i = 0; i < 3; ++i) {
              A(i, j) = symbolic::Variable(
                  "A" + std::to_string(separation_planes_.size()) + "[" +
                  std::to_string(3 * j + i) + "]");
              a_decision_vars(3 * j + i) = A(i, j);
            }
          }
          for (int i = 0; i < 3; ++i) {
            b(i) = symbolic::Variable(
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
  return filtered_collision_pairs.count(std::make_pair(id1, id2)) > 0 ||
         filtered_collision_pairs.count(std::make_pair(id2, id1)) > 0;
}

std::vector<LinkVertexOnPlaneSideRational>
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
    const symbolic::Variables middle_to_link_variables(
        rational_forward_kinematics_.FindTOnPath(expressed_body_index,
                                                 body_to_polytopes.first));
    const symbolic::Variables world_to_middle_variables(
        rational_forward_kinematics_.FindTOnPath(world_index,
                                                 expressed_body_index));
    // Compute the pose of the link (B) and the world (W) in the expressed link
    // (A).
    // TODO(hongkai.dai): save the poses to an unordered set.
    const RationalForwardKinematics::Pose<symbolic::Polynomial> X_AB =
        rational_forward_kinematics_.CalcLinkPoseAsMultilinearPolynomial(
            q_star, body_to_polytopes.first, expressed_body_index);
    const RationalForwardKinematics::Pose<symbolic::Polynomial> X_AW =
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
              *context, rational_forward_kinematics_.plant()
                            .get_body(obstacle->body_index())
                            .body_frame(),
              obstacle->p_BC(), rational_forward_kinematics_.plant()
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
    std::unordered_map<
        OrderedKinematicsChain,
        std::pair<VectorX<symbolic::Variable>, VectorX<symbolic::Monomial>>,
        OrderedKinematicsChainHash>*
        map_ordered_kinematics_chain_to_monomial_basis,
    VectorX<symbolic::Variable>* t_chain,
    VectorX<symbolic::Monomial>* monomial_basis_chain) {
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
    *monomial_basis_chain =
        GenerateMonomialBasisWithOrderUpToOne(symbolic::Variables(*t_chain));
    map_ordered_kinematics_chain_to_monomial_basis->emplace_hint(
        it, std::make_pair(ordered_kinematics_chain,
                           std::make_pair(*t_chain, *monomial_basis_chain)));
  } else {
    *t_chain = it->second.first;
    *monomial_basis_chain = it->second.second;
  }
}

void FindMonomialBasisForAffineSeparatingPlane(
    const RationalForwardKinematics& rational_forward_kinematics,
    const LinkVertexOnPlaneSideRational& rational,
    std::unordered_map<
        UnorderedKinematicsChain,
        std::pair<VectorX<symbolic::Variable>, VectorX<symbolic::Monomial>>,
        UnorderedKinematicsChainHash>*
        map_unordered_kinematics_chain_to_monomial_basis,
    VectorX<symbolic::Variable>* t_chain,
    VectorX<symbolic::Monomial>* monomial_basis_chain) {
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
    const VectorX<symbolic::Variable> t_monomial =
        rational_forward_kinematics.FindTOnPath(
            rational.link_polytope->body_index(),
            rational.expressed_body_index);
    *monomial_basis_chain =
        GenerateMonomialBasisWithOrderUpToOne(symbolic::Variables(t_monomial));
    map_unordered_kinematics_chain_to_monomial_basis->emplace_hint(
        it, std::make_pair(unordered_kinematics_chain,
                           std::make_pair(*t_chain, *monomial_basis_chain)));
  } else {
    *t_chain = it->second.first;
    *monomial_basis_chain = it->second.second;
  }
}
}  // namespace

std::unique_ptr<solvers::MathematicalProgram>
ConfigurationSpaceCollisionFreeRegion::ConstructProgramToVerifyCollisionFreeBox(
    const std::vector<LinkVertexOnPlaneSideRational>& rationals,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper,
    const FilteredCollisionPairs& filtered_collision_pairs,
    const VerificationOption& verification_option) const {
  auto prog = std::make_unique<solvers::MathematicalProgram>();
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
  std::unordered_map<symbolic::Variable::Id,
                     std::pair<symbolic::Polynomial, symbolic::Polynomial>>
      map_t_to_box_bounds;
  map_t_to_box_bounds.reserve(t.size());
  const symbolic::Monomial monomial_one{};
  for (int i = 0; i < t.size(); ++i) {
    map_t_to_box_bounds.emplace(
        rational_forward_kinematics_.t()(i).get_id(),
        std::make_pair(symbolic::Polynomial({{symbolic::Monomial(t(i)), 1},
                                             {monomial_one, -t_lower(i)}}),
                       symbolic::Polynomial({{symbolic::Monomial(t(i)), -1},
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
  std::unordered_map<
      OrderedKinematicsChain,
      std::pair<VectorX<symbolic::Variable>, VectorX<symbolic::Monomial>>,
      OrderedKinematicsChainHash>
      map_ordered_kinematics_chain_to_monomial_basis;

  // This is for affine separating plane.
  std::unordered_map<
      UnorderedKinematicsChain,
      std::pair<VectorX<symbolic::Variable>, VectorX<symbolic::Monomial>>,
      UnorderedKinematicsChainHash>
      map_unordered_kinematics_chain_to_monomial_basis;

  for (const auto& rational : rationals) {
    VectorX<symbolic::Variable> t_chain;
    VectorX<symbolic::Monomial> monomial_basis_chain;
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
    VectorX<symbolic::Polynomial> t_minus_t_lower(t_chain.size());
    VectorX<symbolic::Polynomial> t_upper_minus_t(t_chain.size());
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
    double rho_lower_initial, double rho_upper_initial, double rho_tolerance,
    const VerificationOption& verification_option) const {
  DRAKE_DEMAND(negative_delta_q.size() == positive_delta_q.size());
  DRAKE_DEMAND((negative_delta_q.array() <= 0).all());
  DRAKE_DEMAND((positive_delta_q.array() >= 0).all());
  DRAKE_DEMAND(rho_lower_initial >= 0);
  DRAKE_DEMAND(rho_lower_initial <= rho_upper_initial);
  DRAKE_DEMAND(rho_tolerance > 0);
  DRAKE_DEMAND(
      ((positive_delta_q * rho_upper_initial).array() <= M_PI_2).all());
  DRAKE_DEMAND(
      ((negative_delta_q * rho_upper_initial).array() >= -M_PI_2).all());
  const int nq = rational_forward_kinematics_.plant().num_positions();
  DRAKE_DEMAND(q_star.rows() == nq);
  Eigen::VectorXd q_upper(nq);
  Eigen::VectorXd q_lower(nq);
  for (JointIndex i{0}; i < rational_forward_kinematics_.plant().num_joints();
       ++i) {
    const auto& joint = rational_forward_kinematics_.plant().get_joint(i);
    q_upper.segment(joint.position_start(), joint.num_positions()) =
        joint.position_upper_limits();
    q_lower.segment(joint.position_start(), joint.num_positions()) =
        joint.position_lower_limits();
  }
  const Eigen::VectorXd t_upper_limit =
      rational_forward_kinematics_.ComputeTValue(q_upper, q_star, true);
  const Eigen::VectorXd t_lower_limit =
      rational_forward_kinematics_.ComputeTValue(q_lower, q_star, true);
  double rho_upper = rho_upper_initial;
  double rho_lower = rho_lower_initial;

  const std::vector<LinkVertexOnPlaneSideRational> rationals =
      GenerateLinkOnOneSideOfPlaneRationals(q_star, filtered_collision_pairs);
  solvers::MosekSolver solver;
  solvers::MathematicalProgramResult result;
  while (rho_upper - rho_lower > rho_tolerance) {
    const double rho = (rho_upper + rho_lower) / 2;
    Eigen::VectorXd t_lower(rational_forward_kinematics_.t().size());
    Eigen::VectorXd t_upper(rational_forward_kinematics_.t().size());
    t_lower = rational_forward_kinematics_.ComputeTValue(
        q_star + rho * negative_delta_q, q_star, true);
    t_upper = rational_forward_kinematics_.ComputeTValue(
        q_star + rho * positive_delta_q, q_star, true);
    for (int i = 0; i < rational_forward_kinematics_.t().size(); ++i) {
      t_lower(i) = std::max(t_lower(i), t_lower_limit(i));
      t_upper(i) = std::min(t_upper(i), t_upper_limit(i));
      if (std::isinf(t_lower(i)) || std::isinf(t_upper(i))) {
        throw std::runtime_error(
            "ConfigurationSpaceCollisionFreeRegion: t_lower = -inf or t_upper "
            "= inf is not handled yet.");
      }
    }
    auto prog = ConstructProgramToVerifyCollisionFreeBox(
        rationals, t_lower, t_upper, filtered_collision_pairs,
        verification_option);
    solver.Solve(*prog, {}, {}, &result);
    if (result.get_solution_result() ==
        solvers::SolutionResult::kSolutionFound) {
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

bool ConfigurationSpaceCollisionFreeRegion::IsPostureCollisionFree(
    const systems::Context<double>& context) const {
  const auto& plant = rational_forward_kinematics_.plant();
  math::RigidTransform<double> X_WW;
  X_WW.SetIdentity();
  for (const auto& link_and_polytopes : link_polytopes_) {
    const Eigen::Isometry3d X_WB = plant.EvalBodyPoseInWorld(
        context, plant.get_body(link_and_polytopes.first));
    for (const auto& link_polytope : link_and_polytopes.second) {
      for (const auto& obstacle : obstacles_) {
        if (link_polytope->IsInCollision(
                *obstacle, math::RigidTransform<double>(X_WB), X_WW)) {
          return true;
        }
      }
    }
  }
  return false;
}

std::vector<LinkVertexOnPlaneSideRational>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    std::shared_ptr<const ConvexPolytope> link_polytope,
    std::shared_ptr<const ConvexPolytope> other_side_link_polytope,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    BodyIndex expressed_body_index,
    const Eigen::Ref<const Vector3<symbolic::Expression>>& a_A,
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
    const RationalForwardKinematics::Pose<symbolic::Polynomial>&
        X_AB_multilinear,
    const Eigen::Ref<const Vector3<symbolic::Expression>>& a_A,
    const Eigen::Ref<const Eigen::Vector3d>& p_AC, PlaneSide plane_side,
    SeparatingPlaneOrder a_order) {
  std::vector<LinkVertexOnPlaneSideRational> rational_fun;
  rational_fun.reserve(link_polytope->p_BV().cols());
  const symbolic::Monomial monomial_one{};
  Vector3<symbolic::Polynomial> a_A_poly;
  for (int i = 0; i < 3; ++i) {
    a_A_poly(i) = symbolic::Polynomial({{monomial_one, a_A(i)}});
  }
  for (int i = 0; i < link_polytope->p_BV().cols(); ++i) {
    // Step 1: Compute vertex position.
    const Vector3<symbolic::Polynomial> p_AVi =
        X_AB_multilinear.p_AB +
        X_AB_multilinear.R_AB * link_polytope->p_BV().col(i);

    // Step 2: Compute a_A.dot(p_AVi - p_AC)
    const symbolic::Polynomial point_on_hyperplane_side =
        a_A_poly.dot(p_AVi - p_AC);

    // Step 3: Convert the multilinear polynomial to rational function.
    rational_fun.emplace_back(
        rational_forward_kinematics
            .ConvertMultilinearPolynomialToRationalFunction(
                plane_side == PlaneSide::kPositive
                    ? point_on_hyperplane_side - 1
                    : 1 - point_on_hyperplane_side),
        link_polytope, X_AB_multilinear.frame_A_index, other_side_link_polytope,
        link_polytope->p_BV().col(i), a_A, plane_side, a_order);
  }
  return rational_fun;
}

void AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
    solvers::MathematicalProgram* prog,
    const symbolic::RationalFunction& polytope_on_one_side_rational,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& t_minus_t_lower,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& t_upper_minus_t,
    const Eigen::Ref<const VectorX<symbolic::Monomial>>& monomial_basis,
    const VerificationOption& verification_option) {
  DRAKE_DEMAND(t_minus_t_lower.size() == t_upper_minus_t.size());
  symbolic::Polynomial verified_polynomial =
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
  // the
  // speed issue.
  const symbolic::Polynomial verified_polynomial_expected =
      prog->NewNonnegativePolynomial(monomial_basis,
                                     verification_option.link_polynomial_type)
          .first;
  const symbolic::Polynomial poly_diff{verified_polynomial -
                                       verified_polynomial_expected};
  for (const auto& item : poly_diff.monomial_to_coefficient_map()) {
    prog->AddLinearEqualityConstraint(item.second, 0);
  }
}
}  // namespace multibody
}  // namespace drake
