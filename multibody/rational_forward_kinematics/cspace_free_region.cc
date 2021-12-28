#include "drake/multibody/rational_forward_kinematics/cspace_free_region.h"

#include <limits>

#include <fmt/format.h>

#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace multibody {
const double kInf = std::numeric_limits<double>::infinity();

namespace {
struct DirectedKinematicsChain {
  DirectedKinematicsChain(BodyIndex m_start, BodyIndex m_end)
      : start(m_start), end(m_end) {}

  bool operator==(const DirectedKinematicsChain& other) const {
    return start == other.start && end == other.end;
  }

  BodyIndex start;
  BodyIndex end;
};

struct DirectedKinematicsChainHash {
  size_t operator()(const DirectedKinematicsChain& p) const {
    return p.start * 100 + p.end;
  }
};

// map the kinematics chain to monomial_basis.
// If the separating plane has affine order, then the polynomial we want to
// verify (in the numerator of 1 - aᵀ(x−c) or aᵀ(x−c)-1) only contains the
// monomials tⱼ * ∏(tᵢ, dᵢ), where tᵢ is a t on the "half chain" from the
// expressed frame to either the link or the obstacle, and dᵢ<=2. Namely at
// most one variable has degree 3, all the other variables have degree <= 2.
// If the separating plane has constant order, then the polynomial we want to
// verify only contains the monomials ∏(tᵢ, dᵢ) with dᵢ<=2. In both cases, the
// monomial basis for the SOS polynomial contains all monomials that each
// variable has degree at most 1, and each variable is on the "half chain" from
// the expressed body to this link (either the robot link or the obstacle).
void FindMonomialBasisForPolytopicRegion(
    const RationalForwardKinematics& rational_forward_kinematics,
    const LinkVertexOnPlaneSideRational& rational,
    std::unordered_map<SortedPair<multibody::BodyIndex>,
                       VectorX<drake::symbolic::Monomial>>*
        map_chain_to_monomial_basis,
    VectorX<drake::symbolic::Monomial>* monomial_basis_halfchain) {
  // First check if the monomial basis for this kinematics chain has been
  // computed.
  const SortedPair<multibody::BodyIndex> kinematics_chain(
      rational.link_polytope->body_index(), rational.expressed_body_index);
  const auto it = map_chain_to_monomial_basis->find(kinematics_chain);
  if (it == map_chain_to_monomial_basis->end()) {
    const auto t_halfchain = rational_forward_kinematics.FindTOnPath(
        rational.link_polytope->body_index(), rational.expressed_body_index);
    *monomial_basis_halfchain = GenerateMonomialBasisWithOrderUpToOne(
        drake::symbolic::Variables(t_halfchain));
    map_chain_to_monomial_basis->emplace_hint(
        it, std::make_pair(kinematics_chain, *monomial_basis_halfchain));
  } else {
    *monomial_basis_halfchain = it->second;
  }
}

/**
 * For a polyhedron C * x <= d, find the lower and upper bound for x(i).
 */
void BoundPolyhedronByBox(const Eigen::MatrixXd& C, const Eigen::VectorXd& d,
                          Eigen::VectorXd* x_lower, Eigen::VectorXd* x_upper) {
  solvers::MathematicalProgram prog;
  const auto x = prog.NewContinuousVariables(C.cols());
  prog.AddLinearConstraint(C, Eigen::VectorXd::Constant(d.rows(), -kInf), d, x);
  Eigen::VectorXd cost_coeff = Eigen::VectorXd::Zero(x.rows());
  auto cost = prog.AddLinearCost(cost_coeff, x);
  x_lower->resize(x.rows());
  x_upper->resize(x.rows());
  for (int i = 0; i < x.rows(); ++i) {
    // Compute x_lower(i).
    cost_coeff.setZero();
    cost_coeff(i) = 1;
    cost.evaluator()->UpdateCoefficients(cost_coeff);
    auto result = solvers::Solve(prog);
    (*x_lower)(i) = result.get_optimal_cost();
    // Compute x_upper(i).
    cost_coeff(i) = -1;
    cost.evaluator()->UpdateCoefficients(cost_coeff);
    result = solvers::Solve(prog);
    (*x_upper)(i) = -result.get_optimal_cost();
  }
}
}  // namespace

CspaceFreeRegion::CspaceFreeRegion(
    const MultibodyPlant<double>& plant,
    const std::vector<const ConvexPolytope*>& link_polytopes,
    const std::vector<const ConvexPolytope*>& obstacles,
    SeparatingPlaneOrder plane_order, CspaceRegionType cspace_region_type)
    : rational_forward_kinematics_(plant),
      obstacles_{obstacles},
      plane_order_{plane_order},
      cspace_region_type_{cspace_region_type} {
  // First group the link polytopes by the attached link.
  for (const auto& link_polytope : link_polytopes) {
    DRAKE_DEMAND(link_polytope->body_index() != plant.world_body().index());
    const auto it = link_polytopes_.find(link_polytope->body_index());
    if (it == link_polytopes_.end()) {
      link_polytopes_.emplace_hint(
          it,
          std::make_pair(link_polytope->body_index(),
                         std::vector<const ConvexPolytope*>({link_polytope})));
    } else {
      it->second.push_back(link_polytope);
    }
  }
  // Now create the separating planes.
  // By default, we only consider the pairs between a link polytope and a
  // world obstacle.
  // TODO(hongkai.dai): consider the self collision between link obstacles.
  separating_planes_.reserve(link_polytopes.size() * obstacles.size());
  // If we verify an axis-algined bounding box in the C-space, then for each
  // pair of (link, obstacle), then we only need to consider variables t for the
  // joints on the kinematics chain between the link and the obstacle; if we
  // verify a generic bounding box, then we need to consider t for all joint
  // angles.
  std::unordered_map<SortedPair<BodyIndex>, VectorX<symbolic::Variable>>
      map_link_obstacle_to_t;
  for (const auto& obstacle : obstacles_) {
    DRAKE_DEMAND(obstacle->body_index() == plant.world_body().index());
    for (const auto& [link, polytopes_on_link] : link_polytopes_) {
      for (const auto& link_polytope : polytopes_on_link) {
        Vector3<symbolic::Expression> a;
        symbolic::Expression b;
        const symbolic::Monomial monomial_one{};
        VectorX<symbolic::Variable> plane_decision_vars;
        if (plane_order_ == SeparatingPlaneOrder::kConstant) {
          plane_decision_vars.resize(4);
          for (int i = 0; i < 3; ++i) {
            plane_decision_vars(i) = symbolic::Variable(
                "a" + std::to_string(separating_planes_.size() * 3 + i));
            plane_decision_vars(3) = symbolic::Variable(
                "b" + std::to_string(separating_planes_.size()));
          }
          a = plane_decision_vars.head<3>().cast<symbolic::Expression>();
          b = plane_decision_vars(3);
        } else if (plane_order_ == SeparatingPlaneOrder::kAffine) {
          VectorX<symbolic::Variable> t_for_plane;
          if (cspace_region_type_ == CspaceRegionType::kGenericPolytope) {
            t_for_plane = rational_forward_kinematics_.t();
          } else if (cspace_region_type_ ==
                     CspaceRegionType::kAxisAlignedBoundingBox) {
            SortedPair<BodyIndex> link_obstacle(link_polytope->body_index(),
                                                obstacle->body_index());
            auto it = map_link_obstacle_to_t.find(link_obstacle);
            if (it == map_link_obstacle_to_t.end()) {
              t_for_plane = rational_forward_kinematics_.FindTOnPath(
                  obstacle->body_index(), link_polytope->body_index());
              map_link_obstacle_to_t.emplace_hint(it, link_obstacle,
                                                  t_for_plane);
            } else {
              t_for_plane = it->second;
            }
          }
          // Now create the variable a_coeff, a_constant, b_coeff, b_constant,
          // such that a = a_coeff * t_for_plane + a_constant, and b = b_coeff *
          // t_for_plane + b_constant.
          Matrix3X<symbolic::Variable> a_coeff(3, t_for_plane.rows());
          Vector3<symbolic::Variable> a_constant;
          for (int i = 0; i < 3; ++i) {
            a_constant(i) = symbolic::Variable(
                fmt::format("a_constant{}({})", separating_planes_.size(), i));
            for (int j = 0; j < t_for_plane.rows(); ++j) {
              a_coeff(i, j) = symbolic::Variable(fmt::format(
                  "a_coeff{}({}, {})", separating_planes_.size(), i, j));
            }
          }
          VectorX<symbolic::Variable> b_coeff(t_for_plane.rows());
          for (int i = 0; i < t_for_plane.rows(); ++i) {
            b_coeff(i) = symbolic::Variable(
                fmt::format("b_coeff{}({})", separating_planes_.size(), i));
          }
          symbolic::Variable b_constant(
              fmt::format("b_constant{}", separating_planes_.size()));
          // Now construct a(i) = a_coeff.row(i) * t_for_plane + a_constant(i).
          a = a_coeff * t_for_plane + a_constant;
          b = b_coeff.cast<symbolic::Expression>().dot(t_for_plane) +
              b_constant;
          // Now put a_coeff, a_constant, b_coeff, b_constant to
          // plane_decision_vars.
          plane_decision_vars.resize(4 * t_for_plane.rows() + 4);
          int var_count = 0;
          for (int i = 0; i < 3; ++i) {
            plane_decision_vars.segment(var_count, t_for_plane.rows()) =
                a_coeff.row(i);
            var_count += t_for_plane.rows();
          }
          plane_decision_vars.segment<3>(var_count) = a_constant;
          var_count += 3;
          plane_decision_vars.segment(var_count, t_for_plane.rows()) = b_coeff;
          var_count += t_for_plane.rows();
          plane_decision_vars(var_count) = b_constant;
          var_count++;
        }
        separating_planes_.emplace_back(
            a, b, link_polytope, obstacle,
            internal::FindBodyInTheMiddleOfChain(plant, obstacle->body_index(),
                                                 link_polytope->body_index()),
            plane_order_, plane_decision_vars);
        map_polytopes_to_separating_planes_.emplace(
            SortedPair<ConvexGeometry::Id>(link_polytope->get_id(),
                                           obstacle->get_id()),
            &(separating_planes_[separating_planes_.size() - 1]));
      }
    }
  }
}

std::vector<LinkVertexOnPlaneSideRational>
CspaceFreeRegion::GenerateLinkOnOneSideOfPlaneRationals(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs)
    const {
  std::unordered_map<DirectedKinematicsChain,
                     RationalForwardKinematics::Pose<symbolic::Polynomial>,
                     DirectedKinematicsChainHash>
      body_pair_to_X_AB_multilinear;
  std::vector<LinkVertexOnPlaneSideRational> rationals;
  for (const auto& separating_plane : separating_planes_) {
    if (!IsGeometryPairCollisionIgnored(
            separating_plane.positive_side_polytope->get_id(),
            separating_plane.negative_side_polytope->get_id(),
            filtered_collision_pairs)) {
      // First compute X_AB for both side of the polytopes.
      for (const PlaneSide plane_side :
           {PlaneSide::kPositive, PlaneSide::kNegative}) {
        const ConvexPolytope* polytope;
        const ConvexPolytope* other_side_polytope;
        if (plane_side == PlaneSide::kPositive) {
          polytope = separating_plane.positive_side_polytope;
          other_side_polytope = separating_plane.negative_side_polytope;
        } else {
          polytope = separating_plane.negative_side_polytope;
          other_side_polytope = separating_plane.positive_side_polytope;
        }
        const DirectedKinematicsChain expressed_to_link(
            separating_plane.expressed_link, polytope->body_index());
        auto it = body_pair_to_X_AB_multilinear.find(expressed_to_link);
        if (it == body_pair_to_X_AB_multilinear.end()) {
          body_pair_to_X_AB_multilinear.emplace_hint(
              it, expressed_to_link,
              rational_forward_kinematics_.CalcLinkPoseAsMultilinearPolynomial(
                  q_star, polytope->body_index(),
                  separating_plane.expressed_link));
        }
        it = body_pair_to_X_AB_multilinear.find(expressed_to_link);
        const RationalForwardKinematics::Pose<symbolic::Polynomial>&
            X_AB_multilinear = it->second;
        const std::vector<LinkVertexOnPlaneSideRational>
            rationals_expressed_to_link =
                GenerateLinkOnOneSideOfPlaneRationalFunction(
                    rational_forward_kinematics_, polytope, other_side_polytope,
                    X_AB_multilinear, separating_plane.a, separating_plane.b,
                    plane_side, separating_plane.order);
        // I cannot use "insert" function to append vectors, since
        // LinkVertexOnPlaneSideRational contains const members, hence it does
        // not have an assignment operator.
        std::copy(rationals_expressed_to_link.begin(),
                  rationals_expressed_to_link.end(),
                  std::back_inserter(rationals));
      }
    }
  }
  return rationals;
}

namespace {
// Given t[i], t_lower and t_upper, construct the polynomial t - t_lower and
// t_upper - t.
void ConstructTBoundsPolynomial(
    const std::vector<symbolic::Monomial>& t_monomial,
    const Eigen::Ref<const Eigen::VectorXd>& t_lower,
    const Eigen::Ref<const Eigen::VectorXd>& t_upper,
    VectorX<symbolic::Polynomial>* t_minus_t_lower,
    VectorX<symbolic::Polynomial>* t_upper_minus_t) {
  const symbolic::Monomial monomial_one{};
  t_minus_t_lower->resize(t_monomial.size());
  t_upper_minus_t->resize(t_monomial.size());
  for (int i = 0; i < static_cast<int>(t_monomial.size()); ++i) {
    const symbolic::Polynomial::MapType map_lower{
        {{t_monomial[i], 1}, {monomial_one, -t_lower(i)}}};
    (*t_minus_t_lower)(i) = symbolic::Polynomial(map_lower);
    const symbolic::Polynomial::MapType map_upper{
        {{t_monomial[i], -1}, {monomial_one, t_upper(i)}}};
    (*t_upper_minus_t)(i) = symbolic::Polynomial(map_upper);
  }
}

/**
 * Impose the constraint
 * l_polytope(t) >= 0
 * l_lower(t)>=0
 * l_upper(t)>=0
 * p(t) - l_polytope(t)ᵀ(d - C*t) - l_lower(t)ᵀ(t-t_lower) -
 * l_upper(t)ᵀ(t_upper-t) >=0
 * where l_polytope, l_lower, l_upper are Lagrangian
 * multipliers. p(t) is the numerator of polytope_on_one_side_rational
 * @param monomial_basis The monomial basis for all non-negative polynomials
 * above.
 * @param t_lower_needs_lagrangian If t_lower_needs_lagrangian[i]=false, then
 * lagrangian_lower(i) = 0
 * @param t_upper_needs_lagrangian If t_upper_needs_lagrangian[i]=false, then
 * lagrangian_upper(i) = 0
 * @param[out] lagrangian_polytope l_polytope(t).
 * @param[out] lagrangian_lower l_lower(t).
 * @param[out] lagrangian_upper l_upper(t).
 * @param[out] verified_polynomial p(t) - l_polytope(t)ᵀ(d - C*t) -
 * l_lower(t)ᵀ(t-t_lower) - l_upper(t)ᵀ(t_upper-t)
 */
void AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
    solvers::MathematicalProgram* prog,
    const symbolic::RationalFunction& polytope_on_one_side_rational,
    const VectorX<symbolic::Polynomial>& d_minus_Ct,
    const VectorX<symbolic::Polynomial>& t_minus_t_lower,
    const VectorX<symbolic::Polynomial>& t_upper_minus_t,
    const VectorX<symbolic::Monomial>& monomial_basis,
    const VerificationOption& verification_option,
    const std::vector<bool>& t_lower_needs_lagrangian,
    const std::vector<bool>& t_upper_needs_lagrangian,
    VectorX<symbolic::Polynomial>* lagrangian_polytope,
    VectorX<symbolic::Polynomial>* lagrangian_lower,
    VectorX<symbolic::Polynomial>* lagrangian_upper,
    symbolic::Polynomial* verified_polynomial) {
  lagrangian_polytope->resize(d_minus_Ct.rows());
  lagrangian_lower->resize(t_minus_t_lower.rows());
  lagrangian_upper->resize(t_upper_minus_t.rows());
  *verified_polynomial = polytope_on_one_side_rational.numerator();
  for (int i = 0; i < d_minus_Ct.rows(); ++i) {
    (*lagrangian_polytope)(i) =
        prog->NewNonnegativePolynomial(monomial_basis,
                                       verification_option.lagrangian_type)
            .first;
    *verified_polynomial -= (*lagrangian_polytope)(i)*d_minus_Ct(i);
  }
  for (int i = 0; i < t_minus_t_lower.rows(); ++i) {
    if (t_lower_needs_lagrangian[i]) {
      (*lagrangian_lower)(i) =
          prog->NewNonnegativePolynomial(monomial_basis,
                                         verification_option.lagrangian_type)
              .first;
      *verified_polynomial -= (*lagrangian_lower)(i)*t_minus_t_lower(i);
    } else {
      (*lagrangian_lower)(i) = symbolic::Polynomial();
    }
  }
  for (int i = 0; i < t_upper_minus_t.rows(); ++i) {
    if (t_upper_needs_lagrangian[i]) {
      (*lagrangian_upper)(i) =
          prog->NewNonnegativePolynomial(monomial_basis,
                                         verification_option.lagrangian_type)
              .first;
      *verified_polynomial -= (*lagrangian_upper)(i)*t_upper_minus_t(i);
    } else {
      (*lagrangian_upper)(i) = symbolic::Polynomial();
    }
  }

  const symbolic::Polynomial verified_polynomial_expected =
      prog->NewNonnegativePolynomial(monomial_basis,
                                     verification_option.link_polynomial_type)
          .first;
  const symbolic::Polynomial poly_diff{*verified_polynomial -
                                       verified_polynomial_expected};
  for (const auto& term : poly_diff.monomial_to_coefficient_map()) {
    prog->AddLinearEqualityConstraint(term.second, 0);
  }
}
}  // namespace

CspaceFreeRegion::CspacePolytopeProgramReturn
CspaceFreeRegion::ConstructProgramForCspacePolytope(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const std::vector<LinkVertexOnPlaneSideRational>& rationals,
    const Eigen::Ref<const Eigen::MatrixXd>& C,
    const Eigen::Ref<const Eigen::VectorXd>& d,
    const FilteredCollisionPairs& filtered_collision_pairs,
    const VerificationOption& verification_option) const {
  DRAKE_DEMAND(cspace_region_type_ == CspaceRegionType::kGenericPolytope);
  CspaceFreeRegion::CspacePolytopeProgramReturn ret(rationals.size());
  // Add t as indeterminates
  const auto& t = rational_forward_kinematics_.t();
  ret.prog->AddIndeterminates(t);
  // Add separating planes as decision variables.
  for (const auto& separating_plane : separating_planes_) {
    if (!IsGeometryPairCollisionIgnored(
            separating_plane.positive_side_polytope->get_id(),
            separating_plane.negative_side_polytope->get_id(),
            filtered_collision_pairs)) {
      ret.prog->AddDecisionVariables(separating_plane.decision_variables);
    }
  }
  // Now build the polynomials d(i) - C.row(i) * t
  DRAKE_DEMAND(C.rows() == d.rows() && C.cols() == t.rows());
  VectorX<symbolic::Polynomial> d_minus_Ct_polynomial(C.rows());
  std::vector<symbolic::Monomial> t_monomials;
  t_monomials.reserve(t.rows());
  for (int i = 0; i < t.rows(); ++i) {
    t_monomials.emplace_back(t(i));
  }
  const symbolic::Monomial monomial_one{};
  symbolic::Polynomial::MapType d_minus_Ct_poly_map;
  for (int i = 0; i < C.rows(); ++i) {
    for (int j = 0; j < t.rows(); ++j) {
      auto it = d_minus_Ct_poly_map.find(t_monomials[j]);
      if (it == d_minus_Ct_poly_map.end()) {
        d_minus_Ct_poly_map.emplace_hint(it, t_monomials[j], -C(i, j));
      } else {
        it->second = -C(i, j);
      }
    }
    auto it = d_minus_Ct_poly_map.find(monomial_one);
    if (it == d_minus_Ct_poly_map.end()) {
      d_minus_Ct_poly_map.emplace_hint(it, monomial_one, d(i));
    } else {
      it->second = d(i);
    }
    d_minus_Ct_polynomial(i) = symbolic::Polynomial(d_minus_Ct_poly_map);
  }

  // Build the polynomial for t-t_lower and t_upper-t
  Eigen::VectorXd t_lower, t_upper;
  ComputeBoundsOnT(
      q_star, rational_forward_kinematics_.plant().GetPositionLowerLimits(),
      rational_forward_kinematics_.plant().GetPositionUpperLimits(), &t_lower,
      &t_upper);
  // If C * t <= d already implies t(i) <= t_upper(i) or t(i) >= t_lower(i) for
  // some t, then we don't need to add the lagrangian multiplier for that
  // t_upper(i) - t(i) or t(i) - t_lower(i).
  Eigen::VectorXd t_lower_from_polytope, t_upper_from_polytope;
  BoundPolyhedronByBox(C, d, &t_lower_from_polytope, &t_upper_from_polytope);
  std::vector<bool> t_lower_needs_lagrangian(t.rows(), true);
  std::vector<bool> t_upper_needs_lagrangian(t.rows(), true);
  for (int i = 0; i < t.rows(); ++i) {
    if (t_lower(i) < t_lower_from_polytope(i)) {
      t_lower_needs_lagrangian[i] = false;
    }
    if (t_upper(i) > t_upper_from_polytope(i)) {
      t_upper_needs_lagrangian[i] = false;
    }
  }
  VectorX<symbolic::Polynomial> t_minus_t_lower_poly(t.rows());
  VectorX<symbolic::Polynomial> t_upper_minus_t_poly(t.rows());
  ConstructTBoundsPolynomial(t_monomials, t_lower, t_upper,
                             &t_minus_t_lower_poly, &t_upper_minus_t_poly);

  // Get the monomial basis for each kinematics chain.
  std::unordered_map<SortedPair<multibody::BodyIndex>,
                     VectorX<symbolic::Monomial>>
      map_chain_to_monomial_basis;
  for (int i = 0; i < static_cast<int>(rationals.size()); ++i) {
    VectorX<symbolic::Monomial> monomial_basis_chain;
    FindMonomialBasisForPolytopicRegion(
        rational_forward_kinematics_, rationals[i],
        &map_chain_to_monomial_basis, &monomial_basis_chain);
    // Now add the constraint that C*t<=d and t_lower <= t <= t_upper implies
    // the rational being nonnegative.
    AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
        ret.prog.get(), rationals[i].rational, d_minus_Ct_polynomial,
        t_minus_t_lower_poly, t_upper_minus_t_poly, monomial_basis_chain,
        verification_option, t_lower_needs_lagrangian, t_upper_needs_lagrangian,
        &(ret.polytope_lagrangians[i]), &(ret.t_lower_lagrangians[i]),
        &(ret.t_upper_lagrangians[i]), &(ret.verified_polynomials[i]));
  }
  return ret;
}

bool CspaceFreeRegion::IsPostureInCollision(
    const systems::Context<double>& context) const {
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

std::vector<LinkVertexOnPlaneSideRational>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    const ConvexPolytope* polytope, const ConvexPolytope* other_side_polytope,
    const RationalForwardKinematics::Pose<symbolic::Polynomial>&
        X_AB_multilinear,
    const drake::Vector3<symbolic::Expression>& a_A,
    const symbolic::Expression& b, PlaneSide plane_side,
    SeparatingPlaneOrder plane_order) {
  std::vector<LinkVertexOnPlaneSideRational> rational_fun;
  rational_fun.reserve(polytope->p_BV().cols());
  const symbolic::Monomial monomial_one{};
  // a_A and b are not polynomial of sinθ or cosθ.
  Vector3<symbolic::Polynomial> a_A_poly;
  for (int i = 0; i < 3; ++i) {
    a_A_poly(i) = symbolic::Polynomial({{monomial_one, a_A(i)}});
  }
  const symbolic::Polynomial b_poly({{monomial_one, b}});
  for (int i = 0; i < polytope->p_BV().cols(); ++i) {
    // Step 1: Compute vertex position.
    const Vector3<drake::symbolic::Polynomial> p_AVi =
        X_AB_multilinear.p_AB + X_AB_multilinear.R_AB * polytope->p_BV().col(i);

    // Step 2: Compute a_A.dot(p_AVi) + b
    const drake::symbolic::Polynomial point_on_hyperplane_side =
        a_A_poly.dot(p_AVi) + b_poly;

    // Step 3: Convert the multilinear polynomial to rational function.
    rational_fun.emplace_back(
        rational_forward_kinematics
            .ConvertMultilinearPolynomialToRationalFunction(
                plane_side == PlaneSide::kPositive
                    ? point_on_hyperplane_side - 1
                    : -1 - point_on_hyperplane_side),
        polytope, X_AB_multilinear.frame_A_index, other_side_polytope, a_A, b,
        plane_side, plane_order);
  }
  return rational_fun;
}

bool IsGeometryPairCollisionIgnored(
    const SortedPair<ConvexGeometry::Id>& geometry_pair,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs) {
  return filtered_collision_pairs.count(geometry_pair) > 0;
}

bool IsGeometryPairCollisionIgnored(
    ConvexGeometry::Id id1, ConvexGeometry::Id id2,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs) {
  return IsGeometryPairCollisionIgnored(
      drake::SortedPair<ConvexGeometry::Id>(id1, id2),
      filtered_collision_pairs);
}

void ComputeBoundsOnT(const Eigen::Ref<const Eigen::VectorXd>& q_star,
                      const Eigen::Ref<const Eigen::VectorXd>& q_lower,
                      const Eigen::Ref<const Eigen::VectorXd>& q_upper,
                      Eigen::VectorXd* t_lower, Eigen::VectorXd* t_upper) {
  DRAKE_DEMAND((q_upper.array() >= q_lower.array()).all());
  // Currently I require that q_upper - q_star < pi and q_star - q_lower > -pi.
  DRAKE_DEMAND(((q_upper - q_star).array() < M_PI).all());
  DRAKE_DEMAND(((q_star - q_lower).array() > -M_PI).all());
  *t_lower = ((q_lower - q_star) / 2).array().tan();
  *t_upper = ((q_upper - q_star) / 2).array().tan();
}

}  // namespace multibody
}  // namespace drake
