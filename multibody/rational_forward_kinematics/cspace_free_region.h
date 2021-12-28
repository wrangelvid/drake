#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/multibody/rational_forward_kinematics/convex_geometry.h"
#include "drake/multibody/rational_forward_kinematics/plane_side.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"

/**
 * This file is largely the same as configuration_space_collision_free_region.h.
 * The major differences are
 * 1. The separating hyperplane is parameterized as aᵀx + b ≥ 1 and aᵀx+b ≤ −1
 * 2. We first focus on the generic polytopic region C*t<=d in the configuration
 * space (we will add the special case for axis-aligned bounding box region
 * t_lower <= t <= t_upper later).
 */

namespace drake {
namespace multibody {
/* The separating plane aᵀx + b ≥ 1, aᵀx+b ≤ −1 has parameters a and b. These
 * parameters can be a constant of affine function of t.
 */
enum class SeparatingPlaneOrder {
  kConstant,
  kAffine,
};

/**
 * One polytope is on the "positive" side of the separating plane, namely {x|
 * aᵀx + b ≥ 1}, and the other polytope is on the "negative" side of the
 * separating plane, namely {x|aᵀx+b ≤ −1}.
 */
struct SeparatingPlane {
  SeparatingPlane(
      drake::Vector3<symbolic::Expression> m_a, symbolic::Expression m_b,
      const ConvexPolytope* m_positive_side_polytope,
      const ConvexPolytope* m_negative_side_polytope,
      multibody::BodyIndex m_expressed_link, SeparatingPlaneOrder m_order,
      const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>>&
          m_decision_variables)
      : a{std::move(m_a)},
        b{std::move(m_b)},
        positive_side_polytope{m_positive_side_polytope},
        negative_side_polytope{m_negative_side_polytope},
        expressed_link{std::move(m_expressed_link)},
        order{m_order},
        decision_variables{m_decision_variables} {}

  const Vector3<symbolic::Expression> a;
  const symbolic::Expression b;
  const ConvexPolytope* const positive_side_polytope;
  const ConvexPolytope* const negative_side_polytope;
  const multibody::BodyIndex expressed_link;
  const SeparatingPlaneOrder order;
  const VectorX<symbolic::Variable> decision_variables;
};

/**
 * We need to verify that C * t <= d implies p(t) >= 0, where p(t) is the
 * numerator of the rational function aᵀx + b - 1 or -1 - aᵀx-b. Namely we need
 * to verify the non-negativity of the lagrangian polynomial l(t), together with
 * p(t) - l(t)ᵀ(d - C * t). We can choose the type of the non-negative
 * polynomials (sos, dsos, sdsos).
 */
struct VerificationOption {
  solvers::MathematicalProgram::NonnegativePolynomial link_polynomial_type;
  solvers::MathematicalProgram::NonnegativePolynomial lagrangian_type;
};

/**
 * The rational function representing that a link vertex V is on the desired
 * side of the plane. If the link is on the positive side of the plane, then the
 * rational is aᵀx + b - 1, otherwise it is -1 - aᵀx - b
 */
struct LinkVertexOnPlaneSideRational {
  LinkVertexOnPlaneSideRational(
      symbolic::RationalFunction m_rational,
      const ConvexPolytope* m_link_polytope,
      multibody::BodyIndex m_expressed_body_index,
      const ConvexPolytope* m_other_side_link_polytope,
      Vector3<symbolic::Expression> m_a_A, symbolic::Expression m_b,
      PlaneSide m_plane_side, SeparatingPlaneOrder m_plane_order)
      : rational{std::move(m_rational)},
        link_polytope{m_link_polytope},
        expressed_body_index{m_expressed_body_index},
        other_side_link_polytope{m_other_side_link_polytope},
        a_A{std::move(m_a_A)},
        b{std::move(m_b)},
        plane_side{m_plane_side},
        plane_order{m_plane_order} {}
  const symbolic::RationalFunction rational;
  const ConvexPolytope* const link_polytope;
  const multibody::BodyIndex expressed_body_index;
  const ConvexPolytope* const other_side_link_polytope;
  const Vector3<symbolic::Expression> a_A;
  const symbolic::Expression b;
  const PlaneSide plane_side;
  const SeparatingPlaneOrder plane_order;
};

enum class CspaceRegionType { kGenericPolytope, kAxisAlignedBoundingBox };

/**
 * This class tries to find a large convex set in the configuration space, such
 * that this whole convex set is collision free. We assume that the obstacles
 * are unions of polytopes in the workspace, and the robot link poses
 * (position/orientation) can be written as rational functions of some
 * variables. Such robot can have only revolute (or prismatic joint). We also
 * suppose that the each link of the robot is represented as a union of
 * polytopes. We will find the convex collision free set in the configuration
 * space through convex optimization.
 */
class CspaceFreeRegion {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CspaceFreeRegion)

  using FilteredCollisionPairs =
      std::unordered_set<drake::SortedPair<ConvexGeometry::Id>>;

  CspaceFreeRegion(const multibody::MultibodyPlant<double>& plant,
                   const std::vector<const ConvexPolytope*>& link_polytopes,
                   const std::vector<const ConvexPolytope*>& obstacles,
                   SeparatingPlaneOrder plane_order,
                   CspaceRegionType cspace_region_type);

  const std::unordered_map<SortedPair<ConvexGeometry::Id>,
                           const SeparatingPlane*>&
  map_polytopes_to_separating_planes() const {
    return map_polytopes_to_separating_planes_;
  }

  /**
   * Generate all the rational functions in the form aᵀx + b -1 or -1-aᵀx-b
   * whose non-negativity implies that the separating plane aᵀx + b =0 separates
   * a pair of polytopes.
   * This function loops over all pair of polytopes between a link and an
   * obstacle that are not in filtered_collision_pair.
   */
  // TODO(hongkai.dai): also consider the self-collision pairs.
  std::vector<LinkVertexOnPlaneSideRational>
  GenerateLinkOnOneSideOfPlaneRationals(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs)
      const;

  /**
   * This struct is the return type of ConstructProgramForCspacePolytope, to
   * verify the C-space region C * t <= d is collision free.
   */
  struct CspacePolytopeProgramReturn {
    CspacePolytopeProgramReturn(size_t rationals_size)
        : prog{new solvers::MathematicalProgram()},
          polytope_lagrangians{rationals_size},
          t_lower_lagrangians{rationals_size},
          t_upper_lagrangians{rationals_size},
          verified_polynomials{rationals_size} {}

    std::unique_ptr<solvers::MathematicalProgram> prog;
    // polytope_lagrangians has size rationals.size(), namely it is the number
    // of (link_polytope, obstacle_polytope) pairs. lagrangians[i] has size
    // C.rows()
    std::vector<VectorX<symbolic::Polynomial>> polytope_lagrangians;
    // t_lower_lagrangians[i][j] is the lagrangian for t(j) >= t_lower(j) to
    // verify rationals[i]>= 0.
    std::vector<VectorX<symbolic::Polynomial>> t_lower_lagrangians;
    // t_upper_lagrangians[i][j] is the lagrangian for t(j) <= t_lower(j) to
    // verify rationals[i]>= 0.
    std::vector<VectorX<symbolic::Polynomial>> t_upper_lagrangians;
    // verified_polynomial[i] is p(t) - l_polytope(t)ᵀ(d - C*t) -
    // l_lower(t)ᵀ(t-t_lower) - l_upper(t)ᵀ(t_upper-t)
    std::vector<symbolic::Polynomial> verified_polynomials;
  };

  CspacePolytopeProgramReturn ConstructProgramForCspacePolytope(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const std::vector<LinkVertexOnPlaneSideRational>& rationals,
      const Eigen::Ref<const Eigen::MatrixXd>& C,
      const Eigen::Ref<const Eigen::VectorXd>& d,
      const FilteredCollisionPairs& filtered_collision_pairs,
      const VerificationOption& verification_option = {}) const;

  bool IsPostureInCollision(const systems::Context<double>& context) const;

  const RationalForwardKinematics& rational_forward_kinematics() const {
    return rational_forward_kinematics_;
  }

  const std::vector<SeparatingPlane>& separating_planes() const {
    return separating_planes_;
  }

  const std::map<multibody::BodyIndex, std::vector<const ConvexPolytope*>>&
  link_polytopes() const {
    return link_polytopes_;
  }

  // obstacles_[i] is the i'th polytope, fixed to the world.
  const std::vector<const ConvexPolytope*>& obstacles() const {
    return obstacles_;
  }

 private:
  RationalForwardKinematics rational_forward_kinematics_;
  std::map<multibody::BodyIndex, std::vector<const ConvexPolytope*>>
      link_polytopes_;
  std::vector<const ConvexPolytope*> obstacles_;

  SeparatingPlaneOrder plane_order_;
  CspaceRegionType cspace_region_type_;
  std::vector<SeparatingPlane> separating_planes_;

  // In the key, the first ConvexGeometry::Id is for the polytope on the
  // positive side, the second ConvexGeometry::Id is for the one on the negative
  // side.
  std::unordered_map<SortedPair<ConvexGeometry::Id>, const SeparatingPlane*>
      map_polytopes_to_separating_planes_;
};

/**
 * Generate the rational functions a_A.dot(p_AVi(t)) + b(i) - 1 or -1 -
 * a_A.dot(p_AVi(t)) - b(i). Which represents that the link (whose vertex Vi has
 * position p_AVi in the frame A) is on the positive (or negative) side of the
 * plane a_A * x + b = 0
 * @param X_AB_multilinear The pose of the link frame B in the expressed body
 * frame A. Note that this pose is a multilinear function of sinθ and cosθ.
 */
std::vector<LinkVertexOnPlaneSideRational>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    const ConvexPolytope* polytope, const ConvexPolytope* other_side_polytope,
    const RationalForwardKinematics::Pose<symbolic::Polynomial>&
        X_AB_multilinear,
    const drake::Vector3<symbolic::Expression>& a_A,
    const symbolic::Expression& b, PlaneSide plane_side,
    SeparatingPlaneOrder plane_order);

bool IsGeometryPairCollisionIgnored(
    ConvexGeometry::Id id1, ConvexGeometry::Id id2,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs);

bool IsGeometryPairCollisionIgnored(
    const SortedPair<ConvexGeometry::Id>& geometry_pair,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs);

void ComputeBoundsOnT(const Eigen::Ref<const Eigen::VectorXd>& q_star,
                      const Eigen::Ref<const Eigen::VectorXd>& q_upper,
                      const Eigen::Ref<const Eigen::VectorXd>& q_lower,
                      Eigen::VectorXd* t_lower, Eigen::VectorXd* t_upper);
}  // namespace multibody
}  // namespace drake
