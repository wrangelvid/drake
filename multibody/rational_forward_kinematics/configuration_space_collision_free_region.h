#pragma once

#include <map>
#include <memory>
#include <optional>
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

namespace drake {
namespace multibody {
/**
 * The separating plane aᵀ(x-c) = 1's normal vector a can be a constant or an
 * affine function of t.
 */
enum class SeparatingPlaneOrder {
  kConstant,
  kAffine,
};

/**
 * A separation plane {x | aᵀ(x-c) = 1} separates two polytopes.
 * c is the center of the negative_side_polytope.
 * All the vertices in the positive_side_polytope satisfies aᵀ(v-c) ≥ 1
 * All the vertices in the negative_side_polytope satisfies aᵀ(v-c) ≤ 1
 */
struct SeparationPlane {
  SeparationPlane(
      const drake::Vector3<drake::symbolic::Expression>& m_a,
      std::shared_ptr<const ConvexPolytope> m_positive_side_polytope,
      std::shared_ptr<const ConvexPolytope> m_negative_side_polytope,
      drake::multibody::BodyIndex m_expressed_link,
      SeparatingPlaneOrder m_a_order,
      const Eigen::Ref<const drake::VectorX<drake::symbolic::Variable>>&
          m_decision_variables)
      : a{m_a},
        positive_side_polytope{m_positive_side_polytope},
        negative_side_polytope{m_negative_side_polytope},
        expressed_link{m_expressed_link},
        a_order{m_a_order},
        decision_variables{m_decision_variables} {}
  const drake::Vector3<drake::symbolic::Expression> a;
  std::shared_ptr<const ConvexPolytope> positive_side_polytope;
  std::shared_ptr<const ConvexPolytope> negative_side_polytope;
  // The link frame in which a is expressed.
  const drake::multibody::BodyIndex expressed_link;
  const SeparatingPlaneOrder a_order;
  const drake::VectorX<drake::symbolic::Variable> decision_variables;
};

/** We need to verify that these polynomials are non-negative
 * Lagrangians l_lower(t) >= 0, l_upper(t) >= 0
 * Link polynomial p(t) - l_lower(t) * (t - t_lower) - l_upper(t)(t_upper - t)
 * >= 0
 */
struct VerificationOption {
  VerificationOption()
      : link_polynomial_type{drake::solvers::MathematicalProgram::
                                 NonnegativePolynomial::kSos},
        lagrangian_type{
            drake::solvers::MathematicalProgram::NonnegativePolynomial::kSos} {}
  drake::solvers::MathematicalProgram::NonnegativePolynomial
      link_polynomial_type;
  drake::solvers::MathematicalProgram::NonnegativePolynomial lagrangian_type;
};

/**
 * The rational function representing that a link vertex V is on the desired
 * side of the plane.
 * If the link is on the positive side of the plane, then the rational is
 * aᵀ(x-c)- 1; otherwise it is 1 - aᵀ(x-c).
 */
struct LinkVertexOnPlaneSideRational {
  LinkVertexOnPlaneSideRational(
      drake::symbolic::RationalFunction m_rational,
      std::shared_ptr<const ConvexPolytope> m_link_polytope,
      drake::multibody::BodyIndex m_expressed_body_index,
      std::shared_ptr<const ConvexPolytope> m_other_side_link_polytope,
      const drake::Vector3<drake::symbolic::Expression>& m_a_A,
      PlaneSide m_plane_side, SeparatingPlaneOrder m_a_order)
      : rational(std::move(m_rational)),
        link_polytope(m_link_polytope),
        expressed_body_index(m_expressed_body_index),
        other_side_link_polytope(m_other_side_link_polytope),
        a_A(m_a_A),
        plane_side(m_plane_side),
        a_order{m_a_order} {}
  const drake::symbolic::RationalFunction rational;
  const std::shared_ptr<const ConvexPolytope> link_polytope;
  const drake::multibody::BodyIndex expressed_body_index;
  const std::shared_ptr<const ConvexPolytope> other_side_link_polytope;
  const drake::Vector3<drake::symbolic::Expression> a_A;
  const PlaneSide plane_side;
  const SeparatingPlaneOrder a_order;
};

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
class ConfigurationSpaceCollisionFreeRegion {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ConfigurationSpaceCollisionFreeRegion)

  struct GeometryIdPairHash {
    std::size_t operator()(
        const std::pair<ConvexGeometry::Id, ConvexGeometry::Id>& p) const {
      return std::hash<ConvexGeometry::Id>()(p.first) +
             std::hash<ConvexGeometry::Id>()(p.second);
      // return std::hash<ConvexGeometry::Id>()(p.first * 100) +
      //       std::hash<ConvexGeometry::Id>()(p.second);
    }
  };

  using FilteredCollisionPairs =
      std::unordered_set<drake::SortedPair<ConvexGeometry::Id>>;

  struct FilteredCollisionPairsForBox {
    FilteredCollisionPairsForBox(
        const Eigen::Ref<const Eigen::VectorXd>& m_q_lower,
        const Eigen::Ref<const Eigen::VectorXd>& m_q_upper,
        FilteredCollisionPairs m_filtered_collision_pairs)
        : q_lower(m_q_lower),
          q_upper(m_q_upper),
          filtered_collision_pairs(m_filtered_collision_pairs) {}

    void WriteToFile(const std::string& file_name) const;

    Eigen::VectorXd q_lower;
    Eigen::VectorXd q_upper;
    FilteredCollisionPairs filtered_collision_pairs;
  };

  /**
   * Verify the collision free configuration space for the given robot. The
   * geometry of each robot link is represented as a union of polytopes. The
   * obstacles are also a union ob polytopes.
   */
  ConfigurationSpaceCollisionFreeRegion(
      const drake::multibody::MultibodyPlant<double>& plant,
      const std::vector<std::shared_ptr<const ConvexPolytope>>& link_polytopes,
      const std::vector<std::shared_ptr<const ConvexPolytope>>& obstacles,
      SeparatingPlaneOrder a_order);

  const RationalForwardKinematics& rational_forward_kinematics() const {
    return rational_forward_kinematics_;
  }

  const std::vector<SeparationPlane>& separation_planes() const {
    return separation_planes_;
  }

  /**
   * Generate all the rational functions representing the the link vertices are
   * on the correct side of the planes.
   * @param q_not_in_collision is used to compute the position of negative side
   * polytope center. Namely in the separating hyperplane aᵀ(x−c)=1, the value c
   * is evaluated at this posture, that the negative side of the separating
   * hyperplane will always contain the polytope center at this posture. Set to
   * std::nullopt and we will use q_star as q_not_in_collision.
   */
  std::vector<LinkVertexOnPlaneSideRational>
  GenerateLinkOnOneSideOfPlaneRationals(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs,
      const std::optional<Eigen::VectorXd>& q_not_in_collision =
          std::nullopt) const;

  const std::unordered_map<std::pair<ConvexGeometry::Id, ConvexGeometry::Id>,
                           const SeparationPlane*, GeometryIdPairHash>&
  map_polytopes_to_separation_planes() const {
    return map_polytopes_to_separation_planes_;
  }

  /**
   * This struct is the return type of
   * ConstructProgramToVerifyCollisionFreePolytope, to verify the region C * t
   * <= d is collision free.
   */
  struct ConstructProgramReturn {
    explicit ConstructProgramReturn(size_t rationals_size)
        : prog{new solvers::MathematicalProgram()},
          lagrangians{rationals_size},
          verified_polynomials{rationals_size} {}
    std::unique_ptr<solvers::MathematicalProgram> prog;
    // lagrangians has size rationals.size(), namely it is the number of
    // (link_polytope, obstacle_polytope) pairs. lagrangians[i] has size
    // C.rows()
    std::vector<VectorX<symbolic::Polynomial>> lagrangians;
    // verified_polynomial[i] is p(t) - l(t)ᵀ(d - C*t)
    std::vector<symbolic::Polynomial> verified_polynomials;
  };

  /**
   * Constructs an optimization program to verify that the polytopic region C *
   * t <= d is collision free. The program verifies that C * t <= d implies
   * `rationals[i]` >= 0 for all i, where rationals[i] is the result of calling
   * GenerateLinkOnOneSideOfPlaneRationals.
   */
  ConstructProgramReturn ConstructProgramToVerifyCollisionFreePolytope(
      const std::vector<LinkVertexOnPlaneSideRational>& rationals,
      const Eigen::Ref<const Eigen::MatrixXd>& C,
      const Eigen::Ref<const Eigen::VectorXd>& d,
      const FilteredCollisionPairs& filtered_collision_pairs,
      const VerificationOption& verification_option = {}) const;

  /**
   * Construct an optimization program to verify the the box region t_lower <= t
   * <= t_upper is collision free.
   * The program verifies that t_lower <= t <= t_upper implies @p polynomials[i]
   * >= 0 for all i, where rationals[i] is the result of calling
   * GenerateLinkOnOneSideOfPlaneRationals.
   */
  std::unique_ptr<drake::solvers::MathematicalProgram>
  ConstructProgramToVerifyCollisionFreeBox(
      const std::vector<LinkVertexOnPlaneSideRational>& rationals,
      const Eigen::Ref<const Eigen::VectorXd>& t_lower,
      const Eigen::Ref<const Eigen::VectorXd>& t_upper,
      const FilteredCollisionPairs& filtered_collision_pairs,
      const VerificationOption& verification_option = {}) const;

  struct BinarySearchOption : public VerificationOption {
    explicit BinarySearchOption(
        const Eigen::Ref<const Eigen::VectorXd>& m_delta_q_tol)
        : VerificationOption(), delta_q_tol{m_delta_q_tol} {}
    Eigen::VectorXd delta_q_tol;
  };

  /**
   * Find the largest box in the configuration space, that we can verify to be
   * collision free. The box is defined as
   * max(q_lower, q + ρ * Δ q_) <= q <= min(q_upper, q + ρ * Δ q₊)
   * where q_lower, q_upper are the joint limits.
   * @param q_star The nominal configuration around which we verify the
   * collision free box. This nominal configuration should be collision free.
   * @param filtered_collision_pairs The set of polytope pairs between which the
   * collision check is ignored.
   * @param negative_delta_q Δq₋ in the documentation above.
   * @param positive_delta_q Δq₊ in the documentation above.
   * @param rho_lower_initial The initial guess on the lower bound of ρ. The box
   * defined with rho_lower_initial is collision free.
   * @param rho_upper_initial The initial guess on the upper bound of ρ. The box
   * defined with rho_upper_initial is not collision free.
   */
  double FindLargestBoxThroughBinarySearch(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs,
      const Eigen::Ref<const Eigen::VectorXd>& negative_delta_q,
      const Eigen::Ref<const Eigen::VectorXd>& positive_delta_q,
      double rho_lower_initial, double rho_upper_initial,
      const std::vector<FilteredCollisionPairsForBox>&
          filtered_collision_pairs_for_boxes,
      const BinarySearchOption& option) const;

  struct SearchLagrangianOption : public VerificationOption {
    SearchLagrangianOption() : VerificationOption(), maximize_residue{false} {}

    SearchLagrangianOption(const VerificationOption& option,
                           bool maximize_residue_in)
        : VerificationOption(option), maximize_residue{maximize_residue_in} {}

    bool maximize_residue;
  };

  enum class BilinearAlternationCost {
    kBoxEdgeLengthSum,  ///< Maximize the sum of the edge length on the box of
                        ///< t. Oftentimes I find this cost gives me bad box.
    kVolume,  ///< Maximize the volume of the box of t. This is the preferred
              ///< cost form, but it requires the solver to support exponential
              ///< cone constraint.
  };

  struct BilinearAlternationOption : public SearchLagrangianOption {
    BilinearAlternationOption()
        : SearchLagrangianOption(),
          max_iteration{10},
          optimal_threshold{0.01},
          box_offcenter_ratio{1},
          grow_all_dimension{false},
          cost_type{BilinearAlternationCost::kVolume} {}
    int max_iteration{10};
    double optimal_threshold{0.01};
    // A ratio between [0, 1]. If ratio=1, it means the box is centered at
    // q_star. If ratio = 0, it means that q_star could be a corner of the box.
    double box_offcenter_ratio{1};
    // If grow_all_dimension = true, then it is guaranteed that after bilinear
    // alternation, the optimized box is larger than the initial box along all
    // dimensions.
    bool grow_all_dimension{true};

    BilinearAlternationCost cost_type;
  };

  /**
   * Find the largest box {t | t_lower <= t <= t_upper} around t = 0 (where
   * t = tan(Δ q / 2), such that the box is collision free.
   * We solve the following optimization problem
   * max vol(box)
   * s.t p(t) - l_lower(t)*(t - t_lower) - l_upper(t)*(t_upper - t) >= 0,
   *     l_lower(t), l_upper(t) >= 0,
   *     t_lower <= 0, t_upper >= 0
   * through bilinear alternation.
   * @param q_star The nominal posture around which the box is computed.
   * @param filtered_collision_pairs The set of polytope pairs between which the
   * collision check is ignored.
   * @param t_lower_init The initial lower bound of the box t_lower <= t <=
   * t_upper that we know is collision free.
   * @param t_upper_init The initial upper bound of the box t_lower <= t <=
   * t_upper that we know is collision free.
   * @param t_lower_sol t_lower_sol <= t <= t_upper_sol defines the final result
   * of the collision free box.
   * @param t_upper_sol t_lower_sol <= t <= t_upper_sol defines the final result
   * of the collision free box.
   */
  drake::solvers::MathematicalProgramResult
  FindLargestBoxThroughBilinearAlternation(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs,
      const BilinearAlternationOption& options,
      const Eigen::Ref<const Eigen::VectorXd>& t_lower_init,
      const Eigen::Ref<const Eigen::VectorXd>& t_upper_init,
      Eigen::VectorXd* t_lower_sol, Eigen::VectorXd* t_upper_sol,
      std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>* q_box_sols)
      const;

  struct FindLargestBoxOption : public BilinearAlternationOption {
    explicit FindLargestBoxOption(
        const Eigen::Ref<const Eigen::VectorXd>& m_delta_q_tol)
        : BilinearAlternationOption(), delta_q_tol{m_delta_q_tol} {}

    BinarySearchOption binary_search_option() const {
      BinarySearchOption options(delta_q_tol);
      options.lagrangian_type = lagrangian_type;
      options.link_polynomial_type = link_polynomial_type;
      return options;
    }

    Eigen::VectorXd delta_q_tol;
  };
  /**
   * Find the largest collision free box around q_star.
   * We first run bilinear alternation to find the shape of the box, and then
   * run binary search to enlarge the scale of the box.
   * @return is_success True if we successfully find the box. False otherwise.
   * A possible failure might be the initial box (negative_delta_q_init,
   * positive_delta_q_init) is not collision free.
   */
  bool FindLargestBox(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairsForBox& filtered_collision_pairs_base,
      const std::vector<FilteredCollisionPairsForBox>&
          filtered_collision_pairs_for_boxes,
      const Eigen::Ref<const Eigen::VectorXd>& negative_delta_q_init,
      const Eigen::Ref<const Eigen::VectorXd>& positive_delta_q_init,
      std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>* q_box_sols,
      const FindLargestBoxOption& option) const;

  /**
   * For a given posture (stored in @param context), return if the robot link
   * is in collision with the world obstacle or not.
   */
  bool IsPostureCollisionFree(
      const drake::systems::Context<double>& context) const;

  const std::map<drake::multibody::BodyIndex,
                 std::vector<std::shared_ptr<const ConvexPolytope>>>&
  link_polytopes() const {
    return link_polytopes_;
  }

  // obstacles_[i] is the i'th polytope, fixed to the world.
  const std::vector<std::shared_ptr<const ConvexPolytope>>& obstacles() const {
    return obstacles_;
  }

  /**
   * For a box in the configuration space, find the pair of geometries that can
   * be proved collision free. Note that we only return the pairs that are not
   * in @p existing_filtered_collision_pairs.
   */
  FilteredCollisionPairsForBox FindFilteredCollisionPairsForBox(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const Eigen::Ref<const Eigen::VectorXd>& q_lower,
      const Eigen::Ref<const Eigen::VectorXd>& q_upper,
      const FilteredCollisionPairs& existing_filtered_collision_pairs) const;

 private:
  struct BoxVerificationTuple {
    drake::symbolic::Polynomial p;
    std::vector<drake::symbolic::Polynomial> lagrangians;
    std::vector<drake::MatrixX<drake::symbolic::Variable>> lagrangian_gramians;
    std::vector<drake::symbolic::Polynomial> constraints;
    std::vector<drake::symbolic::Variable> t_bound_vars;
    // constraints are t_lower(i) <= t(i) <= t_upper(i) for each t in t_chain.
    drake::VectorX<drake::symbolic::Variable> t_chain;
    drake::VectorX<drake::symbolic::Monomial> lagrangian_monomial_basis;
    drake::SortedPair<ConvexGeometry::Id> geometry_pair;
  };

  /**
   * In bilinear alternation, we want to impose the following implication
   * constraint(t) >= 0 => p(t)>= 0
   * hence we introduce the lagrangian multiplier lagrangian(t), such that
   * p(t) - lagrangian(t) * constraint(t) >= 0
   * l(t) >= 0
   * This function generate all the tuples (p(t), lagrangian(t), constraint(t))
   * such that the box region is collision free, if the p(t), lagrangian(t)
   * satisfy the non-negativity constraints.
   * @param[out] bilinear_alternation_tuples The tuples of (p(t), lagrangian(t),
   * constraint(t)).
   * @param[out] separation_plane_variables. The variables created for all
   * separation planes.
   * @param[out] t_lower The parameterization of the box.
   * @param[out] t_upper The parameterization of the box.
   */
  void GenerateVerificationConstraintForBilinearAlternation(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs,
      std::vector<BoxVerificationTuple>* bilinear_alternation_tuples,
      drake::VectorX<drake::symbolic::Variable>* t_lower,
      drake::VectorX<drake::symbolic::Variable>* t_upper) const;

  /**
   * This is the "lagrangian step" in the bilinear alternation. We are given the
   * constraints for the box as constraint(t) >= 0, we search for the lagrangian
   * multiplier lagrangian(t) through the following program
   * max r
   * s.t p(t) - r - lagrangianᵀ(t) * constraint(t) >= 0
   * lagrangian(t) >= 0
   * The decision variables are the separating plane, the residue r, and the
   * lagrangian multipliers.
   */
  drake::solvers::MathematicalProgramResult SearchLagrangianMultiplier(
      const std::vector<BoxVerificationTuple>& bilinear_alternation_tuples,
      const drake::VectorX<drake::symbolic::Variable>& t_lower_vars,
      const drake::VectorX<drake::symbolic::Variable>& t_upper_vars,
      const Eigen::VectorXd& t_lower_sol, const Eigen::VectorXd& t_upper_sol,
      const SearchLagrangianOption& option) const;

  /**
   * This is the "box step" in the bilinear alternation.
   * We are given the Lagrangian multipliers lagrangians_sol, and we solve the
   * following problem to find the maximal box given
   * max vol(box)
   * s.t p(t) - lagrangians_solᵀ(t) * constraint(t) >= 0
   *     t_lower <= 0, t_upper >= 0
   * The decision variables are the separating plane, the box parameter t_lower,
   * t_upper.
   * @param t_lower_joint_limit. Due to the joint limits q_lower <= q <=
   * q_upper, t is also bounded (t = tan((q - q_star)/2)).
   * @param t_upper_joint_limit. Due to the joint limits q_lower <= q <=
   * q_upper, t is also bounded (t = tan((q - q_star)/2)).
   * @param lagrangians_sol The solution to the Lagrangians.
   * @param t_lower_init The initial guess of t_lower. This is used to add a
   * constraint on t_lower, if we require that the box to grow on all
   * dimensions.
   * @param t_upper_init The initial guess of t_upper. This is used to add a
   * constraint on t_upper, if we require that the box to grow on all
   * dimensions.
   */
  drake::solvers::MathematicalProgramResult SearchBoxGivenLagrangian(
      const std::vector<BoxVerificationTuple>& bilinear_alternation_tuples,
      const drake::VectorX<drake::symbolic::Variable>& t_lower_vars,
      const drake::VectorX<drake::symbolic::Variable>& t_upper_vars,
      const Eigen::VectorXd& t_lower_joint_limit,
      const Eigen::VectorXd& t_upper_joint_limit,
      const std::vector<std::vector<drake::symbolic::Polynomial>>&
          lagrangians_sol,
      const Eigen::Ref<const Eigen::VectorXd>& t_lower_init,
      const Eigen::Ref<const Eigen::VectorXd>& t_upper_init,
      const BilinearAlternationOption& option) const;

  double FindLargestBoxThroughBinarySearch(
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
      const BinarySearchOption& option) const;

  drake::solvers::MathematicalProgramResult
  FindLargestBoxThroughBilinearAlternation(
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
      const;

  bool IsLinkPairCollisionIgnored(
      ConvexGeometry::Id id1, ConvexGeometry::Id id2,
      const FilteredCollisionPairs& filtered_collision_pairs) const;

  /**
   * Since t = tan((q - q_star) / 2), the joint limit bounds on q implies bounds
   * on t.
   */
  void ComputeBoundsOnT(const Eigen::Ref<const Eigen::VectorXd>& q_star,
                        Eigen::VectorXd* t_lower_limit,
                        Eigen::VectorXd* t_upper_limit) const;

  /**
   * Given a scaling factor rho, determine if the scaled box
   * max(negative_delta_q, negative_joint_limit) <= q - q_star <=
   * min(positive_delta_q, positive_joint_limit) is collision free.
   * @param rationals The rationals representing the postures are always
   * collision free.
   * @param t_lower_joint_limit The lower bound on t from the joint limit.
   * @param t_upper_joint_limit The upper bound on t from the joint limit.
   */
  bool IsBoxScalingFactorFeasible(
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
      const VerificationOption& option) const;

  /**
   * Given the scaling factor rho. Compute the bounds on q that is collision
   * free. We clamp the bounds to be within the joint limits, and clamp
   * positive_delta_q * rho to pi, and negative_delta_q * rho to -pi.
   */
  void CalcConfigurationBoundsFromRho(
      const Eigen::Ref<const Eigen::VectorXd>& negative_delta_q,
      const Eigen::Ref<const Eigen::VectorXd>& positive_delta_q, double rho,
      const Eigen::Ref<const Eigen::VectorXd>& q_star, Eigen::VectorXd* q_lower,
      Eigen::VectorXd* q_upper) const;

  /**
   * Try to increase each dimension of the box by a factor of 2. Repeat
   * enlarging the box in this way repeatedly until the box is not collision
   * free.
   * @return rho. We know that enlarging the box by rho is collision free,
   * but 2 * rho is not collision free.
   * @pre The initial box negative_delta_q <= delta_q <= positive_delta_q is
   * collision free.
   */
  int SequentiallyDoubleBoxSize(
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
      const BinarySearchOption& option) const;

  /**
   * We need to impose the constraint that verified_polynomials[i] >= 0.
   * This is a utility function called in the two SOS problems in bilinear
   * alternation.
   */
  void ConstrainVerifiedPolynomialsNonnegative(
      const std::vector<int>& verified_polynomials_expected_gram_rows,
      const std::vector<
          ConfigurationSpaceCollisionFreeRegion::BoxVerificationTuple>&
          bilinear_alternation_tuples,
      const std::vector<drake::symbolic::Polynomial>& verified_polynomials,
      drake::solvers::MathematicalProgram::NonnegativePolynomial
          nonnegative_polynomial_type,
      drake::solvers::MathematicalProgram* prog) const;

  RationalForwardKinematics rational_forward_kinematics_;
  std::map<drake::multibody::BodyIndex,
           std::vector<std::shared_ptr<const ConvexPolytope>>>
      link_polytopes_;
  // obstacles_[i] is the i'th polytope, fixed to the world.
  std::vector<std::shared_ptr<const ConvexPolytope>> obstacles_;

  SeparatingPlaneOrder a_order_;

  std::vector<SeparationPlane> separation_planes_;

  // In the key, the first ConvexGeometry::Id is for the polytope on the
  // positive side, the second ConvexGeometry::Id is for the one on the negative
  // side.
  std::unordered_map<std::pair<ConvexGeometry::Id, ConvexGeometry::Id>,
                     const SeparationPlane*, GeometryIdPairHash>
      map_polytopes_to_separation_planes_;
};

/**
 * Generate the rational functions a_A.dot(p_AVi(t) - p_AC) <= 1 (or >= 1)
 * which represents that the link (whose vertex Vi has position p_AVi in frame
 * A) is on the negative (or positive, respectively) side of the hyperplane.
 * @param rational_forward_kinematics The utility class that computes the
 * position of Vi in A's frame as a rational function of t.
 * @param link_polytope The polytopic representation of the link collision
 * geometry, Vi is the i'th vertex of the polytope.
 * @param other_side_link_polytope The plane separates two polytopes @p
 * link_polytope and @p other_side_link_polytope.
 * @param q_star The nominal configuration.
 * @param expressed_body_index Frame A in the documentation above. The body in
 * which the position is expressed in.
 * @param a_A The normal vector of the plane. This vector is expressed in
 * frame A.
 * @param p_AC The point within the interior of the negative side of the
 * plane.
 * @param plane_side Whether the link is on the positive or the negative side
 * of the plane.
 * @return rational_fun rational_fun[i] should be non-negative to represent
 * that the vertiex i is on the correct side of the plane. rational_fun[i] =
 * a_A.dot(p_AVi(t) - p_AC) - 1 if @p plane_side = kPositive, rational_fun[i]
 * = 1 - a_A.dot(p_AVi(t) - p_AC) if @p plane_side = kNegative.
 */
std::vector<LinkVertexOnPlaneSideRational>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    std::shared_ptr<const ConvexPolytope> link_polytope,
    std::shared_ptr<const ConvexPolytope> other_side_link_polytope,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    drake::multibody::BodyIndex expressed_body_index,
    const Eigen::Ref<const drake::Vector3<drake::symbolic::Expression>>& a_A,
    const Eigen::Ref<const Eigen::Vector3d>& p_AC, PlaneSide plane_side,
    SeparatingPlaneOrder a_order);

/**
 * Overloaded GenerateLinkOnOnseSideOfPlaneRationalFunction, except X_AB,
 * i.e.,
 * the pose of the link polytope in the expressed_frame is given.
 * @param X_AB_multilinear The pose of the link polytope frame B in the
 * expressed body frame A. Note that this pose is a multilinear function of
 * sinθ and cosθ.
 */
std::vector<LinkVertexOnPlaneSideRational>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    std::shared_ptr<const ConvexPolytope> link_polytope,
    std::shared_ptr<const ConvexPolytope> other_side_link_polytope,
    const RationalForwardKinematics::Pose<drake::symbolic::Polynomial>&
        X_AB_multilinear,
    const Eigen::Ref<const drake::Vector3<drake::symbolic::Expression>>& a_A,
    const Eigen::Ref<const Eigen::Vector3d>& p_AC, PlaneSide plane_side,
    SeparatingPlaneOrder a_order);

/** Impose the constraint that
 * l_lower(t) >= 0                                                         (1)
 * l_upper(t) >= 0                                                         (2)
 * p(t) - l_lower(t) * (t - t_lower) - l_upper(t) (t_upper - t) >= 0       (3)
 * where p(t) is the numerator of @p polytope_on_one_side_rational
 * @param monomial_basis The basis for the monomial of p(t), l_lower(t),
 * l_upper(t) and the polynomial (3) above.
 */
void AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
    drake::solvers::MathematicalProgram* prog,
    const drake::symbolic::RationalFunction& polytope_on_one_side_rational,
    const Eigen::Ref<const drake::VectorX<drake::symbolic::Polynomial>>&
        t_minus_t_lower,
    const Eigen::Ref<const drake::VectorX<drake::symbolic::Polynomial>>&
        t_upper_minus_t,
    const Eigen::Ref<const drake::VectorX<drake::symbolic::Monomial>>&
        monomial_basis,
    const VerificationOption& verification_option = {});

/**
 * Impose the constraint that
 * l(t) >= 0
 * p(t) - l(t)ᵀ(d - C*t) >= 0.
 * where l(t) is the Lagrangian multipliers. p(t) is the numerator of the
 * rational function `polytope_on_one_side_rational`.
 * @param monomial_basis The monomial basis for l(t) and p(t) - l(t)ᵀ(d - C*t)
 * @return (l(t), p(t) - l(t)ᵀ(d-C*t))
 */
std::pair<VectorX<symbolic::Polynomial>, symbolic::Polynomial>
AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
    solvers::MathematicalProgram* prog,
    const drake::symbolic::RationalFunction& polytope_on_one_side_rational,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& d_minus_Ct,
    const Eigen::Ref<const VectorX<symbolic::Monomial>>& monomial_basis,
    const VerificationOption& verification_option = {});

std::vector<ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairsForBox>
ReadFilteredCollisionPairsForBoxesFromFile(const std::string& file_name,
                                           int q_size);

ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairs
ExtractFilteredCollisionPairsForBox(
    const Eigen::Ref<const Eigen::VectorXd>& q_lower,
    const Eigen::Ref<const Eigen::VectorXd>& q_upper,
    const std::vector<
        ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairsForBox>&
        filtered_collision_pairs_for_boxes);
}  // namespace multibody
}  // namespace drake
