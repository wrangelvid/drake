#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/symbolic_test_util.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace multibody {
using drake::Vector3;
using drake::VectorX;
using drake::math::RigidTransformd;
using drake::math::RotationMatrixd;
using drake::multibody::BodyIndex;

void CheckGenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    const std::shared_ptr<const ConvexPolytope> link_polytope,
    const std::shared_ptr<const ConvexPolytope> other_side_link_polytope,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const Eigen::Ref<const Eigen::VectorXd>& q_val,
    BodyIndex expressed_body_index,
    const Eigen::Ref<const Vector3<double>>& a_A_val,
    const Eigen::Ref<const Eigen::Vector3d>& p_AC, PlaneSide plane_side,
    SeparatingPlaneOrder a_order) {
  drake::symbolic::Environment env;
  Vector3<drake::symbolic::Expression> a_A;
  const drake::symbolic::Monomial monomial_one{};
  for (int i = 0; i < 3; ++i) {
    if (a_order == SeparatingPlaneOrder::kConstant) {
      Vector3<drake::symbolic::Variable> a_A_var;
      a_A_var(i) = drake::symbolic::Variable("a_A(" + std::to_string(i) + ")");
      env.insert(a_A_var(i), a_A_val(i));
      a_A(i) = a_A_var(i);
    } else {
      throw std::runtime_error("Need to handle a as an affine function.");
    }
  }
  const std::vector<LinkVertexOnPlaneSideRational> rational_fun =
      GenerateLinkOnOneSideOfPlaneRationalFunction(
          rational_forward_kinematics, link_polytope, other_side_link_polytope,
          q_star, expressed_body_index, a_A, p_AC, plane_side, a_order);
  for (const auto& rational : rational_fun) {
    EXPECT_EQ(rational.other_side_link_polytope, other_side_link_polytope);
  }
  const Eigen::VectorXd t_val =
      rational_forward_kinematics.ComputeTValue(q_val, q_star);
  for (int i = 0; i < t_val.size(); ++i) {
    env.insert(rational_forward_kinematics.t()(i), t_val(i));
  }

  // Compute link points position in the expressed body.
  auto context = rational_forward_kinematics.plant().CreateDefaultContext();
  rational_forward_kinematics.plant().SetPositions(context.get(), q_val);
  const drake::math::RigidTransformd X_AB =
      rational_forward_kinematics.plant().CalcRelativeTransform(
          *context,
          rational_forward_kinematics.plant()
              .get_body(expressed_body_index)
              .body_frame(),
          rational_forward_kinematics.plant()
              .get_body(link_polytope->body_index())
              .body_frame());
  const Eigen::Matrix3Xd p_AV_expected = X_AB * link_polytope->p_BV();

  const Eigen::Matrix3Xd r_A_expected = X_AB.rotation() * link_polytope->r_B();

  const VectorX<drake::symbolic::Variable> t_on_path =
      rational_forward_kinematics.FindTOnPath(expressed_body_index,
                                              link_polytope->body_index());
  const drake::symbolic::Variables t_on_path_set(t_on_path);

  EXPECT_EQ(rational_fun.size(), link_polytope->p_BV().cols());
  const double tol{1E-12};
  for (int i = 0; i < link_polytope->p_BV().cols(); ++i) {
    EXPECT_EQ(rational_fun[i].link_polytope, link_polytope);
    EXPECT_EQ(rational_fun[i].expressed_body_index, expressed_body_index);
    EXPECT_EQ(rational_fun[i].a_A, a_A);
    EXPECT_EQ(rational_fun[i].plane_side, plane_side);
    EXPECT_TRUE(
        rational_fun[i].rational.numerator().indeterminates().IsSubsetOf(
            t_on_path_set));
    // Check that rational_fun[i] only contains the right t.
    const double rational_fun_val =
        rational_fun[i].rational.numerator().Evaluate(env) /
        rational_fun[i].rational.denominator().Evaluate(env);
    const double rational_fun_val_expected =
        plane_side == PlaneSide::kPositive
            ? a_A_val.dot(p_AV_expected.col(i) - p_AC) - 1
            : 1 - a_A_val.dot(p_AV_expected.col(i) - p_AC);
    EXPECT_NEAR(rational_fun_val, rational_fun_val_expected, tol);
  }

  for (int i = 0; i < link_polytope->r_B().cols(); ++i) {
    const int rational_fun_index = i + link_polytope->p_BV().cols();
    EXPECT_EQ(rational_fun[rational_fun_index].link_polytope, link_polytope);
    EXPECT_EQ(rational_fun[rational_fun_index].expressed_body_index,
              expressed_body_index);
    EXPECT_EQ(rational_fun[rational_fun_index].a_A, a_A);
    EXPECT_EQ(rational_fun[rational_fun_index].plane_side, plane_side);
    EXPECT_TRUE(rational_fun[rational_fun_index]
                    .rational.numerator()
                    .indeterminates()
                    .IsSubsetOf(t_on_path_set));
    // Check that rational_fun[i] only contains the right t.
    const double rational_fun_val =
        rational_fun[rational_fun_index].rational.numerator().Evaluate(env) /
        rational_fun[rational_fun_index].rational.denominator().Evaluate(env);
    const double rational_fun_val_expected =
        plane_side == PlaneSide::kPositive ? a_A_val.dot(r_A_expected.col(i))
                                           : -a_A_val.dot(r_A_expected.col(i));
    EXPECT_NEAR(rational_fun_val, rational_fun_val_expected, tol);
  }
}

TEST_F(IiwaTest, GenerateLinkOnOneSideOfPlaneRationalFunction1) {
  const RationalForwardKinematics rational_forward_kinematics(*iiwa_);

  // Arbitrary pose between link polytope and the attached link.
  const RigidTransformd X_6V{
      RotationMatrixd(Eigen::AngleAxisd(
          0.2 * M_PI, Eigen::Vector3d(0.1, 0.4, 0.3).normalized())),
      {0.2, -0.1, 0.3}};
  const auto p_6V = GenerateBoxVertices(Eigen::Vector3d(0.1, 0.2, 0.3), X_6V);
  auto link6_polytope =
      std::make_shared<const ConvexPolytope>(iiwa_link_[6], p_6V);

  auto obstacle = std::make_shared<const ConvexPolytope>(
      world_, GenerateBoxVertices(Eigen::Vector3d::Ones(), {}));

  Eigen::VectorXd q(7);
  q.setZero();
  Eigen::VectorXd q_star(7);
  q_star.setZero();
  Eigen::Vector3d a_A(1.2, -0.4, 3.1);
  const Eigen::Vector3d p_AC(0.5, -2.1, 0.6);
  CheckGenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link6_polytope, nullptr, q_star, q, world_,
      a_A, p_AC, PlaneSide::kPositive, SeparatingPlaneOrder::kConstant);

  q << 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7;
  q_star << -0.25, 0.13, 0.26, 0.65, -0.02, 0.87, 0.42;
  CheckGenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link6_polytope, obstacle, q_star, q,
      iiwa_link_[3], a_A, p_AC, PlaneSide::kNegative,
      SeparatingPlaneOrder::kConstant);
}

TEST_F(IiwaTest, GenerateLinkOnOneSideOfPlaneRationalFunction2) {
  const RationalForwardKinematics rational_forward_kinematics(*iiwa_);

  // Arbitrary pose between link polytope and the attached link.
  RigidTransformd X_3V{
      RotationMatrixd(Eigen::AngleAxisd(
          0.3 * M_PI, Eigen::Vector3d(0.1, 0.4, 0.3).normalized())),
      {-0.2, -0.1, 0.3}};
  const auto p_3V = GenerateBoxVertices(Eigen::Vector3d(0.1, 0.2, 0.3), X_3V);
  auto link3_polytope =
      std::make_shared<const ConvexPolytope>(iiwa_link_[3], p_3V);

  auto obstacle = std::make_shared<const ConvexPolytope>(
      world_, GenerateBoxVertices(Eigen::Vector3d::Ones(),
                                  RigidTransformd::Identity()));

  Eigen::VectorXd q(7);
  q.setZero();
  Eigen::VectorXd q_star(7);
  q_star.setZero();
  Eigen::Vector3d a_A(1.2, -0.4, 3.1);
  const Eigen::Vector3d p_AC(0.5, -2.1, 0.6);
  CheckGenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link3_polytope, obstacle, q_star, q,
      iiwa_link_[7], a_A, p_AC, PlaneSide::kPositive,
      SeparatingPlaneOrder::kConstant);

  q << 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7;
  q_star << -0.25, 0.13, 0.26, 0.65, -0.02, 0.87, 0.42;
  CheckGenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link3_polytope, obstacle, q_star, q,
      iiwa_link_[5], a_A, p_AC, PlaneSide::kNegative,
      SeparatingPlaneOrder::kConstant);

  CheckGenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link3_polytope, obstacle, q_star, q, world_,
      a_A, p_AC, PlaneSide::kNegative, SeparatingPlaneOrder::kConstant);
}

TEST_F(IiwaTest, GenerateLinkOnOneSideOfPlaneRationalFunction3) {
  const RationalForwardKinematics rational_forward_kinematics(*iiwa_);

  // Arbitrary pose between link polytope and the attached link.
  const RigidTransformd X_6V{
      RotationMatrixd(Eigen::AngleAxisd(
          0.2 * M_PI, Eigen::Vector3d(0.1, 0.4, 0.3).normalized())),
      {0.2, -0.1, 0.3}};
  const auto p_6V = GenerateBoxVertices(Eigen::Vector3d(0.1, 0.2, 0.3), X_6V);
  auto link6_polytope =
      std::make_shared<const ConvexPolytope>(iiwa_link_[6], p_6V);

  auto obstacle = std::make_shared<const ConvexPolytope>(
      world_, Eigen::Vector3d::Ones(), Eigen::Matrix3d::Identity());

  Eigen::VectorXd q(7);
  q.setZero();
  Eigen::VectorXd q_star(7);
  q_star.setZero();
  Eigen::Vector3d a_A(1.2, -0.4, 3.1);
  const Eigen::Vector3d p_AC(0.5, -2.1, 0.6);
  CheckGenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link6_polytope, nullptr, q_star, q, world_,
      a_A, p_AC, PlaneSide::kPositive, SeparatingPlaneOrder::kConstant);

  q << 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7;
  q_star << -0.25, 0.13, 0.26, 0.65, -0.02, 0.87, 0.42;
  CheckGenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link6_polytope, obstacle, q_star, q,
      iiwa_link_[3], a_A, p_AC, PlaneSide::kNegative,
      SeparatingPlaneOrder::kConstant);
}

class IiwaConfigurationSpaceTest : public IiwaTest {
 public:
  IiwaConfigurationSpaceTest() {
    // Arbitrarily add some polytopes to links
    link7_polytopes_.emplace_back(std::make_shared<const ConvexPolytope>(
        iiwa_link_[7],
        GenerateBoxVertices(Eigen::Vector3d(0.1, 0.1, 0.2), {})));
    const RigidTransformd X_7P{RotationMatrixd(Eigen::AngleAxisd(
                                   0.2 * M_PI, Eigen::Vector3d::UnitX())),
                               {0.1, 0.2, -0.1}};
    link7_polytopes_.emplace_back(std::make_shared<const ConvexPolytope>(
        iiwa_link_[7],
        GenerateBoxVertices(Eigen::Vector3d(0.1, 0.2, 0.1), X_7P)));

    const RigidTransformd X_5P{X_7P.rotation(), {-0.2, 0.1, 0}};
    link5_polytopes_.emplace_back(std::make_shared<const ConvexPolytope>(
        iiwa_link_[5],
        GenerateBoxVertices(Eigen::Vector3d(0.2, 0.1, 0.2), X_5P)));

    RigidTransformd X_WP = X_5P * Eigen::Translation3d(0.15, -0.1, 0.05);
    obstacles_.emplace_back(std::make_shared<const ConvexPolytope>(
        world_, GenerateBoxVertices(Eigen::Vector3d(0.1, 0.2, 0.15), X_WP)));
    X_WP = X_WP * RigidTransformd(RotationMatrixd(Eigen::AngleAxisd(
                      -0.1 * M_PI, Eigen::Vector3d::UnitY())));
    obstacles_.emplace_back(std::make_shared<const ConvexPolytope>(
        world_, GenerateBoxVertices(Eigen::Vector3d(0.1, 0.25, 0.15), X_WP)));
  }

 protected:
  std::vector<std::shared_ptr<const ConvexPolytope>> link7_polytopes_;
  std::vector<std::shared_ptr<const ConvexPolytope>> link5_polytopes_;
  std::vector<std::shared_ptr<const ConvexPolytope>> obstacles_;
};

TEST_F(IiwaConfigurationSpaceTest, TestConstructor) {
  ConfigurationSpaceCollisionFreeRegion dut(
      *iiwa_, {link7_polytopes_[0], link7_polytopes_[1], link5_polytopes_[0]},
      obstacles_, SeparatingPlaneOrder::kConstant);
  const ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairs
      filtered_collision_pairs{drake::SortedPair<ConvexGeometry::Id>(
          link7_polytopes_[0]->get_id(), obstacles_[0]->get_id())};

  const auto& separation_planes = dut.separation_planes();
  EXPECT_EQ(separation_planes.size(), 6);

  auto CheckSeparationPlane = [&](const SeparationPlane& separation_plane,
                                  ConvexGeometry::Id expected_positive_polytope,
                                  ConvexGeometry::Id expected_negative_polytope,
                                  BodyIndex expressed_body_index) {
    EXPECT_EQ(separation_plane.positive_side_polytope->get_id(),
              expected_positive_polytope);
    EXPECT_EQ(separation_plane.negative_side_polytope->get_id(),
              expected_negative_polytope);
    EXPECT_EQ(separation_plane.expressed_link, expressed_body_index);

    EXPECT_EQ(dut.map_polytopes_to_separation_planes()
                  .find(std::make_pair(expected_positive_polytope,
                                       expected_negative_polytope))
                  ->second,
              &separation_plane);
    EXPECT_EQ(separation_plane.a_order, SeparatingPlaneOrder::kConstant);
    EXPECT_EQ(separation_plane.decision_variables.size(), 3);
    EXPECT_EQ(separation_plane.a, separation_plane.decision_variables
                                      .cast<drake::symbolic::Expression>());
  };

  CheckSeparationPlane(separation_planes[0], link5_polytopes_[0]->get_id(),
                       obstacles_[0]->get_id(), iiwa_link_[3]);
  CheckSeparationPlane(separation_planes[1], link7_polytopes_[0]->get_id(),
                       obstacles_[0]->get_id(), iiwa_link_[4]);
  CheckSeparationPlane(separation_planes[2], link7_polytopes_[1]->get_id(),
                       obstacles_[0]->get_id(), iiwa_link_[4]);
  CheckSeparationPlane(separation_planes[3], link5_polytopes_[0]->get_id(),
                       obstacles_[1]->get_id(), iiwa_link_[3]);
  CheckSeparationPlane(separation_planes[4], link7_polytopes_[0]->get_id(),
                       obstacles_[1]->get_id(), iiwa_link_[4]);
  CheckSeparationPlane(separation_planes[5], link7_polytopes_[1]->get_id(),
                       obstacles_[1]->get_id(), iiwa_link_[4]);
}

TEST_F(IiwaConfigurationSpaceTest, GenerateLinkOnOneSideOfPlanePolynomials) {
  ConfigurationSpaceCollisionFreeRegion dut(
      *iiwa_, {link7_polytopes_[0], link7_polytopes_[1], link5_polytopes_[0]},
      obstacles_, SeparatingPlaneOrder::kConstant);

  EXPECT_EQ(dut.separation_planes().size(), 6);

  const ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairs
      filtered_collision_pairs{drake::SortedPair<ConvexGeometry::Id>(
          link7_polytopes_[0]->get_id(), obstacles_[0]->get_id())};
  const auto rationals = dut.GenerateLinkOnOneSideOfPlaneRationals(
      Eigen::VectorXd::Zero(7), filtered_collision_pairs);
  EXPECT_EQ(rationals.size(), 80);
}

TEST_F(IiwaConfigurationSpaceTest, TestConstructorWithAffineSeparatingPlane) {
  ConfigurationSpaceCollisionFreeRegion dut(
      *iiwa_, {link7_polytopes_[0], link7_polytopes_[1], link5_polytopes_[0]},
      obstacles_, SeparatingPlaneOrder::kAffine);

  const auto& separation_planes = dut.separation_planes();
  EXPECT_EQ(separation_planes.size(), 6);
  auto check_separation_plane =
      [](const SeparationPlane& plane,
         const Eigen::Ref<const VectorX<drake::symbolic::Variable>>&
             t_expected) {
        EXPECT_EQ(plane.a_order, SeparatingPlaneOrder::kAffine);
        EXPECT_EQ(plane.decision_variables.rows(), 3 * t_expected.rows() + 3);
        EXPECT_EQ(drake::symbolic::Variables(plane.decision_variables).size(),
                  3 * t_expected.rows() + 3);
        // Now check if a(i) is an affine function of t_expected.
        for (int i = 0; i < 3; ++i) {
          const drake::symbolic::Polynomial a_poly(
              plane.a(i), drake::symbolic::Variables(t_expected));
          EXPECT_EQ(a_poly.TotalDegree(), 1);
          drake::symbolic::Polynomial a_poly_expected{};
          for (int j = 0; j < t_expected.rows(); ++j) {
            a_poly_expected.AddProduct(
                plane.decision_variables(3 * j + i),
                drake::symbolic::Monomial(t_expected(j), 1));
          }
          a_poly_expected.AddProduct(
              plane.decision_variables(3 * t_expected.rows() + i),
              drake::symbolic::Monomial{});
          EXPECT_EQ(a_poly, a_poly_expected);
        }
      };

  const auto& t = dut.rational_forward_kinematics().t();

  // link5_polytopes_[0] and obstacles_[0]
  check_separation_plane(separation_planes[0], t.head<5>());
  // link7_polytopes_[0] and obstacles_[0]
  check_separation_plane(separation_planes[1], t.head<7>());
  // link7_polytopes_[1] and obstacles_[0]
  check_separation_plane(separation_planes[2], t.head<7>());
  // link5_polytopes_[0] and obstacles_[1]
  check_separation_plane(separation_planes[3], t.head<5>());
  // link7_polytopes_[0] and obstacles_[1]
  check_separation_plane(separation_planes[4], t.head<7>());
  // link7_polytopes_[1] and obstacles_[1]
  check_separation_plane(separation_planes[5], t.head<7>());
}

// The rational's numerator's monomials should be a subset of ∏ᵢ tᵢⁿⁱ,
// where tᵢ is in @p t_on_half_chain, ni = 0, 1, 2
std::unordered_set<drake::symbolic::Monomial>
GenerateMonomialsForLinkOnOneSideOfPlaneRationalConstantSeparatingPlane(
    const Eigen::Ref<const VectorX<drake::symbolic::Variable>>&
        t_on_half_chain) {
  std::queue<drake::symbolic::Monomial> monomials;
  monomials.emplace();
  std::unordered_set<drake::symbolic::Monomial> monomial_set;
  while (!monomials.empty()) {
    const auto& monomial = monomials.front();
    for (int i = 0; i < t_on_half_chain.rows(); ++i) {
      if (monomial.degree(t_on_half_chain(i)) < 2) {
        monomials.push(monomial *
                       drake::symbolic::Monomial(t_on_half_chain(i)));
      }
    }
    monomial_set.insert(monomial);
    monomials.pop();
  }
  return monomial_set;
}

// The rational's numerator's monomials should be a subset of tⱼ ∏ᵢ tᵢⁿⁱ and ∏ᵢ
// tᵢⁿⁱ where tᵢ is in @p t_on_half_chain, ni = 0, 1, 2, and tⱼ is in @p
// t_on_whole_chain.
std::unordered_set<drake::symbolic::Monomial>
GenerateMonomialsForLinkOnOneSideOfPlaneRationalAffineSeparatingPlane(
    const Eigen::Ref<const VectorX<drake::symbolic::Variable>>& t_on_half_chain,
    const Eigen::Ref<const VectorX<drake::symbolic::Variable>>&
        t_on_whole_chain) {
  const std::unordered_set<drake::symbolic::Monomial> monomials_constant_plane =
      GenerateMonomialsForLinkOnOneSideOfPlaneRationalConstantSeparatingPlane(
          t_on_half_chain);
  std::unordered_set<drake::symbolic::Monomial> monomial_set =
      monomials_constant_plane;
  VectorX<drake::symbolic::Monomial> t_monomial(t_on_whole_chain.rows());
  for (int i = 0; i < t_on_whole_chain.rows(); ++i) {
    t_monomial(i) = drake::symbolic::Monomial(t_on_whole_chain(i));
  }
  for (const auto& monomial : monomials_constant_plane) {
    for (int i = 0; i < t_on_whole_chain.rows(); ++i) {
      monomial_set.insert(monomial * t_monomial(i));
    }
  }
  return monomial_set;
}

TEST_F(IiwaConfigurationSpaceTest,
       GenerateLinkOnOneSideOfPlaneRationalsAffineSeparatingPlane) {
  ConfigurationSpaceCollisionFreeRegion dut(*iiwa_, {link5_polytopes_[0]},
                                            {obstacles_[0]},
                                            SeparatingPlaneOrder::kAffine);
  Eigen::VectorXd q_star(7);
  q_star << 0.1, 0.2, 0.3, 0.4, -0.5, -0.2, -0.3;
  const std::vector<LinkVertexOnPlaneSideRational> rationals =
      dut.GenerateLinkOnOneSideOfPlaneRationals(q_star, {});
  // 1 rational for each vertex on link and obstacle polytopes.
  EXPECT_EQ(rationals.size(), 16);
  const auto& t = dut.rational_forward_kinematics().t();
  auto check_rational = [](const LinkVertexOnPlaneSideRational& rational,
                           std::shared_ptr<const ConvexPolytope> link_polytope,
                           BodyIndex expressed_body_index,
                           std::shared_ptr<const ConvexPolytope>
                               other_side_link_polytope,
                           PlaneSide plane_side, SeparatingPlaneOrder a_order,
                           const Eigen::Ref<
                               const VectorX<drake::symbolic::Variable>>&
                               t_on_half_chain,
                           const Eigen::Ref<
                               const VectorX<drake::symbolic::Variable>>&
                               t_on_whole_chain) {
    EXPECT_EQ(rational.link_polytope, link_polytope);
    EXPECT_EQ(rational.expressed_body_index, expressed_body_index);
    EXPECT_EQ(rational.other_side_link_polytope, other_side_link_polytope);
    EXPECT_EQ(rational.plane_side, plane_side);
    EXPECT_EQ(rational.a_order, a_order);
    const std::unordered_set<drake::symbolic::Monomial> monomial_set =
        GenerateMonomialsForLinkOnOneSideOfPlaneRationalAffineSeparatingPlane(
            t_on_half_chain, t_on_whole_chain);
    for (const auto& item :
         rational.rational.numerator().monomial_to_coefficient_map()) {
      EXPECT_GT(monomial_set.count(item.first), 0);
    }
  };
  for (int i = 0; i < 8; ++i) {
    // The first 8 are for the vertex on link5_polytopes_[0]
    // The rational's numerator's monomials should be a subset of tⱼ ∏ᵢ tᵢⁿⁱ,
    // where i = 3, 4, ni = 0, 1, 2, and j = 0, 1, ..., 4.
    check_rational(rationals[i], link5_polytopes_[0], iiwa_link_[3],
                   obstacles_[0], PlaneSide::kPositive,
                   SeparatingPlaneOrder::kAffine, t.segment<2>(3), t.head<5>());
    // The last 8 are for the vertex on obstacles_[0]
    // The rational's numerator's monomials should be a subset of tⱼ ∏ᵢ tᵢⁿⁱ,
    // where i = 0, 1, 2, ni = 0, 1, 2, and j = 0, 1, ..., 4.
    check_rational(rationals[i + 8], obstacles_[0], iiwa_link_[3],
                   link5_polytopes_[0], PlaneSide::kNegative,
                   SeparatingPlaneOrder::kAffine, t.head<3>(), t.head<5>());
  }
}

TEST_F(IiwaConfigurationSpaceTest, IsInCollision) {
  ConfigurationSpaceCollisionFreeRegion dut(
      *iiwa_, {link7_polytopes_[0], link7_polytopes_[1], link5_polytopes_[0]},
      {obstacles_[0], obstacles_[1]}, SeparatingPlaneOrder::kAffine);
  auto context = iiwa_->CreateDefaultContext();
  Eigen::VectorXd q(7);
  q << 0, 0, 0, 0, 0, 0, 0;
  iiwa_->SetPositions(context.get(), q);
  EXPECT_FALSE(dut.IsPostureCollisionFree(*context));

  // Now solve a posture that link7 reaches obstacles_[1]
  drake::multibody::InverseKinematics ik(*iiwa_);

  ik.AddPositionConstraint(iiwa_->get_body(iiwa_link_[7]).body_frame(),
                           link7_polytopes_[1]->p_BC(), iiwa_->world_frame(),
                           obstacles_[1]->p_BC(), obstacles_[1]->p_BC());
  const auto result = drake::solvers::Solve(ik.prog());
  EXPECT_TRUE(result.is_success());
  iiwa_->SetPositions(context.get(), result.GetSolution(ik.q()));
  EXPECT_TRUE(dut.IsPostureCollisionFree(*context));
}

}  // namespace multibody
}  // namespace drake
