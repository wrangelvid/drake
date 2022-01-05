#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"

#include <chrono>
#include <limits>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/symbolic_test_util.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/plant/coulomb_friction.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace multibody {
using drake::Vector3;
using drake::VectorX;
using drake::math::RigidTransformd;
using drake::math::RotationMatrixd;
using drake::multibody::BodyIndex;

const double kInf = std::numeric_limits<double>::infinity();

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
  geometry::SceneGraph<double> sg;
  iiwa_->RegisterAsSourceForSceneGraph(&sg);
  // Arbitrary pose between link polytope and the attached link.
  const RigidTransformd X_6V{
      RotationMatrixd(Eigen::AngleAxisd(
          0.2 * M_PI, Eigen::Vector3d(0.1, 0.4, 0.3).normalized())),
      {0.2, -0.1, 0.3}};
  const auto p_6V = GenerateBoxVertices(Eigen::Vector3d(0.1, 0.2, 0.3), X_6V);
  const auto link6_box_id = iiwa_->RegisterCollisionGeometry(
      iiwa_->get_body(iiwa_link_[6]), X_6V, geometry::Box(0.1, 0.2, 0.3),
      "link6_box", CoulombFriction<double>());
  auto link6_polytope =
      std::make_shared<const ConvexPolytope>(iiwa_link_[6], link6_box_id, p_6V);

  const auto obstacle_id = iiwa_->RegisterCollisionGeometry(
      iiwa_->world_body(), {}, geometry::Box(1, 1, 1), "world_box",
      CoulombFriction<double>());
  auto obstacle = std::make_shared<const ConvexPolytope>(
      world_, obstacle_id, GenerateBoxVertices(Eigen::Vector3d::Ones(), {}));
  iiwa_->Finalize();

  const RationalForwardKinematics rational_forward_kinematics(*iiwa_);

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
  geometry::SceneGraph<double> sg;
  iiwa_->RegisterAsSourceForSceneGraph(&sg);

  // Arbitrary pose between link polytope and the attached link.
  RigidTransformd X_3V{
      RotationMatrixd(Eigen::AngleAxisd(
          0.3 * M_PI, Eigen::Vector3d(0.1, 0.4, 0.3).normalized())),
      {-0.2, -0.1, 0.3}};
  const auto p_3V = GenerateBoxVertices(Eigen::Vector3d(0.1, 0.2, 0.3), X_3V);
  const auto link3_polytope_id = iiwa_->RegisterCollisionGeometry(
      iiwa_->get_body(iiwa_link_[3]), X_3V, geometry::Box(0.1, 0.2, 0.3),
      "link3_box", CoulombFriction<double>());
  auto link3_polytope = std::make_shared<const ConvexPolytope>(
      iiwa_link_[3], link3_polytope_id, p_3V);

  const auto obstacle_id = iiwa_->RegisterCollisionGeometry(
      iiwa_->world_body(), {}, geometry::Box(1, 1, 1), "world_box",
      CoulombFriction<double>{});
  auto obstacle = std::make_shared<const ConvexPolytope>(
      world_, obstacle_id,
      GenerateBoxVertices(Eigen::Vector3d::Ones(),
                          RigidTransformd::Identity()));
  iiwa_->Finalize();

  const RationalForwardKinematics rational_forward_kinematics(*iiwa_);

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
  geometry::SceneGraph<double> sg;
  iiwa_->RegisterAsSourceForSceneGraph(&sg);

  // Arbitrary pose between link polytope and the attached link.
  const RigidTransformd X_6V{
      RotationMatrixd(Eigen::AngleAxisd(
          0.2 * M_PI, Eigen::Vector3d(0.1, 0.4, 0.3).normalized())),
      {0.2, -0.1, 0.3}};
  const auto p_6V = GenerateBoxVertices(Eigen::Vector3d(0.1, 0.2, 0.3), X_6V);
  const auto link6_polytope_id = iiwa_->RegisterCollisionGeometry(
      iiwa_->get_body(iiwa_link_[6]), X_6V, geometry::Box(0.1, 0.2, 0.3),
      "iiwa_link6_box", CoulombFriction<double>());
  auto link6_polytope = std::make_shared<const ConvexPolytope>(
      iiwa_link_[6], link6_polytope_id, p_6V);

  const auto obstacle_id = iiwa_->RegisterCollisionGeometry(
      iiwa_->world_body(), {}, geometry::Box(1, 1, 1), "world_box",
      CoulombFriction<double>());
  auto obstacle = std::make_shared<const ConvexPolytope>(
      world_, obstacle_id, Eigen::Vector3d::Ones(),
      Eigen::Matrix3d::Identity());
  iiwa_->Finalize();
  const RationalForwardKinematics rational_forward_kinematics(*iiwa_);

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
    geometry::SceneGraph<double> sg;
    iiwa_->RegisterAsSourceForSceneGraph(&sg);

    AddBox({}, Eigen::Vector3d(0.1, 0.1, 0.2), iiwa_link_[7], "link7_box1",
           &link7_polytopes_);
    const RigidTransformd X_7P{RotationMatrixd(Eigen::AngleAxisd(
                                   0.2 * M_PI, Eigen::Vector3d::UnitX())),
                               {0.1, 0.2, -0.1}};
    AddBox(X_7P, Eigen::Vector3d(0.1, 0.2, 0.1), iiwa_link_[7], "link7_box2",
           &link7_polytopes_);

    const RigidTransformd X_5P{X_7P.rotation(), {-0.2, 0.1, 0}};
    AddBox(X_5P, Eigen::Vector3d(0.2, 0.1, 0.2), iiwa_link_[5], "link5_box1",
           &link5_polytopes_);

    RigidTransformd X_WP = X_5P * Eigen::Translation3d(0.15, -0.1, 0.05);
    AddBox(X_WP, Eigen::Vector3d(0.1, 0.2, 0.15), world_, "world_box1",
           &obstacles_);
    X_WP = X_WP * RigidTransformd(RotationMatrixd(Eigen::AngleAxisd(
                      -0.1 * M_PI, Eigen::Vector3d::UnitY())));
    AddBox(X_WP, Eigen::Vector3d(0.1, 0.25, 0.15), world_, "world_box2",
           &obstacles_);

    iiwa_->Finalize();
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

TEST_F(IiwaConfigurationSpaceTest, GenerateLinkOnOneSideOfPlanePolynomials1) {
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

TEST_F(IiwaConfigurationSpaceTest, GenerateLinkOnOneSideOfPlaneRationals2) {
  // Test when q_not_in_collision is different from q_star.
  ConfigurationSpaceCollisionFreeRegion dut(*iiwa_, {link7_polytopes_[0]},
                                            {obstacles_[0]},
                                            SeparatingPlaneOrder::kConstant);
  EXPECT_EQ(dut.separation_planes().size(), 1);
  const Eigen::VectorXd q_star = Eigen::VectorXd::Zero(7);
  const ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairs
      filtered_collision_pairs{};
  Eigen::VectorXd q_not_in_collision(7);
  q_not_in_collision << 0.5, 0.5, -0.1, 0.3, 0.2, 0.1, 0.1;
  const auto& plant = dut.rational_forward_kinematics().plant();
  auto context = plant.CreateDefaultContext();
  plant.SetPositions(context.get(), q_not_in_collision);
  ASSERT_TRUE(dut.IsPostureCollisionFree(*context));
  const auto rationals = dut.GenerateLinkOnOneSideOfPlaneRationals(
      q_star, filtered_collision_pairs, q_not_in_collision);
  EXPECT_EQ(rationals.size(),
            link7_polytopes_[0]->p_BV().cols() + obstacles_[0]->p_BV().cols());
  const drake::multibody::BodyIndex expressed_boyd_index =
      internal::FindBodyInTheMiddleOfChain(plant, plant.world_body().index(),
                                           link7_polytopes_[0]->body_index());
  Eigen::Vector3d p_AC;
  plant.SetPositions(context.get(), q_not_in_collision);
  plant.CalcPointsPositions(
      *context, plant.get_body(obstacles_[0]->body_index()).body_frame(),
      obstacles_[0]->p_BC(), plant.get_body(expressed_boyd_index).body_frame(),
      &p_AC);
  std::vector<Eigen::VectorXd> q_samples;
  q_samples.push_back(
      (Eigen::VectorXd(7) << 0.1, -0.2, 0.5, 0.3, 0.2, 0.1, 0.5).finished());
  q_samples.push_back(
      (Eigen::VectorXd(7) << 0.6, -0.4, 0.3, 0.2, 0.1, 0.4, 0.5).finished());
  for (int i = 0; i < link7_polytopes_[0]->p_BV().cols(); ++i) {
    EXPECT_EQ(rationals[i].a_order, SeparatingPlaneOrder::kConstant);
    EXPECT_EQ(rationals[i].expressed_body_index, expressed_boyd_index);
    for (const auto& q : q_samples) {
      plant.SetPositions(context.get(), q);
      Eigen::Vector3d p_AV;
      plant.CalcPointsPositions(
          *context,
          plant.get_body(link7_polytopes_[0]->body_index()).body_frame(),
          link7_polytopes_[0]->p_BV().col(i),
          plant.get_body(expressed_boyd_index).body_frame(), &p_AV);
      const Eigen::VectorXd t_val = ((q - q_star) / 2).array().tan();
      symbolic::Environment env;
      for (int j = 0; j < dut.rational_forward_kinematics().t().rows(); ++j) {
        env.insert(dut.rational_forward_kinematics().t()(j), t_val(j));
      }
      const auto separating_plane =
          dut.map_polytopes_to_separation_planes()
              .find(std::make_pair(link7_polytopes_[0]->get_id(),
                                   obstacles_[0]->get_id()))
              ->second;
      for (int j = 0; j < separating_plane->decision_variables.rows(); ++j) {
        // Set the separating plane a_A's decision variable to some random
        // number.
        env.insert(separating_plane->decision_variables(j), 0.2 * j + 0.1);
      }

      const double rational_val_expected = rationals[i].rational.Evaluate(env);
      Eigen::Vector3d a_A_val;
      for (int j = 0; j < 3; ++j) {
        a_A_val(j) = rationals[i].a_A(j).Evaluate(env);
      }
      const double rational_val =
          rationals[i].plane_side == PlaneSide::kPositive
              ? a_A_val.dot(p_AV - p_AC) - 1
              : 1 - a_A_val.dot(p_AV - p_AC);
      EXPECT_NEAR(rational_val_expected, rational_val, 1E-10);
    }
  }
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

// The rational's numerator's monomials should be a subset of tⱼ ∏ᵢ tᵢⁿⁱ and
// ∏ᵢ / tᵢⁿⁱ where tᵢ is in @p t_on_half_chain,
// ni = 0, 1, 2, and tⱼ is in @p t_on_whole_chain.
std::unordered_set<drake::symbolic::Monomial>
        GenerateMonomialsForLinkOnOneSideOfPlaneRationalAffineSeparatingPlane(
            const Eigen::Ref<const VectorX<drake::symbolic::Variable>>&
                t_on_half_chain,
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
  EXPECT_TRUE(dut.IsPostureCollisionFree(*context));

  // Now solve a posture that link7 reaches obstacles_[1]
  drake::multibody::InverseKinematics ik(*iiwa_);

  ik.AddPositionConstraint(iiwa_->get_body(iiwa_link_[7]).body_frame(),
                           link7_polytopes_[1]->p_BC(), iiwa_->world_frame(),
                           obstacles_[1]->p_BC(), obstacles_[1]->p_BC());
  const auto result = drake::solvers::Solve(ik.prog());
  EXPECT_TRUE(result.is_success());
  iiwa_->SetPositions(context.get(), result.GetSolution(ik.q()));
  EXPECT_FALSE(dut.IsPostureCollisionFree(*context));
}

// Checks if the polyhedron C * x <= d is bounded.
bool IsPolyhedronBounded(const Eigen::Ref<const Eigen::MatrixXd>& C,
                         const Eigen::Ref<const Eigen::VectorXd>& d) {
  solvers::MathematicalProgram prog;
  auto x = prog.NewContinuousVariables(C.cols());
  prog.AddLinearConstraint(C, Eigen::VectorXd::Constant(d.rows(), -kInf), d, x);
  auto cost = prog.AddLinearCost(Eigen::VectorXd::Zero(x.rows()), 0, x);
  for (int i = 0; i < x.rows(); ++i) {
    Eigen::VectorXd new_cost_coeff = Eigen::VectorXd::Zero(x.rows());
    new_cost_coeff(i) = 1;
    cost.evaluator()->UpdateCoefficients(new_cost_coeff);
    auto result = solvers::Solve(prog);
    std::cout << i << " min \n";
    if (!result.is_success()) {
      return false;
    } else {
      std::cout << result.get_optimal_cost() << "\n";
    }
    new_cost_coeff(i) = -1;
    cost.evaluator()->UpdateCoefficients(new_cost_coeff);
    result = solvers::Solve(prog);
    std::cout << i << " max \n";
    if (!result.is_success()) {
      return false;
    } else {
      std::cout << -result.get_optimal_cost() << "\n";
    }
  }
  return true;
}

TEST_F(IiwaConfigurationSpaceTest,
       ConstructProgramToVerifyCollisionFreePolytope) {
  ConfigurationSpaceCollisionFreeRegion dut(*iiwa_, {link7_polytopes_[0]},
                                            {obstacles_[0], obstacles_[1]},
                                            SeparatingPlaneOrder::kAffine);
  const auto& plant = dut.rational_forward_kinematics().plant();
  auto context = plant.CreateDefaultContext();
  const Eigen::VectorXd q_star = Eigen::VectorXd::Zero(7);
  Eigen::VectorXd q_not_in_collision(7);
  q_not_in_collision << 0.5, 0.3, -0.2, 0.1, 0.4, 0.2, 0.1;
  plant.SetPositions(context.get(), q_not_in_collision);
  ASSERT_TRUE(dut.IsPostureCollisionFree(*context));

  // First generate a region C * t <= d.
  Eigen::Matrix<double, 24, 7> C;
  Eigen::Matrix<double, 24, 1> d;
  // I create matrix C with arbitrary values, such that C * t is a small
  // polytope surrounding q_not_in_collision.
  // clang-format off
  C << 1, 0, 0, 0, 2, 0, 0,
       -1, 0, 0, 0, 0, 1, 0,
       0, 1, 1, 0, 0, 0, 1,
       0, -1, -2, 0, 0, -1, 0,
       1, 1, 0, 2, 0, 0, 1,
       1, 0, 2, -1, 0, 1, 0,
       0, -1, 2, -2, 1, 3, 2,
       0, 1, -2, 1, 2, 4, 3,
       0, 3, -2, 2, 0, 1, -1,
       1, 0, 3, 2, 0, -1, 1,
       0, 1, -1, -2, 3, -2, 1,
       1, 0, -1, 1, 3, 2, 0,
       -1, -0.1, -0.2, 0, 0.3, 0.1, 0.1,
       -2, 0.1, 0.2, 0.2, -0.3, -0.1, 0.1,
       -1, 1, 1, 0, -1, 1, 0,
       0, 0.2, 0.1, 0, -1, 0.1, 0,
       0.1, 2, 0.2, 0.1, -0.1, -0.2, 0.1,
       -0.1, -2, 0.1, 0.2, -0.15, -0.1, -0.1,
       0.3, 0.5, 0.1, 0.7, -0.4, 1.2, 3.1,
       -0.5, 0.3, 0.2, -0.5, 1.2, 0.7, -0.5,
       0.4, 0.6, 1.2, -0.3, -0.5, 1.2, -0.1,
       1.5, -0.1, 0.6, 1.5, 0.4, 2.1, 0.3,
       0.5, 1.5, 0.3, 0.2, 1.5, -0.1, 0.5,
       0.5, 0.2, -0.1, 1.2, -0.3, 1.1, -0.4;

  // clang-format on
  // Now I take some samples of t slightly away from q_not_in_collision. C * t
  // <= d contains all these samples.
  Eigen::Matrix<double, 7, 6> t_samples;
  t_samples.col(0) = ((q_not_in_collision - q_star) / 2).array().tan();
  t_samples.col(1) =
      t_samples.col(0) +
      (Eigen::VectorXd(7) << 0.11, -0.02, 0.03, 0.01, 0, 0.02, 0.02).finished();
  t_samples.col(2) = t_samples.col(0) + (Eigen::VectorXd(7) << -0.005, 0.01,
                                         -0.02, 0.01, 0.005, 0.01, -0.02)
                                            .finished();
  t_samples.col(3) = t_samples.col(0) + (Eigen::VectorXd(7) << 0.02, -0.13,
                                         0.01, 0.02, -0.03, 0.01, 0.15)
                                            .finished();
  t_samples.col(4) = t_samples.col(0) + (Eigen::VectorXd(7) << 0.01, -0.04,
                                         0.003, 0.01, -0.01, -0.11, -0.08)
                                            .finished();
  t_samples.col(5) = t_samples.col(0) + (Eigen::VectorXd(7) << -0.01, -0.02,
                                         0.013, -0.02, 0.03, -0.03, -0.1)
                                            .finished();
  d = (C * t_samples).rowwise().maxCoeff();
  ASSERT_TRUE(IsPolyhedronBounded(C, d));

  ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairs
      filtered_collision_pairs{};
  auto clock_start = std::chrono::system_clock::now();
  const auto rationals = dut.GenerateLinkOnOneSideOfPlaneRationals(
      q_star, filtered_collision_pairs, q_not_in_collision);
  auto ret = dut.ConstructProgramToVerifyCollisionFreePolytope(
      rationals, C, d, filtered_collision_pairs);
  auto clock_now = std::chrono::system_clock::now();
  std::cout << "Elapsed Time: "
            << static_cast<float>(
                   std::chrono::duration_cast<std::chrono::milliseconds>(
                       clock_now - clock_start)
                       .count()) /
                   1000
            << "s\n";
  // First make sure that the lagrangians and verified polynomial has the right
  // size.
  EXPECT_EQ(ret.lagrangians.size(), rationals.size());
  EXPECT_EQ(ret.verified_polynomials.size(), rationals.size());
  for (const auto& lagrangian : ret.lagrangians) {
    EXPECT_EQ(lagrangian.rows(), C.rows());
    for (int i = 0; i < lagrangian.rows(); ++i) {
      for (int j = 0; j < dut.rational_forward_kinematics().t().rows(); ++j) {
        EXPECT_LE(
            lagrangian(i).Degree(dut.rational_forward_kinematics().t()(j)), 2);
      }
    }
  }
  // Make sure that each term in verified_polynomial has at most degree 3 for
  // each t, and at most one t has degree 3.
  for (const auto& verified_poly : ret.verified_polynomials) {
    for (const auto& [monomial, coeff] :
         verified_poly.monomial_to_coefficient_map()) {
      int degree_3_count = 0;
      for (int i = 0; i < dut.rational_forward_kinematics().t().rows(); ++i) {
        const int t_degree =
            monomial.degree(dut.rational_forward_kinematics().t()(i));
        EXPECT_LE(t_degree, 3);
        if (t_degree == 3) {
          degree_3_count++;
        }
      }
      EXPECT_LE(degree_3_count, 1);
    }
  }
  // Now check if ret.verified_polynomials is correct.
  VectorX<symbolic::Polynomial> d_minus_Ct(d.rows());
  for (int i = 0; i < d_minus_Ct.rows(); ++i) {
    d_minus_Ct(i) = symbolic::Polynomial(
        d(i) - C.row(i).dot(dut.rational_forward_kinematics().t()),
        symbolic::Variables(dut.rational_forward_kinematics().t()));
  }
  for (int i = 0; i < static_cast<int>(rationals.size()); ++i) {
    symbolic::Polynomial eval_expected = rationals[i].rational.numerator();
    for (int j = 0; j < C.rows(); ++j) {
      eval_expected -= ret.lagrangians[i](j) * d_minus_Ct(j);
    }
    const symbolic::Polynomial eval = ret.verified_polynomials[i];
    EXPECT_TRUE(eval.CoefficientsAlmostEqual(eval_expected, 1E-10));
  }
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result = solvers::Solve(*(ret.prog), std::nullopt, solver_options);
  EXPECT_TRUE(result.is_success());
}

}  // namespace multibody
}  // namespace drake
