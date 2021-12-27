#include "drake/multibody/rational_forward_kinematics/cspace_free_region.h"

#include <chrono>
#include <limits>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/symbolic_test_util.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities2.h"
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
class IiwaCspaceTest : public IiwaTest {
 public:
  IiwaCspaceTest() {
    // Arbitrarily add some polytopes to links
    link7_polytopes_.emplace_back(new const ConvexPolytope(
        iiwa_link_[7],
        GenerateBoxVertices(Eigen::Vector3d(0.1, 0.1, 0.2), {})));
    const RigidTransformd X_7P{RotationMatrixd(Eigen::AngleAxisd(
                                   0.2 * M_PI, Eigen::Vector3d::UnitX())),
                               {0.1, 0.2, -0.1}};
    link7_polytopes_.emplace_back(new const ConvexPolytope(
        iiwa_link_[7],
        GenerateBoxVertices(Eigen::Vector3d(0.1, 0.2, 0.1), X_7P)));

    const RigidTransformd X_5P{X_7P.rotation(), {-0.2, 0.1, 0}};
    link5_polytopes_.emplace_back(new const ConvexPolytope(
        iiwa_link_[5],
        GenerateBoxVertices(Eigen::Vector3d(0.2, 0.1, 0.2), X_5P)));

    RigidTransformd X_WP = X_5P * Eigen::Translation3d(0.15, -0.1, 0.05);
    obstacles_.emplace_back(new const ConvexPolytope(
        world_, GenerateBoxVertices(Eigen::Vector3d(0.1, 0.2, 0.15), X_WP)));
    X_WP = X_WP * RigidTransformd(RotationMatrixd(Eigen::AngleAxisd(
                      -0.1 * M_PI, Eigen::Vector3d::UnitY())));
    obstacles_.emplace_back(new const ConvexPolytope(
        world_, GenerateBoxVertices(Eigen::Vector3d(0.1, 0.25, 0.15), X_WP)));
  }

 protected:
  std::vector<std::unique_ptr<const ConvexPolytope>> link7_polytopes_;
  std::vector<std::unique_ptr<const ConvexPolytope>> link5_polytopes_;
  std::vector<std::unique_ptr<const ConvexPolytope>> obstacles_;
};

// Checks if p is an affine polynomial of x, namely p = a * x + b.
void CheckIsAffinePolynomial(
    const symbolic::Polynomial& p,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const symbolic::Variables& decision_vars) {
  EXPECT_EQ(p.TotalDegree(), 1);
  EXPECT_EQ(p.monomial_to_coefficient_map().size(), x.rows() + 1);
  for (int i = 0; i < x.rows(); ++i) {
    EXPECT_EQ(p.Degree(x(i)), 1);
  }
  for (const auto& decision_var : p.decision_variables()) {
    EXPECT_TRUE(decision_vars.find(decision_var) != decision_vars.end());
  }
}
void TestCspaceFreeRegionConstructor(
    const multibody::MultibodyPlant<double>& plant,
    const std::vector<const ConvexPolytope*>& link_polytopes,
    const std::vector<const ConvexPolytope*>& obstacles,
    SeparatingPlaneOrder plane_order, CspaceRegionType cspace_region_type) {
  const CspaceFreeRegion dut(plant, link_polytopes, obstacles, plane_order,
                             cspace_region_type);
  EXPECT_EQ(dut.separating_planes().size(),
            link_polytopes.size() * obstacles.size());
  int link_polytopes_count = 0;
  for (const auto& [link, polytopes_on_link] : dut.link_polytopes()) {
    link_polytopes_count += polytopes_on_link.size();
    for (const auto& polytope : polytopes_on_link) {
      EXPECT_EQ(polytope->body_index(), link);
    }
  }
  EXPECT_EQ(link_polytopes_count, link_polytopes.size());
  // Now check the separating planes.
  int separating_plane_count = 0;
  for (const auto& obstacle : obstacles) {
    for (const auto& [link, polytopes_on_link] : dut.link_polytopes()) {
      for (const auto& polytope : polytopes_on_link) {
        const auto& separating_plane =
            dut.separating_planes()[separating_plane_count];
        EXPECT_TRUE(separating_plane.positive_side_polytope->get_id() ==
                        polytope->get_id() ||
                    separating_plane.positive_side_polytope->get_id() ==
                        obstacle->get_id());
        EXPECT_TRUE(separating_plane.negative_side_polytope->get_id() ==
                        polytope->get_id() ||
                    separating_plane.negative_side_polytope->get_id() ==
                        obstacle->get_id());
        EXPECT_NE(separating_plane.positive_side_polytope->get_id(),
                  separating_plane.negative_side_polytope->get_id());
        const auto& a = separating_plane.a;
        const auto& b = dut.separating_planes()[separating_plane_count].b;
        const symbolic::Variables t_vars(dut.rational_forward_kinematics().t());
        if (plane_order == SeparatingPlaneOrder::kConstant) {
          for (int i = 0; i < 3; ++i) {
            const symbolic::Polynomial a_poly(a(i), t_vars);
            EXPECT_EQ(a_poly.TotalDegree(), 0);
          }
          EXPECT_EQ(symbolic::Polynomial(b, t_vars).TotalDegree(), 0);
        } else if (plane_order == SeparatingPlaneOrder::kAffine) {
          VectorX<symbolic::Variable> t_for_plane;
          if (cspace_region_type == CspaceRegionType::kGenericPolytope) {
            t_for_plane = dut.rational_forward_kinematics().t();
          } else {
            t_for_plane = dut.rational_forward_kinematics().FindTOnPath(
                link, plant.world_body().index());
          }
          // Check if a, b are affine function of t_for_plane.
          const symbolic::Variables decision_vars(
              separating_plane.decision_variables);
          EXPECT_EQ(decision_vars.size(), 4 * t_for_plane.rows() + 4);
          CheckIsAffinePolynomial(symbolic::Polynomial(b, t_vars), t_for_plane,
                                  decision_vars);
          for (int i = 0; i < 3; ++i) {
            CheckIsAffinePolynomial(symbolic::Polynomial(a(i), t_vars),
                                    t_for_plane, decision_vars);
          }
        }
        separating_plane_count++;
      }
    }
  }
  // Check map_polytopes_to_separating_planes().
  EXPECT_EQ(dut.map_polytopes_to_separating_planes().size(),
            dut.separating_planes().size());
  for (const auto& separating_plane : dut.separating_planes()) {
    const SortedPair<ConvexGeometry::Id> polytope_pair(
        separating_plane.positive_side_polytope->get_id(),
        separating_plane.negative_side_polytope->get_id());
    const auto it =
        dut.map_polytopes_to_separating_planes().find(polytope_pair);
    EXPECT_NE(it, dut.map_polytopes_to_separating_planes().end());
    EXPECT_EQ(it->second->positive_side_polytope->get_id(),
              separating_plane.positive_side_polytope->get_id());
    EXPECT_EQ(it->second->negative_side_polytope->get_id(),
              separating_plane.negative_side_polytope->get_id());
  }
}

TEST_F(IiwaCspaceTest, TestConstructor) {
  TestCspaceFreeRegionConstructor(
      *iiwa_, {link7_polytopes_[0].get()}, {obstacles_[0].get()},
      SeparatingPlaneOrder::kConstant, CspaceRegionType::kGenericPolytope);
  TestCspaceFreeRegionConstructor(
      *iiwa_, {link7_polytopes_[0].get(), link7_polytopes_[1].get()},
      {obstacles_[0].get(), obstacles_[1].get()}, SeparatingPlaneOrder::kAffine,
      CspaceRegionType::kGenericPolytope);
  TestCspaceFreeRegionConstructor(
      *iiwa_, {link7_polytopes_[0].get(), link7_polytopes_[1].get()},
      {obstacles_[0].get(), obstacles_[1].get()}, SeparatingPlaneOrder::kAffine,
      CspaceRegionType::kAxisAlignedBoundingBox);
  // Link poltyope not on the end-effector. For C-space generic polytope,
  // t_for_plane should still be all t, but for axis-aligned bounding box,
  // t_for_plane should only contain t on the kinematics chain between the link
  // polytope and the obstacle.
  TestCspaceFreeRegionConstructor(
      *iiwa_, {link7_polytopes_[0].get(), link5_polytopes_[0].get()},
      {obstacles_[0].get(), obstacles_[1].get()}, SeparatingPlaneOrder::kAffine,
      CspaceRegionType::kAxisAlignedBoundingBox);
  TestCspaceFreeRegionConstructor(
      *iiwa_, {link7_polytopes_[0].get(), link5_polytopes_[0].get()},
      {obstacles_[0].get(), obstacles_[1].get()}, SeparatingPlaneOrder::kAffine,
      CspaceRegionType::kGenericPolytope);
}

void TestGenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    const SeparatingPlane& separating_plane, PlaneSide plane_side,
    const Eigen::Ref<const Eigen::VectorXd>& q_star) {
  const ConvexPolytope* link_polytope;
  const ConvexPolytope* other_side_polytope;
  if (plane_side == PlaneSide::kPositive) {
    link_polytope = separating_plane.positive_side_polytope;
    other_side_polytope = separating_plane.negative_side_polytope;
  } else {
    link_polytope = separating_plane.negative_side_polytope;
    other_side_polytope = separating_plane.positive_side_polytope;
  }
  const auto X_AB_multilinear =
      rational_forward_kinematics.CalcLinkPoseAsMultilinearPolynomial(
          q_star, link_polytope->body_index(), separating_plane.expressed_link);

  const auto rationals = GenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link_polytope, other_side_polytope,
      X_AB_multilinear, separating_plane.a, separating_plane.b, plane_side,
      separating_plane.order);
  EXPECT_EQ(rationals.size(), link_polytope->p_BV().cols());
  for (const auto& rational : rationals) {
    EXPECT_EQ(rational.link_polytope->get_id(), link_polytope->get_id());
    EXPECT_EQ(rational.other_side_link_polytope->get_id(),
              other_side_polytope->get_id());
  }
  // Now take many samples of q, evaluate a.dot(x) + b - 1 or -1 - a.dot(x) - b
  // for these sampled q.
  std::vector<Eigen::VectorXd> q_samples;
  q_samples.push_back(
      q_star +
      (Eigen::VectorXd(7) << 0.1, 0.2, -0.1, -0.3, 1.2, 0.5, 0.1).finished());
  q_samples.push_back(
      q_star +
      (Eigen::VectorXd(7) << 0.3, -0.4, -0.8, -0.3, 1.1, -0.5, 0.4).finished());
  q_samples.push_back(
      q_star + (Eigen::VectorXd(7) << -0.3, -0.7, -1.2, -0.9, 1.3, -0.7, 0.3)
                   .finished());
  symbolic::Environment env;
  // Set the plane decision variables to arbitrary values.
  const Eigen::VectorXd plane_decision_var_vals = Eigen::VectorXd::LinSpaced(
      separating_plane.decision_variables.rows(), -2, 3);
  env.insert(separating_plane.decision_variables, plane_decision_var_vals);
  const auto& plant = rational_forward_kinematics.plant();
  auto context = plant.CreateDefaultContext();
  for (const auto& q : q_samples) {
    plant.SetPositions(context.get(), q);
    Eigen::Matrix3Xd p_AV(3, link_polytope->p_BV().cols());
    plant.CalcPointsPositions(
        *context, plant.get_body(link_polytope->body_index()).body_frame(),
        link_polytope->p_BV(),
        plant.get_body(separating_plane.expressed_link).body_frame(), &p_AV);
    const Eigen::VectorXd t_val = ((q - q_star) / 2).array().tan();
    for (int i = 0; i < t_val.rows(); ++i) {
      auto it = env.find(rational_forward_kinematics.t()(i));
      if (it == env.end()) {
        env.insert(rational_forward_kinematics.t()(i), t_val(i));
      } else {
        it->second = t_val(i);
      }
    }

    for (int i = 0; i < static_cast<int>(rationals.size()); ++i) {
      const double rational_val = rationals[i].rational.Evaluate(env);
      // Now evaluate this rational function.
      Eigen::Vector3d a_val;
      for (int j = 0; j < 3; ++j) {
        a_val(j) = separating_plane.a(j).Evaluate(env);
      }
      const double b_val = separating_plane.b.Evaluate(env);
      const double rational_val_expected =
          plane_side == PlaneSide::kPositive
              ? a_val.dot(p_AV.col(i)) + b_val - 1
              : -1 - a_val.dot(p_AV.col(i)) - b_val;
      EXPECT_NEAR(rational_val, rational_val_expected, 1E-12);
    }
  }
}

TEST_F(IiwaCspaceTest, GenerateLinkOnOneSideOfPlaneRationalFunction1) {
  const CspaceFreeRegion dut(
      *iiwa_, {link7_polytopes_[0].get()}, {obstacles_[0].get()},
      SeparatingPlaneOrder::kAffine, CspaceRegionType::kGenericPolytope);
  const Eigen::VectorXd q_star1 = Eigen::VectorXd::Zero(7);
  const Eigen::VectorXd q_star2 =
      (Eigen::VectorXd(7) << 0.1, 0.2, -0.1, 0.3, 0.2, 0.4, 0.2).finished();

  const auto& separating_plane = dut.separating_planes()[0];
  for (const auto plane_side : {PlaneSide::kPositive, PlaneSide::kNegative}) {
    TestGenerateLinkOnOneSideOfPlaneRationalFunction(
        dut.rational_forward_kinematics(), separating_plane, plane_side,
        q_star1);
    TestGenerateLinkOnOneSideOfPlaneRationalFunction(
        dut.rational_forward_kinematics(), separating_plane, plane_side,
        q_star2);
  }
}

void TestGenerateLinkOnOneSideOfPlaneRationals(
    const CspaceFreeRegion& dut,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const CspaceFreeRegion::FilteredCollisionPairs& filtered_collision_pairs) {
  const auto rationals = dut.GenerateLinkOnOneSideOfPlaneRationals(
      q_star, filtered_collision_pairs);
  // Check the size of rationals.
  int rationals_size = 0;
  for (const auto& [link_pair, separating_plane] :
       dut.map_polytopes_to_separating_planes()) {
    if (!IsGeometryPairCollisionIgnored(link_pair.first(), link_pair.second(),
                                        filtered_collision_pairs)) {
      rationals_size +=
          separating_plane->positive_side_polytope->p_BV().cols() +
          separating_plane->negative_side_polytope->p_BV().cols();
    }
  }
  EXPECT_EQ(rationals.size(), rationals_size);
}

TEST_F(IiwaCspaceTest, GenerateLinkOnOneSideOfPlaneRationals) {
  const CspaceFreeRegion dut1(
      *iiwa_, {link7_polytopes_[0].get()}, {obstacles_[0].get()},
      SeparatingPlaneOrder::kAffine, CspaceRegionType::kGenericPolytope);
  const Eigen::VectorXd q_star = Eigen::VectorXd::Zero(7);
  TestGenerateLinkOnOneSideOfPlaneRationals(dut1, q_star, {});

  // Multiple pairs of polytopes.
  const CspaceFreeRegion dut2(
      *iiwa_, {link7_polytopes_[0].get(), link5_polytopes_[0].get()},
      {obstacles_[0].get(), obstacles_[1].get()}, SeparatingPlaneOrder::kAffine,
      CspaceRegionType::kGenericPolytope);
  TestGenerateLinkOnOneSideOfPlaneRationals(dut2, q_star, {});
  // Now test with filtered collision pairs.
  const CspaceFreeRegion::FilteredCollisionPairs filtered_collision_pairs{
      {{link7_polytopes_[0]->get_id(), obstacles_[0]->get_id()}}};
  TestGenerateLinkOnOneSideOfPlaneRationals(dut2, q_star,
                                            filtered_collision_pairs);
}
}  // namespace multibody
}  // namespace drake
