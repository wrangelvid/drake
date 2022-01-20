#include "drake/geometry/optimization/hpolyhedron.h"

#include <limits>

#include <gtest/gtest.h>

#include "drake/common/eigen_types.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/geometry_frame.h"
#include "drake/geometry/optimization/test_utilities.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/random_rotation.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace geometry {
namespace optimization {

using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using internal::CheckAddPointInSetConstraints;
using internal::MakeSceneGraphWithShape;
using math::RigidTransformd;
using math::RotationMatrixd;
using solvers::Binding;
using solvers::Constraint;
using solvers::MathematicalProgram;

GTEST_TEST(HPolyhedronTest, UnitBoxTest) {
  Matrix<double, 6, 3> A;
  A << Matrix3d::Identity(), -Matrix3d::Identity();
  Vector6d b = Vector6d::Ones();

  // Test constructor.
  HPolyhedron H(A, b);
  EXPECT_EQ(H.ambient_dimension(), 3);
  EXPECT_TRUE(CompareMatrices(A, H.A()));
  EXPECT_TRUE(CompareMatrices(b, H.b()));

  // Test MakeUnitBox method.
  HPolyhedron Hbox = HPolyhedron::MakeUnitBox(3);
  EXPECT_EQ(Hbox.ambient_dimension(), 3);
  EXPECT_TRUE(CompareMatrices(A, Hbox.A()));
  EXPECT_TRUE(CompareMatrices(b, Hbox.b()));

  // Test PointInSet.
  EXPECT_TRUE(H.PointInSet(Vector3d(.8, .3, -.9)));
  EXPECT_TRUE(H.PointInSet(Vector3d(-1.0, 1.0, 1.0)));
  EXPECT_FALSE(H.PointInSet(Vector3d(1.1, 1.2, 0.4)));

  // Test AddPointInSetConstraints.
  EXPECT_TRUE(CheckAddPointInSetConstraints(H, Vector3d(.8, .3, -.9)));
  EXPECT_TRUE(CheckAddPointInSetConstraints(H, Vector3d(-1.0, 1.0, 1.0)));
  EXPECT_FALSE(CheckAddPointInSetConstraints(H, Vector3d(1.1, 1.2, 0.4)));

  // Test SceneGraph constructor.
  auto [scene_graph, geom_id] =
      MakeSceneGraphWithShape(Box(2.0, 2.0, 2.0), RigidTransformd::Identity());
  auto context = scene_graph->CreateDefaultContext();
  auto query =
      scene_graph->get_query_output_port().Eval<QueryObject<double>>(*context);

  HPolyhedron H_scene_graph(query, geom_id);
  EXPECT_TRUE(CompareMatrices(A, H_scene_graph.A()));
  EXPECT_TRUE(CompareMatrices(b, H_scene_graph.b()));
}

GTEST_TEST(HPolyhedronTest, ArbitraryBoxTest) {
  RigidTransformd X_WG(RotationMatrixd::MakeZRotation(M_PI / 2.0),
                       Vector3d(-4.0, -5.0, -6.0));
  auto [scene_graph, geom_id] =
      MakeSceneGraphWithShape(Box(1.0, 2.0, 3.0), X_WG);
  auto context = scene_graph->CreateDefaultContext();
  auto query =
      scene_graph->get_query_output_port().Eval<QueryObject<double>>(*context);
  HPolyhedron H(query, geom_id);

  EXPECT_EQ(H.ambient_dimension(), 3);
  // Rotated box should end up with lb=[-5,-5.5,-7.5], ub=[-3,-4.5,-4.5].
  Vector3d in1_W{-4.9, -5.4, -7.4}, in2_W{-3.1, -4.6, -4.6},
      out1_W{-5.1, -5.6, -7.6}, out2_W{-2.9, -4.4, -4.4};

  EXPECT_LE(query.ComputeSignedDistanceToPoint(in1_W)[0].distance, 0.0);
  EXPECT_LE(query.ComputeSignedDistanceToPoint(in2_W)[0].distance, 0.0);
  EXPECT_GE(query.ComputeSignedDistanceToPoint(out1_W)[0].distance, 0.0);
  EXPECT_GE(query.ComputeSignedDistanceToPoint(out2_W)[0].distance, 0.0);

  EXPECT_TRUE(H.PointInSet(in1_W));
  EXPECT_TRUE(H.PointInSet(in2_W));
  EXPECT_FALSE(H.PointInSet(out1_W));
  EXPECT_FALSE(H.PointInSet(out2_W));

  EXPECT_TRUE(CheckAddPointInSetConstraints(H, in1_W));
  EXPECT_TRUE(CheckAddPointInSetConstraints(H, in2_W));
  EXPECT_FALSE(CheckAddPointInSetConstraints(H, out1_W));
  EXPECT_FALSE(CheckAddPointInSetConstraints(H, out2_W));

  // Test reference_frame frame.
  SourceId source_id = scene_graph->RegisterSource("F");
  FrameId frame_id = scene_graph->RegisterFrame(source_id, GeometryFrame("F"));
  auto context2 = scene_graph->CreateDefaultContext();
  const RigidTransformd X_WF{math::RollPitchYawd(.1, .2, 3),
                             Vector3d{.5, .87, .1}};
  const FramePoseVector<double> pose_vector{{frame_id, X_WF}};
  scene_graph->get_source_pose_port(source_id).FixValue(context2.get(),
                                                        pose_vector);
  auto query2 =
      scene_graph->get_query_output_port().Eval<QueryObject<double>>(*context2);
  HPolyhedron H_F(query2, geom_id, frame_id);

  const RigidTransformd X_FW = X_WF.inverse();
  EXPECT_TRUE(H_F.PointInSet(X_FW * in1_W));
  EXPECT_TRUE(H_F.PointInSet(X_FW * in2_W));
  EXPECT_FALSE(H_F.PointInSet(X_FW * out1_W));
  EXPECT_FALSE(H_F.PointInSet(X_FW * out2_W));
}

GTEST_TEST(HPolyhedronTest, HalfSpaceTest) {
  RigidTransformd X_WG(RotationMatrixd::MakeYRotation(M_PI / 2.0),
                       Vector3d(-1.2, -2.1, -6.4));
  auto [scene_graph, geom_id] = MakeSceneGraphWithShape(HalfSpace(), X_WG);
  auto context = scene_graph->CreateDefaultContext();
  auto query =
      scene_graph->get_query_output_port().Eval<QueryObject<double>>(*context);
  HPolyhedron H(query, geom_id);

  EXPECT_EQ(H.ambient_dimension(), 3);

  // Rotated HalfSpace should be x <= -1.2.
  Vector3d in1_W{-1.21, 0.0, 0.0}, in2_W{-1.21, 2., 3.}, out1_W{-1.19, 0, 0},
      out2_W{-1.19, 2., 3.};

  EXPECT_LE(query.ComputeSignedDistanceToPoint(in1_W)[0].distance, 0.0);
  EXPECT_LE(query.ComputeSignedDistanceToPoint(in2_W)[0].distance, 0.0);
  EXPECT_GE(query.ComputeSignedDistanceToPoint(out1_W)[0].distance, 0.0);
  EXPECT_GE(query.ComputeSignedDistanceToPoint(out2_W)[0].distance, 0.0);

  EXPECT_TRUE(H.PointInSet(in1_W));
  EXPECT_TRUE(H.PointInSet(in2_W));
  EXPECT_FALSE(H.PointInSet(out1_W));
  EXPECT_FALSE(H.PointInSet(out2_W));
}

GTEST_TEST(HPolyhedronTest, UnitBox6DTest) {
  HPolyhedron H = HPolyhedron::MakeUnitBox(6);
  EXPECT_EQ(H.ambient_dimension(), 6);

  Vector6d in1_W{Vector6d::Constant(-.99)}, in2_W{Vector6d::Constant(.99)},
      out1_W{Vector6d::Constant(-1.01)}, out2_W{Vector6d::Constant(1.01)};

  EXPECT_TRUE(H.PointInSet(in1_W));
  EXPECT_TRUE(H.PointInSet(in2_W));
  EXPECT_FALSE(H.PointInSet(out1_W));
  EXPECT_FALSE(H.PointInSet(out2_W));
}

GTEST_TEST(HPolyhedronTest, InscribedEllipsoidTest) {
  // Test a unit box.
  HPolyhedron H = HPolyhedron::MakeUnitBox(3);
  Hyperellipsoid E = H.MaximumVolumeInscribedEllipsoid();
  // The exact tolerance will be solver dependent; this is (hopefully)
  // conservative enough.
  const double kTol = 1e-4;
  EXPECT_TRUE(CompareMatrices(E.center(), Vector3d::Zero(), kTol));
  EXPECT_TRUE(CompareMatrices(E.A().transpose() * E.A(),
                              Matrix3d::Identity(3, 3), kTol));

  // A non-trivial example, taken some real problem data.  The addition of the
  // extra half-plane constraints cause the optimal ellipsoid to be far from
  // axis-aligned.
  Matrix<double, 8, 3> A;
  Matrix<double, 8, 1> b;
  // clang-format off
  A << Matrix3d::Identity(),
       -Matrix3d::Identity(),
       .9, -.3, .1,
       .9, -.3, .1;
  b << 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 1.3, 0.8;
  // clang-format on
  HPolyhedron H2(A, b);
  Hyperellipsoid E2 = H2.MaximumVolumeInscribedEllipsoid();
  // Check that points just inside the boundary of the ellipsoid are inside the
  // polytope.
  Matrix3d C = E2.A().inverse();
  RandomGenerator generator;
  for (int i = 0; i < 10; ++i) {
    const RotationMatrixd R = math::UniformlyRandomRotationMatrix(&generator);
    SCOPED_TRACE(fmt::format("With random rotation matrix\n{}", R.matrix()));
    Vector3d x = C * R.matrix() * Vector3d(0.99, 0.0, 0.0) + E2.center();
    EXPECT_TRUE(E2.PointInSet(x));
    EXPECT_TRUE(H2.PointInSet(x));
  }

  // Make sure the ellipsoid touches the polytope, by checking that the minimum
  // residual, bᵢ − aᵢd − |aᵢC|₂, is zero.
  const VectorXd polytope_halfspace_residue =
      b - A * E2.center() - ((A * C).rowwise().lpNorm<2>());
  EXPECT_NEAR(polytope_halfspace_residue.minCoeff(), 0, kTol);
}

GTEST_TEST(HPolyhedronTest, ChebyshevCenter) {
  HPolyhedron box = HPolyhedron::MakeUnitBox(6);
  EXPECT_TRUE(CompareMatrices(box.ChebyshevCenter(), Vector6d::Zero(), 1e-6));
}

// A rotated long thin rectangle in 2 dimensions.
GTEST_TEST(HPolyhedronTest, ChebyshevCenter2) {
  Matrix<double, 4, 2> A;
  Vector4d b;
  // clang-format off
  A << -2, -1,  // 2x + y ≥ 4
        2,  1,  // 2x + y ≤ 6
       -1,  2,  // x - 2y ≥ 2
        1, -2;  // x - 2y ≤ 8
  b << -4, 6, -2, 8;
  // clang-format on
  HPolyhedron H(A, b);
  const VectorXd center = H.ChebyshevCenter();
  EXPECT_TRUE(H.PointInSet(center));
  // For the rectangle, the center should have distance = 1.0 from the first
  // two half-planes, and ≥ 1.0 for the other two.
  const VectorXd distance = b - A*center;
  EXPECT_NEAR(distance[0], 1.0, 1e-6);
  EXPECT_NEAR(distance[1], 1.0, 1e-6);
  EXPECT_GE(distance[2], 1.0 - 1e-6);
  EXPECT_GE(distance[3], 1.0 - 1e-6);
}

GTEST_TEST(HPolyhedronTest, CloneTest) {
  HPolyhedron H = HPolyhedron::MakeBox(Vector3d{-3, -4, -5}, Vector3d{6, 7, 8});
  std::unique_ptr<ConvexSet> clone = H.Clone();
  EXPECT_EQ(clone->ambient_dimension(), H.ambient_dimension());
  HPolyhedron* pointer = dynamic_cast<HPolyhedron*>(clone.get());
  ASSERT_NE(pointer, nullptr);
  EXPECT_TRUE(CompareMatrices(H.A(), pointer->A()));
  EXPECT_TRUE(CompareMatrices(H.b(), pointer->b()));
}

GTEST_TEST(HPolyhedronTest, NonnegativeScalingTest) {
  const Vector3d lb{1, 1, 1}, ub{2, 3, 4};
  HPolyhedron H = HPolyhedron::MakeBox(lb, ub);

  MathematicalProgram prog;
  auto x = prog.NewContinuousVariables(3, "x");
  auto t = prog.NewContinuousVariables(1, "t")[0];

  std::vector<Binding<Constraint>> constraints =
      H.AddPointInNonnegativeScalingConstraints(&prog, x, t);

  EXPECT_EQ(constraints.size(), 2);

  prog.SetInitialGuess(x, .99 * ub);
  prog.SetInitialGuess(t, 1.0);
  EXPECT_TRUE(prog.CheckSatisfiedAtInitialGuess(constraints, 0));

  prog.SetInitialGuess(x, 1.01 * ub);
  prog.SetInitialGuess(t, 1.0);
  EXPECT_FALSE(prog.CheckSatisfiedAtInitialGuess(constraints, 0));

  prog.SetInitialGuess(x, .99 * ub);
  prog.SetInitialGuess(t, -0.01);
  EXPECT_FALSE(prog.CheckSatisfiedAtInitialGuess(constraints, 0));

  prog.SetInitialGuess(x, .49 * ub);
  prog.SetInitialGuess(t, 0.5);
  EXPECT_TRUE(prog.CheckSatisfiedAtInitialGuess(constraints, 0));

  prog.SetInitialGuess(x, .51 * ub);
  prog.SetInitialGuess(t, 0.5);
  EXPECT_FALSE(prog.CheckSatisfiedAtInitialGuess(constraints, 0));

  prog.SetInitialGuess(x, 1.99 * ub);
  prog.SetInitialGuess(t, 2.0);
  EXPECT_TRUE(prog.CheckSatisfiedAtInitialGuess(constraints, 0));

  prog.SetInitialGuess(x, 2.01 * ub);
  prog.SetInitialGuess(t, 2.0);
  EXPECT_FALSE(prog.CheckSatisfiedAtInitialGuess(constraints, 0));
}

GTEST_TEST(HPolyhedronTest, IsBounded) {
  Vector4d lb, ub;
  lb << -1, -3, -5, -2;
  ub << 2, 4, 5.4, 3;
  HPolyhedron H = HPolyhedron::MakeBox(lb, ub);
  EXPECT_TRUE(H.IsBounded());
}

GTEST_TEST(HPolyhedronTest, IsBounded2) {
  // Box with zero volume.
  const Vector2d lb{1, -3}, ub{1, 3};
  HPolyhedron H = HPolyhedron::MakeBox(lb, ub);
  EXPECT_TRUE(H.IsBounded());
}

GTEST_TEST(HPolyhedronTest, IsBounded3) {
  // Unbounded (2 inequalities in 3 dimensions).
  HPolyhedron H(MatrixXd::Identity(2, 3), Vector2d::Ones());
  EXPECT_FALSE(H.IsBounded());
}

GTEST_TEST(HPolyhedronTest, IsBounded4) {
  // Unbounded (A is low rank).
  Matrix3d A;
  // clang-format off
  A << 1, 2, 3,
       1, 2, 3,
       0, 0, 1;
  // clang-format on
  HPolyhedron H(A, Vector3d::Ones());
  EXPECT_FALSE(H.IsBounded());
}

GTEST_TEST(HPolyhedronTest, CartesianPowerTest) {
  // First test the concept. If x ∈ H, then [x; x]  ∈ H x H and
  // [x; x; x]  ∈ H x H x H.
  MatrixXd A{4, 2};
  A << MatrixXd::Identity(2, 2), -MatrixXd::Identity(2, 2);
  VectorXd b = VectorXd::Ones(4);
  HPolyhedron H(A, b);
  VectorXd x = VectorXd::Zero(2);
  EXPECT_TRUE(H.PointInSet(x));
  EXPECT_TRUE(H.CartesianPower(2).PointInSet((VectorXd(4) << x, x).finished()));
  EXPECT_TRUE(
      H.CartesianPower(3).PointInSet((VectorXd(6) << x, x, x).finished()));

  // Now test the HPolyhedron-specific behavior.
  MatrixXd A_1{2, 3};
  MatrixXd A_2{4, 6};
  MatrixXd A_3{6, 9};
  VectorXd b_1{2};
  VectorXd b_2{4};
  VectorXd b_3{6};
  MatrixXd zero = MatrixXd::Zero(2, 3);
  // clang-format off
  A_1 << 1, 2, 3,
         4, 5, 6;
  b_1 << 1, 2;
  A_2 <<  A_1, zero,
         zero,  A_1;
  b_2 << b_1, b_1;
  A_3 <<  A_1, zero, zero,
         zero,  A_1, zero,
         zero, zero,  A_1;
  b_3 << b_1, b_1, b_1;
  // clang-format on
  HPolyhedron H_1(A_1, b_1);
  HPolyhedron H_2 = H_1.CartesianPower(2);
  HPolyhedron H_3 = H_1.CartesianPower(3);
  EXPECT_TRUE(CompareMatrices(H_2.A(), A_2));
  EXPECT_TRUE(CompareMatrices(H_2.b(), b_2));
  EXPECT_TRUE(CompareMatrices(H_3.A(), A_3));
  EXPECT_TRUE(CompareMatrices(H_3.b(), b_3));
}

GTEST_TEST(HPolyhedronTest, CartesianProductTest) {
  HPolyhedron H_A = HPolyhedron::MakeUnitBox(2);
  VectorXd x_A = VectorXd::Zero(2);
  EXPECT_TRUE(H_A.PointInSet(x_A));

  HPolyhedron H_B = HPolyhedron::MakeBox(Vector2d(2, 2), Vector2d(4, 4));
  VectorXd x_B = 3 * VectorXd::Ones(2);
  EXPECT_TRUE(H_B.PointInSet(x_B));

  HPolyhedron H_C = H_A.CartesianProduct(H_B);
  VectorXd x_C{x_A.size() + x_B.size()};
  x_C << x_A, x_B;
  EXPECT_TRUE(H_C.PointInSet(x_C));
}

GTEST_TEST(HPolyhedronTest, AxisAlignedContainment) {
  Eigen::Vector2d lower_limit = -Eigen::Vector2d::Ones();
  Eigen::Vector2d upper_limit = Eigen::Vector2d::Ones();
  double scale{0.25};

  HPolyhedron outer = HPolyhedron::MakeBox(lower_limit, upper_limit);
  HPolyhedron inner = HPolyhedron::MakeBox(scale*lower_limit, scale*upper_limit);

  EXPECT_TRUE(inner.ContainedInOtherHPolyhedron(outer));
  EXPECT_FALSE(outer.ContainedInOtherHPolyhedron(inner));
}

GTEST_TEST(HPolyhedronTest, L1BallContainedInInfinityBall3D) {
  Eigen::MatrixXd A_L1(8,3);
  Eigen::VectorXd b_L1 = Eigen::VectorXd::Ones(8);
  // clang-format off
  A_L1 <<  1, 1, 1,
           1, 1,-1,
           1,-1, 1,
           1,-1,-1,
          -1, 1, 1,
          -1, 1,-1,
          -1,-1, 1,
          -1,-1,-1;
  //clang-format on
  HPolyhedron L1_ball = HPolyhedron(A_L1, b_L1);

  Eigen::Vector3d lower_limit = -Eigen::Vector3d::Ones();
  Eigen::Vector3d upper_limit = Eigen::Vector3d::Ones();
  HPolyhedron Linfty_ball = HPolyhedron::MakeBox(lower_limit, upper_limit);


  EXPECT_TRUE(L1_ball.ContainedInOtherHPolyhedron(Linfty_ball));
  EXPECT_FALSE(Linfty_ball.ContainedInOtherHPolyhedron(L1_ball));
}

GTEST_TEST(HPolyhedronTest, L1BallIrredundantIntersectionInfinityBall3D) {
  Eigen::MatrixXd A_L1(8,3);
  Eigen::VectorXd b_L1 = Eigen::VectorXd::Ones(8);
  // clang-format off
  A_L1 <<  1, 1, 1,
           1, 1,-1,
           1,-1, 1,
           1,-1,-1,
          -1, 1, 1,
          -1, 1,-1,
          -1,-1, 1,
          -1,-1,-1;
  //clang-format on
  HPolyhedron L1_ball = HPolyhedron(A_L1, b_L1);

  Eigen::Vector3d lower_limit = -Eigen::Vector3d::Ones();
  Eigen::Vector3d upper_limit = Eigen::Vector3d::Ones();
  HPolyhedron Linfty_ball = HPolyhedron::MakeBox(lower_limit, upper_limit);

  HPolyhedron IntersectionBall = L1_ball.IrredundantIntersection(Linfty_ball);
  EXPECT_TRUE(CompareMatrices(A_L1, IntersectionBall.A()));
  EXPECT_TRUE(CompareMatrices(b_L1, IntersectionBall.b()));
}


GTEST_TEST(HPolyhedronTest, OffsetIrredundantBoxes) {
  Eigen::Vector2d left_box_lower = {-1,-1};
  Eigen::Vector2d left_box_upper = {0.25,1};
  HPolyhedron left_box = HPolyhedron::MakeBox(left_box_lower, left_box_upper);

  Eigen::Vector2d right_box_lower = {-0.25,-1};
  Eigen::Vector2d right_box_upper = {1,1};
  HPolyhedron right_box = HPolyhedron::MakeBox(right_box_lower, right_box_upper);

  HPolyhedron intersection_right_into_left = left_box.IrredundantIntersection(right_box);
  HPolyhedron intersection_left_into_right = right_box.IrredundantIntersection(left_box);

  Eigen::MatrixXd A_right_into_left_expected(5,2);
  Eigen::VectorXd b_right_into_left_expected(5);
  Eigen::MatrixXd A_left_into_right_expected(5,2);
  Eigen::VectorXd b_left_into_right_expected(5);

  A_right_into_left_expected.topRows(4) = left_box.A();
  b_right_into_left_expected.topRows(4) = left_box.b();
  A_left_into_right_expected.topRows(4) = right_box.A();
  b_left_into_right_expected.topRows(4) = right_box.b();

  A_right_into_left_expected.row(4) = right_box.A().row(2);
  b_right_into_left_expected.row(4) = right_box.b().row(2);

  A_left_into_right_expected.row(4) = left_box.A().row(0);
  b_left_into_right_expected.row(4) = left_box.b().row(0);

  EXPECT_TRUE(CompareMatrices(A_right_into_left_expected, intersection_right_into_left.A()));
  EXPECT_TRUE(CompareMatrices(b_right_into_left_expected, intersection_right_into_left.b()));

  EXPECT_TRUE(CompareMatrices(A_left_into_right_expected, intersection_left_into_right.A()));
  EXPECT_TRUE(CompareMatrices(b_left_into_right_expected, intersection_left_into_right.b()));


}

GTEST_TEST(HPolyhedronTest, IrredundantBallIntersectionContainedInBothOriginal) {
  Eigen::MatrixXd A0(36,3);
  Eigen::VectorXd b0(36);
  Eigen::MatrixXd A1(7,3);
  Eigen::VectorXd b1(7);
  //clang-format off
  A0 <<    1.       ,  0.       ,   0.       ,
           0.        , 1.        ,  0.        ,
           0.        , 0.        ,  1.        ,
          -1.        ,-0.        , -0.        ,
          -0.        ,-1.        , -0.        ,
          -0.        ,-0.        , -1.        ,
           0.14569377, 0.97734161, -0.15354705,
           0.14507821, 0.97918599, -0.14194054,
           0.13324741, 0.99019755,  0.04187999,
           0.13325049, 0.99019886,  0.0418393 ,
           0.13325357, 0.99020016,  0.0417986 ,
           0.13325665, 0.99020147,  0.04175789,
           0.13325973, 0.99020277,  0.04171717,
           0.17673294, 0.97152064, -0.15783889,
           0.17723733, 0.96946316, -0.16949369,
           0.17721183, 0.96957292, -0.16889145,
           0.17718487, 0.9696883 , -0.1682561 ,
           0.17715619, 0.96981032, -0.16758167,
           0.13834252, 0.69751068, -0.70309331,
           0.13851255, 0.71147702, -0.68892287,
           0.13751017, 0.85533306, -0.49949606,
           0.13751053, 0.85531603, -0.49952512,
           0.16696735, 0.67944639, -0.71447499,
           0.16685434, 0.69267701, -0.7016824 ,
           0.16228808, 0.83192935, -0.53061864,
           0.16228944, 0.83190782, -0.53065197,
           0.12214756, 0.58253193, -0.80357733,
           0.1228507 , 0.59772306, -0.79223409,
           0.12840434, 0.76631789, -0.62949918,
           0.12840368, 0.76628412, -0.62954042,
           0.12840302, 0.76625034, -0.62958167,
           0.15616761, 0.5642631 , -0.81069034,
           0.15653994, 0.57980132, -0.79957843,
           0.1571008 , 0.75279325, -0.63923522,
           0.15710151, 0.75275753, -0.63927711,
           0.15710222, 0.75272179, -0.63931901;
  b0 << 1.73205081,  1.73205081,  1.73205081, 1.73205081,  1.73205081,  1.73205081,
       -0.90505503, -0.89488313, -0.67312999,-0.67319184, -0.67325369, -0.67331556,
       -0.67337743, -0.90233114, -0.91194952,-0.91146518, -0.91095276, -0.9104072 ,
       -0.91818143, -0.92969365, -0.97994542,-0.9799523 , -0.89818648, -0.90957952,
       -0.97451322, -0.97451584, -0.8897345 ,-0.90448817, -1.02214108, -1.0221306 ,
       -1.02212011, -0.86783947, -0.88323654,-1.00963666, -1.00962362, -1.00961056;


  A1 <<   1.        ,  0.        ,  0.       ,
          0.        ,  1.        ,  0.       ,
          0.        ,  0.        ,  1.       ,
         -1.        , -0.        , -0.       ,
         -0.        , -1.        , -0.       ,
         -0.        , -0.        , -1.       ,
          0.93523194, -0.33785034,  0.1058223;

  b1 << 1.73205081, 1.73205081, 1.73205081, 1.73205081, 1.73205081, 1.73205081, 0.26346887;
  //clang-format on
  HPolyhedron P0(A0, b0);
  HPolyhedron P1(A1, b1);
  HPolyhedron IrredP0intoP1 = P1.IrredundantIntersection(P0);
  HPolyhedron IrredP1intoP0 = P0.IrredundantIntersection(P1);



  EXPECT_TRUE(IrredP0intoP1.ContainedInOtherHPolyhedron(P0));
  EXPECT_TRUE(IrredP0intoP1.ContainedInOtherHPolyhedron(P1));
  EXPECT_TRUE(IrredP1intoP0.ContainedInOtherHPolyhedron(P0));
  EXPECT_TRUE(IrredP1intoP0.ContainedInOtherHPolyhedron(P1));
}

GTEST_TEST(HPolyhedronTest, ReduceL1LInfBallIntersection) {
  Eigen::MatrixXd A_L1(8,3);
  Eigen::VectorXd b_L1 = Eigen::VectorXd::Ones(8);
  // clang-format off
  A_L1 <<  1, 1, 1,
           1, 1,-1,
           1,-1, 1,
           1,-1,-1,
          -1, 1, 1,
          -1, 1,-1,
          -1,-1, 1,
          -1,-1,-1;
  //clang-format on
  HPolyhedron L1_ball = HPolyhedron(A_L1, b_L1);

  Eigen::Vector3d lower_limit = -Eigen::Vector3d::Ones();
  Eigen::Vector3d upper_limit = Eigen::Vector3d::Ones();
  HPolyhedron Linfty_ball = HPolyhedron::MakeBox(lower_limit, upper_limit);


  Eigen::MatrixXd A_int(A_L1.rows() + Linfty_ball.A().rows(), 3);
  Eigen::MatrixXd b_int(A_int.rows(), 3);
  A_int.topRows(A_L1.rows()) = A_L1;
  b_int.topRows(b_L1.rows()) = b_L1;
  A_int.bottomRows(Linfty_ball.A().rows()) = Linfty_ball.A();
  b_int.bottomRows(Linfty_ball.b().rows()) = Linfty_ball.b();
  HPolyhedron polyhedron_to_reduce(A_int, b_int);
  HPolyhedron reduced_polyhedron = polyhedron_to_reduce.ReduceInequalities();

  EXPECT_TRUE(CompareMatrices(reduced_polyhedron.A(), A_L1));
  EXPECT_TRUE(CompareMatrices(reduced_polyhedron.b(), b_L1));
}

}  // namespace optimization
}  // namespace geometry
}  // namespace drake
