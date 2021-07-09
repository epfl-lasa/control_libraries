#include <gtest/gtest.h>
#include <cmath>

#include "state_representation/space/cartesian/CartesianPose.hpp"
#include "state_representation/space/cartesian/CartesianTwist.hpp"
#include "state_representation/space/cartesian/CartesianWrench.hpp"

using namespace state_representation;

TEST(CartesianStateTest, IdentityInitialization) {
  CartesianState identity = CartesianState::Identity("test");
  // the joint state should not be considered empty (as it is properly initialized)
  EXPECT_FALSE(identity.is_empty());
  // all data should be zero except for orientation that should be identity
  EXPECT_EQ(identity.get_position().norm(), 0);
  EXPECT_EQ(identity.get_orientation().norm(), 1);
  EXPECT_EQ(identity.get_orientation().w(), 1);
  EXPECT_EQ(identity.get_twist().norm(), 0);
  EXPECT_EQ(identity.get_accelerations().norm(), 0);
  EXPECT_EQ(identity.get_wrench().norm(), 0);
}

TEST(CartesianStateTest, RandomStateInitialization) {
  CartesianState random = CartesianState::Random("test");
  // all data should be random (non 0)
  EXPECT_GT(random.get_position().norm(), 0);
  EXPECT_GT(abs(random.get_orientation().w()), 0);
  EXPECT_GT(abs(random.get_orientation().x()), 0);
  EXPECT_GT(abs(random.get_orientation().y()), 0);
  EXPECT_GT(abs(random.get_orientation().z()), 0);
  EXPECT_GT(random.get_twist().norm(), 0);
  EXPECT_GT(random.get_accelerations().norm(), 0);
  EXPECT_GT(random.get_wrench().norm(), 0);
}

TEST(CartesianStateTest, RandomPoseInitialization) {
  CartesianPose random = CartesianPose::Random("test");
  // only position should be random
  EXPECT_GT(random.get_position().norm(), 0);
  EXPECT_GT(abs(random.get_orientation().w()), 0);
  EXPECT_GT(abs(random.get_orientation().x()), 0);
  EXPECT_GT(abs(random.get_orientation().y()), 0);
  EXPECT_GT(abs(random.get_orientation().z()), 0);
  EXPECT_EQ(static_cast<CartesianState&>(random).get_twist().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(random).get_accelerations().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(random).get_wrench().norm(), 0);
}

TEST(CartesianStateTest, RandomTwistInitialization) {
  CartesianTwist random = CartesianTwist::Random("test");
  // only position should be random
  EXPECT_EQ(static_cast<CartesianState&>(random).get_position().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(random).get_orientation().norm(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(random).get_orientation().w(), 1);
  EXPECT_GT(random.get_twist().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(random).get_accelerations().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(random).get_wrench().norm(), 0);
}

TEST(CartesianStateTest, RandomWrenchInitialization) {
  CartesianWrench random = CartesianWrench::Random("test");
  // only position should be random
  EXPECT_EQ(static_cast<CartesianState&>(random).get_position().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(random).get_orientation().norm(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(random).get_orientation().w(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(random).get_twist().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(random).get_accelerations().norm(), 0);
  EXPECT_GT(random.get_wrench().norm(), 0);
}

TEST(CartesianStateTest, CopyState) {
  CartesianState state1 = CartesianState::Random("test");
  CartesianState state2(state1);
  EXPECT_EQ(state1.get_name(), state2.get_name());
  EXPECT_EQ(state1.get_reference_frame(), state2.get_reference_frame());
  EXPECT_TRUE(state1.data().isApprox(state2.data()));
  CartesianState state3 = state1;
  EXPECT_EQ(state1.get_name(), state3.get_name());
  EXPECT_EQ(state1.get_reference_frame(), state3.get_reference_frame());
  EXPECT_TRUE(state1.data().isApprox(state3.data()));

  CartesianState state4;
  EXPECT_TRUE(state4.is_empty());
  CartesianState state5 = state4;
  EXPECT_TRUE(state5.is_empty());
}

TEST(CartesianStateTest, CopyPose) {
  CartesianPose pose1 = CartesianPose::Random("test");
  CartesianPose pose2(pose1);
  EXPECT_EQ(pose1.get_name(), pose2.get_name());
  EXPECT_EQ(pose1.get_reference_frame(), pose2.get_reference_frame());
  EXPECT_TRUE(pose1.data().isApprox(pose2.data()));
  EXPECT_EQ(static_cast<CartesianState&>(pose2).get_twist().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(pose2).get_accelerations().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(pose2).get_wrench().norm(), 0);
  CartesianPose pose3 = pose1;
  EXPECT_EQ(pose1.get_name(), pose3.get_name());
  EXPECT_EQ(pose1.get_reference_frame(), pose3.get_reference_frame());
  EXPECT_TRUE(pose1.data().isApprox(pose3.data()));
  EXPECT_EQ(static_cast<CartesianState&>(pose3).get_twist().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(pose3).get_accelerations().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(pose3).get_wrench().norm(), 0);
  // try to change non pose variables prior to the copy, those should be discarded
  static_cast<CartesianState&>(pose1).set_twist(Eigen::VectorXd::Random(6));
  static_cast<CartesianState&>(pose1).set_accelerations(Eigen::VectorXd::Random(6));
  static_cast<CartesianState&>(pose1).set_wrench(Eigen::VectorXd::Random(6));
  CartesianPose pose4 = pose1;
  EXPECT_TRUE(pose1.data().isApprox(pose4.data()));
  EXPECT_EQ(static_cast<CartesianState&>(pose4).get_twist().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(pose4).get_accelerations().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(pose4).get_wrench().norm(), 0);
  // copy a state, only the pose variables should be non 0
  CartesianPose pose5 = CartesianState::Random("test");
  EXPECT_EQ(static_cast<CartesianState&>(pose5).get_twist().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(pose5).get_accelerations().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(pose5).get_wrench().norm(), 0);

  CartesianPose pose6;
  EXPECT_TRUE(pose6.is_empty());
  CartesianPose pose7 = pose6;
  EXPECT_TRUE(pose7.is_empty());
}

TEST(CartesianStateTest, CopyTwist) {
  CartesianTwist twist1 = CartesianTwist::Random("test");
  CartesianTwist twist2(twist1);
  EXPECT_EQ(twist1.get_name(), twist2.get_name());
  EXPECT_EQ(twist1.get_reference_frame(), twist2.get_reference_frame());
  EXPECT_TRUE(twist1.data().isApprox(twist2.data()));
  EXPECT_EQ(static_cast<CartesianState&>(twist2).get_position().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(twist2).get_orientation().norm(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(twist2).get_orientation().w(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(twist2).get_accelerations().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(twist2).get_wrench().norm(), 0);
  CartesianTwist twist3 = twist1;
  EXPECT_EQ(twist1.get_name(), twist3.get_name());
  EXPECT_EQ(twist1.get_reference_frame(), twist3.get_reference_frame());
  EXPECT_TRUE(twist1.data().isApprox(twist3.data()));
  EXPECT_EQ(static_cast<CartesianState&>(twist3).get_position().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(twist3).get_orientation().norm(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(twist3).get_orientation().w(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(twist3).get_accelerations().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(twist3).get_wrench().norm(), 0);
  // try to change non pose variables prior to the copy, those should be discarded
  static_cast<CartesianState&>(twist1).set_pose(Eigen::VectorXd::Random(7));
  static_cast<CartesianState&>(twist1).set_accelerations(Eigen::VectorXd::Random(6));
  static_cast<CartesianState&>(twist1).set_wrench(Eigen::VectorXd::Random(6));
  CartesianTwist twist4 = twist1;
  EXPECT_TRUE(twist1.data().isApprox(twist4.data()));
  EXPECT_EQ(static_cast<CartesianState&>(twist4).get_position().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(twist4).get_orientation().norm(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(twist4).get_orientation().w(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(twist4).get_accelerations().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(twist4).get_wrench().norm(), 0);
  // copy a state, only the pose variables should be non 0
  CartesianTwist twist5 = CartesianState::Random("test");
  EXPECT_EQ(static_cast<CartesianState&>(twist5).get_position().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(twist5).get_orientation().norm(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(twist5).get_orientation().w(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(twist5).get_accelerations().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(twist5).get_wrench().norm(), 0);

  CartesianTwist twist6;
  EXPECT_TRUE(twist6.is_empty());
  CartesianTwist twist7 = twist6;
  EXPECT_TRUE(twist7.is_empty());
}

TEST(CartesianStateTest, CopyWrench) {
  CartesianWrench wrench1 = CartesianWrench::Random("test");
  CartesianWrench wrench2(wrench1);
  EXPECT_EQ(wrench1.get_name(), wrench2.get_name());
  EXPECT_EQ(wrench1.get_reference_frame(), wrench2.get_reference_frame());
  EXPECT_TRUE(wrench1.data().isApprox(wrench2.data()));
  EXPECT_EQ(static_cast<CartesianState&>(wrench2).get_position().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(wrench2).get_orientation().norm(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(wrench2).get_orientation().w(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(wrench2).get_twist().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(wrench2).get_accelerations().norm(), 0);
  CartesianWrench wrench3 = wrench1;
  EXPECT_EQ(wrench1.get_name(), wrench3.get_name());
  EXPECT_EQ(wrench1.get_reference_frame(), wrench3.get_reference_frame());
  EXPECT_TRUE(wrench1.data().isApprox(wrench3.data()));
  EXPECT_EQ(static_cast<CartesianState&>(wrench3).get_position().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(wrench3).get_orientation().norm(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(wrench3).get_orientation().w(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(wrench3).get_twist().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(wrench3).get_accelerations().norm(), 0);
  // try to change non pose variables prior to the copy, those should be discarded
  static_cast<CartesianState&>(wrench1).set_pose(Eigen::VectorXd::Random(7));
  static_cast<CartesianState&>(wrench1).set_twist(Eigen::VectorXd::Random(6));
  static_cast<CartesianState&>(wrench1).set_accelerations(Eigen::VectorXd::Random(6));
  CartesianWrench wrench4 = wrench1;
  EXPECT_TRUE(wrench1.data().isApprox(wrench4.data()));
  EXPECT_EQ(static_cast<CartesianState&>(wrench4).get_position().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(wrench4).get_orientation().norm(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(wrench4).get_orientation().w(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(wrench4).get_twist().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(wrench4).get_accelerations().norm(), 0);
  // copy a state, only the pose variables should be non 0
  CartesianWrench wrench5 = CartesianState::Random("test");
  EXPECT_EQ(static_cast<CartesianState&>(wrench5).get_position().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(wrench5).get_orientation().norm(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(wrench5).get_orientation().w(), 1);
  EXPECT_EQ(static_cast<CartesianState&>(wrench5).get_twist().norm(), 0);
  EXPECT_EQ(static_cast<CartesianState&>(wrench5).get_accelerations().norm(), 0);

  CartesianWrench wrench6;
  EXPECT_TRUE(wrench6.is_empty());
  CartesianWrench wrench7 = wrench6;
  EXPECT_TRUE(wrench7.is_empty());
}

TEST(CartesianStateTest, GetData) {
  CartesianState cs = CartesianState::Random("test");
  Eigen::VectorXd concatenated_state(25);
  concatenated_state << cs.get_pose(), cs.get_twist(), cs.get_accelerations(), cs.get_wrench();
  EXPECT_NEAR(concatenated_state.norm(), cs.data().norm(), 1e-4);
}

TEST(CartesianStateTest, CartesianStateToStdVector) {
  CartesianState cs = CartesianState::Random("test");
  std::vector<double> vec_data = cs.to_std_vector();
  for (size_t i = 0; i < vec_data.size(); ++i) {
    EXPECT_EQ(cs.data()(i), vec_data[i]);
  }
}

TEST(CartesianStateTest, CartesianPoseToStdVector) {
  CartesianPose cp = CartesianPose::Random("test");
  std::vector<double> vec_data = cp.to_std_vector();
  EXPECT_EQ(vec_data.size(), 7);
  for (size_t i = 0; i < vec_data.size(); ++i) {
    EXPECT_EQ(cp.data()(i), vec_data[i]);
  }
}

TEST(CartesianStateTest, CartesianTwistToStdVector) {
  CartesianTwist ct = CartesianTwist::Random("test");
  std::vector<double> vec_data = ct.to_std_vector();
  EXPECT_EQ(vec_data.size(), 6);
  for (size_t i = 0; i < vec_data.size(); ++i) {
    EXPECT_EQ(ct.data()(i), vec_data[i]);
  }
}

TEST(CartesianStateTest, CartesianWrenchToStdVector) {
  CartesianWrench cw = CartesianWrench::Random("test");
  std::vector<double> vec_data = cw.to_std_vector();
  EXPECT_EQ(vec_data.size(), 6);
  for (size_t i = 0; i < vec_data.size(); ++i) {
    EXPECT_EQ(cw.data()(i), vec_data[i]);
  }
}

TEST(CartesianStateTest, NegateQuaternion) {
  Eigen::Quaterniond q = Eigen::Quaterniond::UnitRandom();
  Eigen::Quaterniond q2 = Eigen::Quaterniond(-q.coeffs());
  EXPECT_EQ(q.w(), -q2.w());
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(q.vec()(i), -q2.vec()(i));
  }
}

TEST(CartesianStateTest, MultiplyTransformsBothOperators) {
  Eigen::Vector3d pos1(1, 2, 3);
  Eigen::Quaterniond rot1(1, 0, 0, 0);
  CartesianPose tf1("t1", pos1, rot1);
  Eigen::Vector3d pos2(4, 5, 6);
  Eigen::Quaterniond rot2(1, 0, 0, 0);
  CartesianPose tf2("t2", pos2, rot2, "t1");
  CartesianPose tf3 = tf1 * tf2;
  tf1 *= tf2;
  EXPECT_EQ(tf3.get_name(), "t2");
  for (int i = 0; i < tf1.get_position().size(); ++i)
    EXPECT_NEAR(tf1.get_position()(i), tf3.get_position()(i), 0.00001);
}

TEST(CartesianStateTest, MultiplyTransformsSameOrientation) {
  Eigen::Vector3d pos1(1, 2, 3);
  Eigen::Quaterniond rot1(1, 0, 0, 0);
  CartesianPose tf1("t1", pos1, rot1);
  Eigen::Vector3d pos2(4, 5, 6);
  Eigen::Quaterniond rot2(1, 0, 0, 0);
  CartesianPose tf2("t2", pos2, rot2, "t1");
  tf1 *= tf2;
  Eigen::Vector3d pos_truth(5, 7, 9);
  for (int i = 0; i < pos_truth.size(); ++i) {
    EXPECT_NEAR(tf1.get_position()(i), pos_truth(i), 0.00001);
  }
}

TEST(CartesianStateTest, MultiplyTransformsDifferentOrientation) {
  Eigen::Vector3d pos1(1, 2, 3);
  Eigen::Quaterniond rot1(0.70710678, 0.70710678, 0., 0.);
  CartesianPose tf1("t1", pos1, rot1);
  Eigen::Vector3d pos2(4, 5, 6);
  Eigen::Quaterniond rot2(0., 0., 0.70710678, 0.70710678);
  CartesianPose tf2("t2", pos2, rot2, "t1");
  tf1 *= tf2;
  Eigen::Vector3d pos_truth(5, -4, 8);
  Eigen::Quaterniond rot_truth(0., 0., 0., 1.);
  for (int i = 0; i < pos_truth.size(); ++i) {
    EXPECT_NEAR(tf1.get_position()(i), pos_truth(i), 0.00001);
  }
  EXPECT_GT(abs(tf1.get_orientation().dot(rot_truth)), 1 - 10E-4);
}

TEST(CartesianStateTest, TestInverseNullOrientation) {
  Eigen::Vector3d pos1(1, 2, 3);
  Eigen::Quaterniond rot1(1., 0., 0., 0.);
  CartesianPose tf1("t1", pos1, rot1);
  tf1 = tf1.inverse();
  Eigen::Vector3d pos_truth(-1, -2, -3);
  Eigen::Quaterniond rot_truth(1., 0., 0., 0.);
  EXPECT_EQ(tf1.get_name(), "world");
  EXPECT_EQ(tf1.get_reference_frame(), "t1");
  for (int i = 0; i < pos_truth.size(); ++i) {
    EXPECT_NEAR(tf1.get_position()(i), pos_truth(i), 0.00001);
  }
  EXPECT_GT(abs(tf1.get_orientation().dot(rot_truth)), 1 - 10E-4);
}

TEST(CartesianStateTest, TestInverseNonNullOrientation) {
  Eigen::Vector3d pos1(1, 2, 3);
  Eigen::Quaterniond rot1(0.70710678, 0.70710678, 0., 0.);
  CartesianPose tf1("t1", pos1, rot1);
  tf1 = tf1.inverse();
  Eigen::Vector3d pos_truth(-1, -3, 2);
  Eigen::Quaterniond rot_truth(0.70710678, -0.70710678, 0., 0.);
  for (int i = 0; i < pos_truth.size(); ++i) {
    EXPECT_NEAR(tf1.get_position()(i), pos_truth(i), 0.00001);
  }
  EXPECT_GT(abs(tf1.get_orientation().dot(rot_truth)), 1 - 10E-4);
}

TEST(CartesianStateTest, TestMultiplyInverseNonNullOrientation) {
  Eigen::Vector3d pos1(1, 2, 3);
  Eigen::Quaterniond rot1(0.70710678, 0.70710678, 0., 0.);
  CartesianPose tf1("t1", pos1, rot1);
  tf1 *= tf1.inverse();
  Eigen::Vector3d pos_truth(0, 0, 0);
  Eigen::Quaterniond rot_truth(1., 0., 0., 0.);
  for (int i = 0; i < pos_truth.size(); ++i) {
    EXPECT_NEAR(tf1.get_position()(i), pos_truth(i), 0.00001);
  }
  EXPECT_GT(abs(tf1.get_orientation().dot(rot_truth)), 1 - 10E-4);
}

TEST(CartesianStateTest, TestInverseStates) {
  CartesianState a_state_b = CartesianState::Random("B", "A");
  auto b_state_a = a_state_b.inverse();
  auto expect_null = a_state_b * b_state_a;

  EXPECT_NEAR(expect_null.get_pose().norm(), 1, 1e-5);
  EXPECT_NEAR(expect_null.get_linear_velocity().norm(), 0, 1e-5);
  EXPECT_NEAR(expect_null.get_angular_velocity().norm(), 0, 1e-5);
  EXPECT_NEAR(expect_null.get_linear_acceleration().norm(), 0, 1e-5);
  EXPECT_NEAR(expect_null.get_angular_acceleration().norm(), 0, 1e-5);
  EXPECT_NEAR(expect_null.get_force().norm(), 0, 1e-5);
  EXPECT_NEAR(expect_null.get_torque().norm(), 0, 1e-5);

  // conservation of power must hold
  auto power = a_state_b.get_twist().transpose() * a_state_b.get_wrench();
  auto power_inverse = b_state_a.get_twist().transpose() * b_state_a.get_wrench();
  EXPECT_NEAR(power, power_inverse, 1e-5);
}

TEST(CartesianStateTest, MultiplyPoseAndState) {
  CartesianPose p = CartesianPose::Random("test");
  CartesianState s = CartesianPose::Random("test2", "test");
  CartesianState res = p * s;
  CartesianPose res2 = p * static_cast<CartesianPose>(s);
  for (int i = 0; i < res.get_position().size(); ++i) {
    EXPECT_NEAR(res.get_position()(i), res2.get_position()(i), 0.00001);
  }
  EXPECT_GT(abs(res.get_orientation().dot(res2.get_orientation())), 1 - 10E-4);
}

TEST(CartesianStateTest, TestAddTwoPoses) {
  Eigen::Vector3d pos1 = Eigen::Vector3d::Zero();
  Eigen::Quaterniond rot1 = Eigen::Quaterniond::Identity();
  CartesianPose tf1("t1", pos1, rot1);
  Eigen::Vector3d pos2(1, 0, 0);
  Eigen::Quaterniond rot2(0, 1, 0, 0);
  CartesianPose tf2("t1", pos2, rot2);
}

TEST(CartesianStateTest, TestAddDisplacement) {
  Eigen::Vector3d pos1 = Eigen::Vector3d::Zero();
  Eigen::Quaterniond rot1 = Eigen::Quaterniond::Identity();
  CartesianPose tf1("t1", pos1, rot1);
  CartesianTwist vel("t1");
  vel.set_linear_velocity(Eigen::Vector3d(0.1, 0.1, 0.1));
  vel.set_angular_velocity(Eigen::Vector3d(0.1, 0.1, 0));
  std::chrono::milliseconds dt1(10);
  std::chrono::milliseconds dt2(1000);
  std::chrono::seconds dt3(1);
}

TEST(CartesianStateTest, TestPoseToVelocity) {
  Eigen::Vector3d pos1 = Eigen::Vector3d::Zero();
  Eigen::Quaterniond rot1 = Eigen::Quaterniond::Identity();
  CartesianPose tf1("t1", pos1, rot1);
  Eigen::Vector3d pos2(1, 0, 0);
  Eigen::Quaterniond rot2(0, 1, 0, 0);
  CartesianPose tf2("t1", pos2, rot2);
  std::chrono::seconds dt1(1);
  std::chrono::seconds dt2(10);
  std::chrono::milliseconds dt3(100);
}

TEST(CartesianStateTest, TestStateStateMultiplication) {
  CartesianState a_state_b = CartesianState::Identity("B", "A");
  CartesianState b_state_c = CartesianState::Random("C", "B");

  auto a_state_c = a_state_b * b_state_c;

  // if the base frame is identity, the transformation should not affect the data
  EXPECT_TRUE(a_state_c.get_pose().isApprox(b_state_c.get_pose()));
  EXPECT_TRUE(a_state_c.get_accelerations().isApprox(b_state_c.get_accelerations()));
  EXPECT_TRUE(a_state_c.get_twist().isApprox(b_state_c.get_twist()));
  EXPECT_TRUE(a_state_c.get_wrench().isApprox(b_state_c.get_wrench()));

  a_state_b = CartesianState::Random("B", "A");
  a_state_c = a_state_b * b_state_c;

  // conservation of power dictates that the product of force and velocity of
  // frame C must be the same in any reference frame.
  double b_power_c = b_state_c.get_twist().transpose() * b_state_c.get_wrench();
  double projected_b_power_c = (a_state_c.get_twist() - a_state_b.get_twist()).transpose()
      * (a_state_c.get_wrench() - a_state_b.get_wrench());

  EXPECT_NEAR(b_power_c, projected_b_power_c, 1e-3);
}

TEST(CartesianStateTest, TestImplicitConversion) {
  Eigen::Vector3d pos1 = Eigen::Vector3d::Zero();
  Eigen::Quaterniond rot1 = Eigen::Quaterniond::Identity();
  CartesianPose tf1("t1", pos1, rot1);
  CartesianTwist vel("t1");
  vel.set_linear_velocity(Eigen::Vector3d(0.1, 0.1, 0.1));
  vel.set_angular_velocity(Eigen::Vector3d(0.1, 0.1, 0));
  tf1 += vel;
}

TEST(CartesianStateTest, TestVelocityClamping) {
  CartesianTwist vel("test", Eigen::Vector3d(1, -2, 3), Eigen::Vector3d(1, 2, -3));
  vel.clamp(1, 0.5);
  EXPECT_LE(vel.get_linear_velocity().norm(), 1);
  EXPECT_LE(vel.get_angular_velocity().norm(), 0.5);
  vel *= 0.01;
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(vel.clamped(1, 0.5, 0.1, 0.1).get_linear_velocity()(i), 0);
    EXPECT_EQ(vel.clamped(1, 0.5, 0.1, 0.1).get_angular_velocity()(i), 0);
  }
}

TEST(CartesianStateTest, TestPoseDistance) {
  CartesianPose p1("test",
                   Eigen::Vector3d(-0.353849997774433, -0.823586525156853, -1.57705702279920),
                   Eigen::Quaterniond(-0.20495, -0.53709, -0.26785, -0.77316));
  CartesianPose p2("test",
                   Eigen::Vector3d(0.507974650905946, 0.281984063670556, 0.0334798822444514),
                   Eigen::Quaterniond(-0.34616, 0.78063, 0.51161, 0.095041));
  EXPECT_FLOAT_EQ(p1.dist(p1), 0);
  EXPECT_FLOAT_EQ(dist(p1, p2), p1.dist(p2));
  EXPECT_NEAR(p1.dist(p2, CartesianStateVariable::POSITION), 2.13514804509215, 1e-3);
  EXPECT_NEAR(p1.dist(p2, CartesianStateVariable::ORIENTATION), 1.955608640373294, 1e-3);
  EXPECT_NEAR(p1.dist(p2), 4.090756685465443, 1e-3);
}

TEST(CartesianStateTest, TestFilter) {
  CartesianPose tf1("t1", Eigen::Vector3d::Random(), Eigen::Quaterniond::UnitRandom());
  CartesianPose tf2("t1", Eigen::Vector3d::Random(), Eigen::Quaterniond::UnitRandom());
  for (int i = 0; i < 1000; ++i) {
    CartesianPose temp = tf1;
    double alpha = 0.1;
    tf1 = (1 - alpha) * tf1 + alpha * tf2;
    EXPECT_LT((tf1.get_position() - ((1 - alpha) * temp.get_position() + alpha * tf2.get_position())).norm(), 1e-4);
  }
}

TEST(CartesianStateTest, TestToSTDVector) {
  CartesianPose p = CartesianPose::Random("t1");
  std::vector<double> v = p.to_std_vector();
  for (unsigned int i = 0; i < 3; ++i) {
    EXPECT_NEAR(p.get_position()(i), v[i], 0.00001);
  }
  EXPECT_NEAR(p.get_orientation().w(), v[3], 0.00001);
  EXPECT_NEAR(p.get_orientation().x(), v[4], 0.00001);
  EXPECT_NEAR(p.get_orientation().y(), v[5], 0.00001);
  EXPECT_NEAR(p.get_orientation().z(), v[6], 0.00001);
}

TEST(CartesianStateTest, TestAllNorms) {
  double tolerance = 1e-4;
  CartesianState cs = CartesianState::Random("cs");
  // first test all norms
  std::vector<double> norms = cs.norms();
  EXPECT_TRUE(norms.size() == 8);
  EXPECT_NEAR(norms[0], cs.get_position().norm(), tolerance);
  EXPECT_NEAR(norms[1], cs.get_orientation().norm(), tolerance);
  EXPECT_NEAR(norms[2], cs.get_linear_velocity().norm(), tolerance);
  EXPECT_NEAR(norms[3], cs.get_angular_velocity().norm(), tolerance);
  EXPECT_NEAR(norms[4], cs.get_linear_acceleration().norm(), tolerance);
  EXPECT_NEAR(norms[5], cs.get_angular_acceleration().norm(), tolerance);
  EXPECT_NEAR(norms[6], cs.get_force().norm(), tolerance);
  EXPECT_NEAR(norms[7], cs.get_torque().norm(), tolerance);
}

TEST(CartesianStateTest, TestPoseNorms) {
  std::vector<double> norms;
  double tolerance = 1e-4;
  CartesianState cs = CartesianState::Random("cs");
  // independent variables first
  norms = cs.norms(CartesianStateVariable::POSITION);
  EXPECT_TRUE(norms.size() == 1);
  EXPECT_NEAR(norms[0], cs.get_position().norm(), tolerance);
  norms = cs.norms(CartesianStateVariable::ORIENTATION);
  EXPECT_TRUE(norms.size() == 1);
  EXPECT_NEAR(norms[0], cs.get_orientation().norm(), tolerance);
  // then grouped by two
  norms = cs.norms(CartesianStateVariable::POSE);
  EXPECT_TRUE(norms.size() == 2);
  EXPECT_NEAR(norms[0], cs.get_position().norm(), tolerance);
  EXPECT_NEAR(norms[1], cs.get_orientation().norm(), tolerance);
  // test with CartesianPose default variable
  CartesianPose cp = CartesianPose::Random("cp");
  std::vector<double> pose_norms = cp.norms();
  EXPECT_TRUE(pose_norms.size() == 2);
  EXPECT_NEAR(pose_norms[0], cp.get_position().norm(), tolerance);
  EXPECT_NEAR(pose_norms[1], cp.get_orientation().norm(), tolerance);
}

TEST(CartesianStateTest, TestTwistNorms) {
  std::vector<double> norms;
  double tolerance = 1e-4;
  CartesianState cs = CartesianState::Random("cs");
  // independent variables first
  norms = cs.norms(CartesianStateVariable::LINEAR_VELOCITY);
  EXPECT_TRUE(norms.size() == 1);
  EXPECT_NEAR(norms[0], cs.get_linear_velocity().norm(), tolerance);
  norms = cs.norms(CartesianStateVariable::ANGULAR_VELOCITY);
  EXPECT_TRUE(norms.size() == 1);
  EXPECT_NEAR(norms[0], cs.get_angular_velocity().norm(), tolerance);
  // then grouped by two
  norms = cs.norms(CartesianStateVariable::TWIST);
  EXPECT_TRUE(norms.size() == 2);
  EXPECT_NEAR(norms[0], cs.get_linear_velocity().norm(), tolerance);
  EXPECT_NEAR(norms[1], cs.get_angular_velocity().norm(), tolerance);
  // test with CartesianTwist default variable
  CartesianTwist ct = CartesianTwist::Random("ct");
  std::vector<double> twist_norms = ct.norms();
  EXPECT_TRUE(twist_norms.size() == 2);
  EXPECT_NEAR(twist_norms[0], ct.get_linear_velocity().norm(), tolerance);
  EXPECT_NEAR(twist_norms[1], ct.get_angular_velocity().norm(), tolerance);
}

TEST(CartesianStateTest, TestAccelerationNorms) {
  std::vector<double> norms;
  double tolerance = 1e-4;
  CartesianState cs = CartesianState::Random("cs");
  // independent variables first
  norms = cs.norms(CartesianStateVariable::LINEAR_ACCELERATION);
  EXPECT_EQ(norms.size(), 1);
  EXPECT_NEAR(norms[0], cs.get_linear_acceleration().norm(), tolerance);
  norms = cs.norms(CartesianStateVariable::ANGULAR_ACCELERATION);
  EXPECT_EQ(norms.size(), 1);
  EXPECT_NEAR(norms[0], cs.get_angular_acceleration().norm(), tolerance);
  // then grouped by two
  norms = cs.norms(CartesianStateVariable::ACCELERATIONS);
  EXPECT_EQ(norms.size(), 2);
  EXPECT_NEAR(norms[0], cs.get_linear_acceleration().norm(), tolerance);
  EXPECT_NEAR(norms[1], cs.get_angular_acceleration().norm(), tolerance);
}

TEST(CartesianStateTest, TestWrenchNorms) {
  std::vector<double> norms;
  double tolerance = 1e-4;
  CartesianState cs = CartesianState::Random("cs");
  // independent variables first
  norms = cs.norms(CartesianStateVariable::FORCE);
  EXPECT_EQ(norms.size(), 1);
  EXPECT_NEAR(norms[0], cs.get_force().norm(), tolerance);
  norms = cs.norms(CartesianStateVariable::TORQUE);
  EXPECT_EQ(norms.size(), 1);
  EXPECT_NEAR(norms[0], cs.get_torque().norm(), tolerance);
  // then grouped by two
  norms = cs.norms(CartesianStateVariable::WRENCH);
  EXPECT_EQ(norms.size(), 2);
  EXPECT_NEAR(norms[0], cs.get_force().norm(), tolerance);
  EXPECT_NEAR(norms[1], cs.get_torque().norm(), tolerance);
  // test with CartesianTwist default variable
  CartesianWrench cw = CartesianWrench::Random("cw");
  std::vector<double> wrench_norms = cw.norms();
  EXPECT_EQ(wrench_norms.size(), 2);
  EXPECT_NEAR(wrench_norms[0], cw.get_force().norm(), tolerance);
  EXPECT_NEAR(wrench_norms[1], cw.get_torque().norm(), tolerance);
}

TEST(CartesianStateTest, TestNormalize) {
  double tolerance = 1e-4;
  CartesianState cs = CartesianState::Random("cs");
  cs.normalize();
  std::vector<double> norms = cs.norms();
  for (double n : norms) {
    EXPECT_NEAR(n, 1.0, tolerance);
  }
}

TEST(CartesianStateTest, TestNormalized) {
  double tolerance = 1e-4;
  CartesianState cs = CartesianState::Random("cs");
  CartesianState csn = cs.normalized();
  std::vector<double> norms = csn.norms();
  for (double n : norms) {
    EXPECT_NEAR(n, 1.0, tolerance);
  }
}
