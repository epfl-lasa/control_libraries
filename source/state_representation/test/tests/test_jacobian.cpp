#include "state_representation/robot/Jacobian.hpp"
#include "state_representation/exceptions/IncompatibleStatesException.hpp"
#include <gtest/gtest.h>

using namespace state_representation;
using namespace state_representation::exceptions;

TEST(JacobianTest, TestCreate) {
  Jacobian jac("robot", 7, "test");
  EXPECT_TRUE(jac.rows() == 6);
  EXPECT_TRUE(jac.cols() == 7);
  EXPECT_TRUE(jac.is_empty());
  EXPECT_TRUE(jac.get_frame() == "test");
  EXPECT_TRUE(jac.get_reference_frame() == "world");
  for (std::size_t i = 0; i < jac.cols(); ++i) {
    EXPECT_TRUE(jac.get_joint_names().at(i) == ("joint" + std::to_string(i)));
    EXPECT_TRUE(jac.col(i).norm() == 0);
  }
}

TEST(JacobianTest, TestCreateWithVectorOfJoints) {
  Jacobian jac("robot", std::vector<std::string>{"j1", "j2"}, "test", "test_ref");
  EXPECT_TRUE(jac.get_joint_names().at(0) == "j1");
  EXPECT_TRUE(jac.get_joint_names().at(1) == "j2");
  EXPECT_TRUE(jac.get_reference_frame() == "test_ref");
}

TEST(JacobianTest, TestSetData) {
  Jacobian jac("robot", 3, "test");
  jac.set_data(Eigen::MatrixXd::Random(6, 3));
  EXPECT_FALSE(jac.is_empty());
  for (std::size_t i = 0; i < jac.cols(); ++i) {
    EXPECT_TRUE(jac.col(i).norm() > 0);
  }
  bool except_thrown = false;
  try {
    jac.set_data(Eigen::MatrixXd::Random(7, 6));
  } catch (const IncompatibleSizeException& e) {
    except_thrown = true;
  }
  EXPECT_TRUE(except_thrown);
}

TEST(JacobianTest, TestRandomCreate) {
  Jacobian jac = Jacobian::Random("robot", 7, "test");
  EXPECT_FALSE(jac.is_empty());
  for (std::size_t i = 0; i < jac.cols(); ++i) {
    EXPECT_TRUE(jac.col(i).norm() > 0);
  }
}

TEST(JacobianTest, TestTranspose) {
  Jacobian jac = Jacobian::Random("robot", 7, "test");
  Jacobian jact = jac.transpose();

  EXPECT_TRUE(jact.rows() == 7);
  EXPECT_TRUE(jact.cols() == 6);

  for (std::size_t i = 0; i < jac.cols(); ++i) {
    EXPECT_TRUE(jac.col(i).isApprox(jact.row(i)));
  }
}

TEST(JacobianTest, TestMutltiplyWithEigen) {
  Jacobian jac = Jacobian::Random("robot", 7, "test");
  Eigen::MatrixXd mat1 = Eigen::MatrixXd::Random(7, 2);
  Eigen::MatrixXd res1 = jac * mat1;
  Eigen::MatrixXd res_truth = jac.data() * mat1;

  EXPECT_TRUE(res1.isApprox(res_truth));

  Eigen::MatrixXd mat2 = Eigen::VectorXd::Random(6, 1);
  bool except_thrown = false;
  try {
    Eigen::MatrixXd res2 = jac * mat2;
  } catch (const IncompatibleSizeException& e) {
    except_thrown = true;
  }
  EXPECT_TRUE(except_thrown);
}

TEST(JacobianTest, TestSolve) {
  Jacobian jac = Jacobian::Random("robot", 7, "test");
  Eigen::MatrixXd mat1 = Eigen::VectorXd::Random(7, 1);
  bool except_thrown = false;
  try {
    Eigen::MatrixXd res1 = jac.solve(mat1);
  } catch (const IncompatibleSizeException& e) {
    except_thrown = true;
  }
  EXPECT_TRUE(except_thrown);

  Eigen::MatrixXd mat2 = Eigen::VectorXd::Random(6, 1);
  Eigen::MatrixXd res2 = jac.solve(mat2);

  EXPECT_TRUE(res2.rows() == 7);
  EXPECT_TRUE(res2.cols() == 1);
}

TEST(JacobianTest, TestJointToCartesian) {
  Jacobian jac = Jacobian::Random("robot", 7, "test", "test_ref");
  JointVelocities jvel = JointVelocities::Random("robot", 7);
  CartesianTwist cvel = jac * jvel;

  EXPECT_TRUE(cvel.get_name() == jac.get_frame());
  EXPECT_TRUE(cvel.get_reference_frame() == jac.get_reference_frame());
  EXPECT_TRUE(cvel.data().isApprox(jac.data() * jvel.data()));
}

TEST(JacobianTest, TestCartesianToJoint) {
  Jacobian jac = Jacobian::Random("robot", 7, "test", "test_ref");
  CartesianTwist cvel = CartesianTwist::Random("test");

  bool except_thrown1 = false;
  try {
    JointVelocities jvel = jac.solve(cvel);
  } catch (const IncompatibleStatesException& e) {
    except_thrown1 = true;
  }
  EXPECT_TRUE(except_thrown1);
  cvel.set_reference_frame("test_ref");

  state_representation::JointVelocities jvel2;
  bool except_thrown2 = false;
  try {
    jvel2 = jac.pseudoinverse() * cvel;
  } catch (const IncompatibleSizeException& e) {
    except_thrown2 = true;
  }
  EXPECT_FALSE(except_thrown2);
  EXPECT_TRUE(jvel2.data().norm() > 0);
}

TEST(JacobianTest, TestChangeReferenceFrame) {
  Jacobian jac_in_test_ref = Jacobian::Random("robot", 7, "test", "test_ref");
  CartesianPose wTtest_ref = CartesianPose::Random("test_ref", "world");
  Jacobian jac_in_world = wTtest_ref * jac_in_test_ref;
  EXPECT_TRUE(jac_in_world.get_reference_frame() == wTtest_ref.get_reference_frame());
  // use a proxy operation with a twist to check correctness
  CartesianTwist vel_in_world = CartesianTwist::Random("test", "world");
  CartesianTwist vel_in_test_ref = wTtest_ref.inverse() * vel_in_world;
  JointVelocities jt1 = jac_in_world.solve(vel_in_world);
  JointVelocities jt2 = jac_in_test_ref.solve(vel_in_test_ref);
  EXPECT_TRUE(jt1.data().isApprox(jt2.data()));
}
