/**
 * @author Baptiste Busch
 * @date 2019/09/09
 */

#pragma once

#include "state_representation/Robot/JointPositions.hpp"
#include "state_representation/Robot/JointState.hpp"

namespace StateRepresentation {
class JointPositions;

/**
 * @class JointVelocities
 * @brief Class to define velocities of the joints
 */
class JointVelocities : public JointState {
public:
  /**
   * Empty constructor
   */
  explicit JointVelocities();

  /**
   * @brief Constructor with name and number of joints provided
   * @brief name the name of the state
   * @brief nb_joints the number of joints for initialization
   */
  explicit JointVelocities(const std::string& robot_name, unsigned int nb_joints = 0);

  /**
   * @brief Constructor with name and list of joint names provided
   * @brief name the name of the state
   * @brief joint_names list of joint names
   */
  explicit JointVelocities(const std::string& robot_name, const std::vector<std::string>& joint_names);

  /**
   * @brief Constructor with name and velocities values provided
   * @brief name the name of the state
   * @brief velocities the vector of velocities
   */
  explicit JointVelocities(const std::string& robot_name, const Eigen::VectorXd& velocities);

  /**
   * @brief Constructor with name, a list of joint names  and velocities values provided
   * @brief name the name of the state
   * @brief joint_names list of joint names
   * @brief velocities the vector of velocities
   */
  explicit JointVelocities(const std::string& robot_name, const std::vector<std::string>& joint_names,
                           const Eigen::VectorXd& velocities);

  /**
   * @brief Copy constructor
   */
  JointVelocities(const JointVelocities& velocities);

  /**
   * @brief Copy constructor from a JointState
   */
  JointVelocities(const JointState& state);

  /**
   * @brief Copy constructor from a JointPositions by considering that it is equivalent to dividing the positions by 1 second
   */
  JointVelocities(const JointPositions& positions);

  /**
   * @brief Constructor for the zero JointVelocities
   * @param robot_name the name of the associated robot
   * @param nb_joints the number of joints for initialization
   * @return JointState with zero values in all attributes
   */
  static JointVelocities Zero(const std::string& robot_name, unsigned int nb_joints);

  /**
   * @brief Constructor for the zero JointVelocities
   * @param robot_name the name of the associated robot
   * @param joint_names list of joint names
   * @return JointState with zero values in all attributes
   */
  static JointVelocities Zero(const std::string& robot_name, const std::vector<std::string>& joint_names);

  /**
   * @brief Constructor for the random JointVelocities
   * @param robot_name the name of the associated robot
   * @param nb_joints the number of joints for initialization
   * @return JointState with random values in all attributes
   */
  static JointVelocities Random(const std::string& robot_name, unsigned int nb_joints);

  /**
   * @brief Constructor for the random JointVelocities
   * @param robot_name the name of the associated robot
   * @param joint_names list of joint names
   * @return JointState with random values in all attributes
   */
  static JointVelocities Random(const std::string& robot_name, const std::vector<std::string>& joint_names);

  /**
   * @brief Copy assignment operator that have to be defined to the custom assignment operator
   * @param state the state with value to assign
   * @return reference to the current state with new values
   */
  JointVelocities& operator=(const JointVelocities& state);

  /**
   * @brief Overload the += operator
   * @param velocities JointVelocities to add
   * @return the current JointVelocities added the JointVelocities given in argument
   */
  JointVelocities& operator+=(const JointVelocities& velocities);

  /**
   * @brief Overload the + operator
   * @param velocities JointVelocities to add
   * @return the current JointVelocities added the JointVelocities given in argument
   */
  JointVelocities operator+(const JointVelocities& velocities) const;

  /**
   * @brief Overload the -= operator
   * @param velocities JointVelocities to subtract
   * @return the current JointVelocities subtracted the JointVelocities given in argument
   */
  JointVelocities& operator-=(const JointVelocities& velocities);

  /**
   * @brief Overload the - operator
   * @param velocities JointVelocities to subtract
   * @return the current JointVelocities subtracted the JointVelocities given in argument
   */
  JointVelocities operator-(const JointVelocities& velocities) const;

  /**
   * @brief Overload the *= operator with a double gain
   * @param lambda the gain to multiply with
   * @return the JointVelocities multiplied by lambda
   */
  JointVelocities& operator*=(double lambda);

  /**
   * @brief Overload the * operator with a double gain
   * @param lambda the gain to multiply with
   * @return the JointVelocities multiplied by lambda
   */
  JointVelocities operator*(double lambda) const;

  /**
   * @brief Overload the *= operator with an array of gains
   * @param lambda the gain array to multiply with
   * @return the JointVelocities multiplied by lambda
   */
  JointVelocities& operator*=(const Eigen::ArrayXd& lambda);

  /**
   * @brief Overload the *= operator with an array of gains
   * @param lambda the gain array to multiply with
   * @return the JointVelocities multiplied by lambda
   */
  JointVelocities operator*(const Eigen::ArrayXd& lambda) const;

  /**
   * @brief Overload the *= operator with a matrix of gains
   * @param lambda the matrix to multiply with
   * @return the JointVelocities multiplied by lambda
   */
  JointVelocities& operator*=(const Eigen::MatrixXd& lambda);

  /**
   * @brief Overload the * operator with a matrix of gains
   * @param lambda the matrix to multiply with
   * @return the JointVelocities multiplied by lambda
   */
  JointVelocities operator*(const Eigen::MatrixXd& lambda) const;

  /**
   * @brief Overload the /= operator with a scalar
   * @param lambda the scalar to divide with
   * @return the JointVelocities divided by lambda
   */
  JointVelocities& operator/=(double lambda);

  /**
   * @brief Overload the / operator with a scalar
   * @param lambda the scalar to divide with
   * @return the JointVelocities divided by lambda
   */
  JointVelocities operator/(double lambda) const;

  /**
   * @brief Overload the * operator with a time period
   * @param dt the time period to multiply with
   * @return the JointPositions corresponding to the displacement over the time period
   */
  JointPositions operator*(const std::chrono::nanoseconds& dt) const;

  /**
   * @brief Return a copy of the JointVelocities
   * @return the copy
   */
  JointVelocities copy() const;

  /**
   * @brief Returns the velocities data as an Eigen vector
   * @return the velocities data vector
   */
  Eigen::VectorXd data() const;

  /**
   * @brief Clamp inplace the magnitude of the velocity to the values in argument
   * @param max_absolute_value the maximum magnitude of torque for all the joints
   * @param noise_ratio if provided, this value will be used to apply a dead zone under which
   * the torque will be set to 0
   */
  void clamp(double max_absolute_value, double noise_ratio = 0.);

  /**
   * @brief Clamp inplace the magnitude of the velocity to the values in argument
   * @param max_absolute_value the maximum magnitude of torque for all the joints
   * @param noise_ratio if provided, this value will be used to apply a dead zone under which
   * the torque will be set to 0
   * @return the clamped JointVelocities
   */
  JointVelocities clamped(double max_absolute_value, double noise_ratio = 0.) const;

  /**
   * @brief Clamp inplace the magnitude of the velocity to the values in argument
   * @param max_absolute_value_array the maximum magnitude of torque for each joint
   * @param noise_ratio_array if provided, this value will be used to apply a dead zone under which
   * the torque will be set to 0
   */
  void clamp(const Eigen::ArrayXd& max_absolute_value_array, const Eigen::ArrayXd& noise_ratio_array);

  /**
   * @brief Clamp inplace the magnitude of the velocity to the values in argument
   * @param max_absolute_value_array the maximum magnitude of torque for each joint
   * @param noise_ratio_array if provided, this value will be used to apply a dead zone under which
   * the torque will be set to 0
   * @return the clamped JointVelocities
   */
  JointVelocities clamped(const Eigen::ArrayXd& max_absolute_value_array,
                          const Eigen::ArrayXd& noise_ratio_array) const;

  /**
   * @brief Overload the ostream operator for printing
   * @param os the ostream to append the string representing the state
   * @param state the state to print
   * @return the appended ostream
   */
  friend std::ostream& operator<<(std::ostream& os, const JointVelocities& velocities);

  /**
   * @brief Overload the * operator with a scalar
   * @param lambda the scalar to multiply with
   * @return the JointVelocities provided multiply by lambda
   */
  friend JointVelocities operator*(double lambda, const JointVelocities& velocities);

  /**
   * @brief Overload the * operator with an array of gains
   * @param lambda the array to multiply with
   * @return the JointVelocities provided multiply by lambda
   */
  friend JointVelocities operator*(const Eigen::ArrayXd& lambda, const JointVelocities& velocities);

  /**
   * @brief Overload the * operator with a matrix of gains
   * @param lambda the matrix to multiply with
   * @return the JointVelocities provided multiply by lambda
   */
  friend JointVelocities operator*(const Eigen::MatrixXd& lambda, const JointVelocities& velocities);

  /**
   * @brief Overload the * operator with a time period
   * @param dt the time period to multiply with
   * @return the JointPositions corresponding to the displacement over the time period
   */
  friend JointPositions operator*(const std::chrono::nanoseconds& dt, const JointVelocities& velocities);
};

inline JointVelocities& JointVelocities::operator=(const JointVelocities& state) {
  JointState::operator=(state);
  return (*this);
}
}// namespace StateRepresentation
