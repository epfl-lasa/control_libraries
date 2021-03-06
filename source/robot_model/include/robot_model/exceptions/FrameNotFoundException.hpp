#pragma once

#include <exception>

namespace robot_model::exceptions {
class FrameNotFoundException : public std::invalid_argument {
public:
  explicit FrameNotFoundException(const std::string& frame_name) :
      invalid_argument("Frame with name or ID " + frame_name + " is not in the robot model") {};
};
}// namespace robot_model::exceptions