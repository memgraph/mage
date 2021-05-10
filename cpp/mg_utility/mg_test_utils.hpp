#pragma once

#include <chrono>
#include <cmath>
#include <vector>
#include <cassert>

namespace mg_test_utility {

constexpr double ABSOLUTE_ERROR_EPSILON { 10e-4 };
constexpr double AVERAGE_ABSOLUTE_ERROR_EPSILON { 10e-5 };

/// This class is threadsafe
class Timer {
 public:
  explicit Timer() : start_time_(std::chrono::steady_clock::now()) {}

  template <typename TDuration = std::chrono::duration<double>>
  TDuration Elapsed() const {
    return std::chrono::duration_cast<TDuration>(std::chrono::steady_clock::now() - start_time_);
  }

 private:
  std::chrono::steady_clock::time_point start_time_;
};

///
///@brief Method that calculates the maximum absolute error between vector components.
/// The vector type must be an arithmetic type. Vectors must have the same size.
///
///@param result Container that stores calculated values
///@param correct Container that stores accurate values
///@return maximum absolute error
///
template <typename T = double>
T MaxAbsoluteError(const std::vector<T> &result, const std::vector<T> &correct) {
  static_assert(
      std::is_arithmetic_v<T>,
      "mg_test_utility::MaxAbsoluteError expects the vector type to be an arithmetic type.\n");

  assert (result.size() == correct.size());

  auto size = correct.size();
  T max_absolute_error = 0;

  for (auto index = 0; index < size; index++) {
    T absolute_distance = std::abs(result[index] - correct[index]);
    max_absolute_error = std::max(max_absolute_error, absolute_distance);
  }

  return max_absolute_error;
}

///
///@brief Method that calculates the average absolute error between vector components.
/// The vector type must be an arithmetic type. Vectors must have the same size.
///
///@param result Container that stores calculated values
///@param correct Container that stores accurate values
///@return average absolute error
///
template <typename T = double>
double AverageAbsoluteError(const std::vector<T> &result, const std::vector<T> &correct) {
  static_assert(
      std::is_arithmetic_v<T>,
      "mg_test_utility::AverageAbsoluteError expects the vector type to be an arithmetic type.\n");

  assert (result.size() == correct.size());

  auto size = correct.size();
  T manhattan_distance = 0;

  for (auto index = 0; index < size; index++) {
    T absolute_distance = std::abs(result[index] - correct[index]);
    manhattan_distance += absolute_distance;
  }

  double average_absolute_error = 0;
  if (size > 0) average_absolute_error = (double)manhattan_distance / size;
  return average_absolute_error;
}

///
///@brief A method that determines whether given vectors are the same within the defined tolerance.
/// The vector type must be an arithmetic type. Vectors must have the same size.
///
///@param result Container that stores calculated values
///@param correct Container that stores accurate values
///@return true if the maximum absolute error is lesser than ABSOLUTE_ERROR_EPSILON
/// and the average absolute error is lesser than AVERAGE_ABSOLUTE_ERROR_EPSILON,
/// false otherwise
///
template <typename T = double>
bool TestEqualVectors(const std::vector<T> &result, const std::vector<T> &correct) {
  T max_absolute_error = MaxAbsoluteError(result, correct);
  double average_absolute_error = AverageAbsoluteError(result, correct);
  return (max_absolute_error < ABSOLUTE_ERROR_EPSILON && average_absolute_error < AVERAGE_ABSOLUTE_ERROR_EPSILON);
}

}  // namespace mg_test_utility