#pragma once

#include <cassert>
#include <chrono>
#include <cmath>
#include <stack>
#include <vector>

namespace mg_test_utility {

constexpr double ABSOLUTE_ERROR_EPSILON{10e-4};
constexpr double AVERAGE_ABSOLUTE_ERROR_EPSILON{10e-5};

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
template <typename T>
T MaxAbsoluteError(const std::vector<T> &result, const std::vector<T> &correct) {
  static_assert(std::is_arithmetic_v<T>,
                "mg_test_utility::MaxAbsoluteError expects the vector type to be an arithmetic type.\n");

  assert(result.size() == correct.size());

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
template <typename T>
double AverageAbsoluteError(const std::vector<T> &result, const std::vector<T> &correct) {
  static_assert(std::is_arithmetic_v<T>,
                "mg_test_utility::AverageAbsoluteError expects the vector type to be an arithmetic type.\n");

  assert(result.size() == correct.size());

  auto size = correct.size();
  T manhattan_distance = 0;

  for (auto index = 0; index < size; index++) {
    T absolute_distance = std::abs(result[index] - correct[index]);
    manhattan_distance += absolute_distance;
  }

  double average_absolute_error = 0;
  if (size > 0) average_absolute_error = static_cast<double>(manhattan_distance) / size;
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
template <typename T>
bool TestEqualVectors(const std::vector<T> &result, const std::vector<T> &correct,
                      const double absolute_error = ABSOLUTE_ERROR_EPSILON,
                      double average_error = AVERAGE_ABSOLUTE_ERROR_EPSILON) {
  auto max_absolute_error = MaxAbsoluteError(result, correct);
  if (max_absolute_error >= absolute_error) return false;

  auto average_absolute_error = AverageAbsoluteError(result, correct);
  if (average_absolute_error >= average_error) return false;

  return true;
}

///
///@brief A method that determines whether given vectors with (int, arithmetic_type) values are the same within the
/// defined tolerance. Vectors must have the same size.
///
///@param result Container that stores calculated values
///@param correct Container that stores accurate values
///@param absolute_error Absolute error between any two pairs of resulting and correct value
///@return true if the maximum absolute error is lesser than ABSOLUTE_ERROR_EPSILON
/// false otherwise
///
template <typename T>
bool TestEqualVectorPairs(std::vector<std::pair<std::uint64_t, T>> &result,
                          std::vector<std::pair<std::uint64_t, T>> &correct,
                          const double absolute_error = ABSOLUTE_ERROR_EPSILON) {
  if (result.size() != correct.size()) return false;

  std::sort(result.begin(), result.end());
  std::sort(correct.begin(), correct.end());

  auto size = result.size();
  for (std::size_t i = 0; i < size; i++) {
    auto [result_id, result_value] = result[i];
    auto [correct_id, correct_value] = correct[i];

    if (result_id != correct_id) return false;
    if (std::abs(result_value - correct_value) > absolute_error) return false;
  }

  return true;
}
}  // namespace mg_test_utility
