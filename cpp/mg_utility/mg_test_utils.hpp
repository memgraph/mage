#pragma once

#include <chrono>

namespace mg_test_utility {
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

}  // namespace mg_test_utility