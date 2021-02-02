/// @file

#pragma once

#include <vector>

namespace algorithms {

/// Simple struct that keeps the state of nodes in algorithms that rely on
/// DFS traversal.
struct NodeState {
  std::vector<bool> visited;
  std::vector<int32_t> discovery, low_link, depth;
  std::vector<uint32_t> parent;
};
}  // namespace algorithms
