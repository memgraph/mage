#include <atomic>
#include <future>
#include <queue>
#include <stack>
#include <vector>

#include "betweenness_centrality.hpp"

namespace betweenness_centrality_util {

void BFS(std::uint64_t source_node, const mg_graph::GraphView<> &graph, std::stack<std::uint64_t> &visited,
         std::vector<std::vector<std::uint64_t>> &predecessors, std::vector<std::uint64_t> &shortest_paths_counter, std::vector<int> &distance) {

  shortest_paths_counter[source_node] = 1;
  distance[source_node] = 0;

  std::queue<std::uint64_t> BFS_queue;
  BFS_queue.push(source_node);

  while (!BFS_queue.empty()) {
    auto current_node_id = BFS_queue.front();
    BFS_queue.pop();
    visited.push(current_node_id);

    for (auto neighbor : graph.Neighbours(current_node_id)) {
      auto neighbor_id = neighbor.node_id;

      // node found for the first time
      if (distance[neighbor_id] < 0) {
        BFS_queue.push(neighbor_id);
        distance[neighbor_id] = distance[current_node_id] + 1;
      }

      // shortest path from node to neighbor_id goes through current_node
      if (distance[neighbor_id] == distance[current_node_id] + 1) {
        shortest_paths_counter[neighbor_id] += shortest_paths_counter[current_node_id];
        predecessors[neighbor_id].emplace_back(current_node_id);
      }
    }
  }
}
}  // namespace betweenness_centrality_util

namespace {
///
///@brief An in-place method that normalizes a vector by multiplying each component by a given constant.
///
///@param vec The vector that should be normalized
///@param constant The constant with which the components of a vector are multiplied
///
void Normalize(std::vector<double> &vec, double constant) {
  for (auto &value : vec) value *= constant;
}
}  // namespace

namespace betweenness_centrality_alg {

  std::vector<double> BetweennessCentrality(const mg_graph::GraphView<>& graph, bool directed, bool normalize, int threads) {
    const auto number_of_nodes = graph.Nodes().size();
    std::vector<std::atomic<double>> betweenness_centrality(number_of_nodes);
    for (auto& bc : betweenness_centrality) bc.store(0.0, std::memory_order_relaxed);
  
    auto process_nodes = [&](std::uint64_t start, std::uint64_t end) {
      // Thread-local BFS resources
      std::vector<std::uint64_t> shortest_paths_counter;
      std::vector<int> distance;
      std::vector<std::vector<std::uint64_t>> predecessors;
      std::vector<double> dependency;
      std::stack<std::uint64_t> visited;
  
      for (auto node_id = start; node_id < end; ++node_id) {
        // Reset BFS state
        shortest_paths_counter.assign(number_of_nodes, 0);
        distance.assign(number_of_nodes, -1);
        predecessors.assign(number_of_nodes, {});
        dependency.assign(number_of_nodes, 0.0);
        visited = {};
  
        // Execute BFS
        shortest_paths_counter[node_id] = 1;
        distance[node_id] = 0;
        betweenness_centrality_util::BFS(node_id, graph, visited, predecessors, shortest_paths_counter, distance);
  
        // Calculate dependencies
        while (!visited.empty()) {
          const auto current_node = visited.top();
          visited.pop();
  
          for (const auto p : predecessors[current_node]) {
            const double fraction = static_cast<double>(shortest_paths_counter[p]) / 
                                  static_cast<double>(shortest_paths_counter[current_node]);
            dependency[p] += fraction * (1 + dependency[current_node]);
          }
  
          if (current_node != node_id) {
            const double value = directed ? dependency[current_node] : dependency[current_node] / 2.0;
            betweenness_centrality[current_node].fetch_add(value, std::memory_order_relaxed);
          }
        }
      }
    };
  
    // Distribute work across threads
    const auto nodes_per_thread = (number_of_nodes + threads - 1) / threads;
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < threads; ++i) {
      const auto start = i * nodes_per_thread;
      const auto end = std::min((i + 1) * nodes_per_thread, number_of_nodes);
      if (start >= number_of_nodes) break;
      
      futures.emplace_back(std::async(std::launch::async, process_nodes, start, end));
    }
  
    // Wait for completion
    for (auto& future : futures) future.wait();
  
    // Convert to regular vector
    std::vector<double> result;
    result.reserve(number_of_nodes);
    for (const auto& bc : betweenness_centrality) {
      result.push_back(bc.load(std::memory_order_relaxed));
    }
  
    // Normalization remains unchanged
    if (normalize) {
      const auto number_of_pairs = (number_of_nodes - 1) * (number_of_nodes - 2);
      const auto numerator = directed ? 1.0 : 2.0;
      const double constant = number_of_nodes > 2 ? numerator / static_cast<double>(number_of_pairs) : 1.0;
      Normalize(result, constant);
    }
  
    return result;
  }

}  // namespace betweenness_centrality_alg
