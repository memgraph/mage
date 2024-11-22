#include <mgp.hpp>

#include <omp.h>
#include <queue>
#include <stack>
#include <vector>

#include "betweenness_centrality.hpp"

namespace betweenness_centrality_util {

void BFS(const std::uint64_t source_node, const mg_graph::GraphView<> &graph, std::stack<std::uint64_t> &visited,
         std::vector<std::vector<std::uint64_t>> &predecessors, std::vector<std::uint64_t> &shortest_paths_counter) {
  auto number_of_nodes = graph.Nodes().size();

  // -1 to indicate that node is not visited
  std::vector<int> distance(number_of_nodes, -1);

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

std::vector<double> BetweennessCentrality(const mg_graph::GraphView<> &graph, bool directed, bool normalize,
                                          int threads) {
  auto number_of_nodes = graph.Nodes().size();
  std::vector<double> betweenness_centrality(number_of_nodes, 0);

  // perform bfs for every node in the graph
  omp_set_dynamic(0);
  omp_set_num_threads(threads);
#pragma omp parallel for
  for (std::uint64_t node_id = 0; node_id < number_of_nodes; node_id++) {
    // data structures used in BFS
    std::stack<std::uint64_t> visited;
    std::vector<std::vector<std::uint64_t>> predecessors(number_of_nodes, std::vector<std::uint64_t>());
    std::vector<std::uint64_t> shortest_paths_counter(number_of_nodes, 0);
    betweenness_centrality_util::BFS(node_id, graph, visited, predecessors, shortest_paths_counter);

    std::vector<double> dependency(number_of_nodes, 0);

    while (!visited.empty()) {
      auto current_node = visited.top();
      visited.pop();

      for (auto p : predecessors[current_node]) {
        double fraction = static_cast<double>(shortest_paths_counter[p]) / shortest_paths_counter[current_node];
        dependency[p] += fraction * (1 + dependency[current_node]);
      }

      if (current_node != node_id) {
        if (directed) {
#pragma omp atomic update
          betweenness_centrality[current_node] += dependency[current_node];
        }
        // centrality scores need to be divided by two since all shortest paths are considered twice
        else {
#pragma omp atomic update
          betweenness_centrality[current_node] += dependency[current_node] / 2.0;
        }
      }
    }
  }

  if (normalize) {
    // normalized by dividing the value by the number of pairs of nodes
    // not including the node whose value we normalize
    auto number_of_pairs = (number_of_nodes - 1) * (number_of_nodes - 2);
    const auto numerator = directed ? 1.0 : 2.0;
    double constant = number_of_nodes > 2 ? numerator / number_of_pairs : 1.0;
    Normalize(betweenness_centrality, constant);
  }

  return betweenness_centrality;
}

}  // namespace betweenness_centrality_alg
