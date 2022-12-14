#include <omp.h>
#include <queue>
#include <stack>
#include <vector>
#include <chrono>
using namespace std::chrono;

#include "betweenness_centrality.hpp"

namespace betweenness_centrality_util {

void BFS(const std::uint64_t source_node,std::unordered_map<std::uint64_t,std::vector<std::uint64_t>>& adj_matrix, std::stack<std::uint64_t> &visited,
         std::unordered_map<std::uint64_t,std::vector<std::uint64_t>> &predecessors, std::unordered_map<std::uint64_t,std::uint64_t> &shortest_paths_counter) {
  // -1 to indicate that node is not visited
  std::unordered_map<std::int64_t,int> distance;

  shortest_paths_counter[source_node] = 1;
  std::queue<std::uint64_t> BFS_queue;
  BFS_queue.push(source_node);
  while (!BFS_queue.empty()) {
    auto current_node_id = BFS_queue.front();
    BFS_queue.pop();
    visited.push(current_node_id);
    for (auto& neighbor_id : adj_matrix[current_node_id]) {
      // node found for the first time
      if (distance.find(neighbor_id) == distance.end()) {
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

std::map<std::uint64_t,double> BetweennessCentrality(std::unordered_map<std::uint64_t,std::vector<std::uint64_t>>& adj_matrix, bool directed, bool normalize,
                                          int threads) {
  std::map<std::uint64_t,double> betweenness_centrality{};
  std::vector<std::uint64_t>keys;
  for(const auto &entries:adj_matrix){
    keys.push_back(entries.first);
  }

  // perform bfs for every node in the graph
  omp_set_dynamic(0);
  omp_set_num_threads(threads);  
#pragma omp parallel for shared(betweenness_centrality)
    for(auto i = 0; i < keys.size(); i++) {
    auto node_id = keys[i];
    // data structures used in BFS
    std::stack<std::uint64_t> visited;
    std::unordered_map<std::uint64_t,std::vector<std::uint64_t>> predecessors;
    std::unordered_map<std::uint64_t,std::uint64_t> shortest_paths_counter;
    auto start = steady_clock::now();
    betweenness_centrality_util::BFS(node_id, adj_matrix, visited, predecessors, shortest_paths_counter);
    auto end = duration_cast<milliseconds>(steady_clock::now() - start).count();

    std::map<std::uint64_t,double> dependency;

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

  // if (normalize) {
  //   // normalized by dividing the value by the number of pairs of nodes
  //   // not including the node whose value we normalize
  //   auto number_of_pairs = (number_of_nodes - 1) * (number_of_nodes - 2);
  //   const auto numerator = directed ? 1.0 : 2.0;
  //   double constant = number_of_nodes > 2 ? numerator / number_of_pairs : 1.0;
  //   Normalize(betweenness_centrality, constant);
  // }

  return betweenness_centrality;
}

}  // namespace betweenness_centrality_alg
