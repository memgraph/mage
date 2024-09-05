#include "leiden.hpp"
#include <queue>
#include <random>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <type_traits>
#include "data_structures/graph_view.hpp"

namespace leiden_alg {

const double gamma = 1.0;

// TODO: Optimize this function
std::int64_t BinomialCoefficient(const int n, const int k) {
  std::vector<int> aSolutions(k);
  aSolutions[0] = n - k + 1;

  for (int i = 1; i < k; ++i) {
    aSolutions[i] = aSolutions[i - 1] * (n - k + 1 + i) / (i + 1);
  }

  return aSolutions[k - 1];
}

double computeCPM (std::unordered_map<std::int64_t, std::int64_t> &partitions, const mg_graph::GraphView<> &graph) {
    double h = 0.0;
    std::unordered_map<std::int64_t, std::int64_t> community_sizes;
    std::unordered_map<std::int64_t, std::int64_t> community_edges;

    for (const auto &[node_id, community_id] : partitions) {
        community_edges[community_id] = 0;
        community_sizes[community_id] = 0;
    }

    for (const auto &[id, from, to] : graph.Edges()) {
        auto from_partition = partitions[from];
        auto to_partition = partitions[to];
        if (from_partition == to_partition) {
            community_edges[from_partition]++;
        }
        community_sizes[from_partition]++;
        community_sizes[to_partition]++;
    }

    for (const auto& [community, num_edges] : community_edges) {
        auto community_size = community_sizes[community] / 2;
        h += num_edges - gamma * BinomialCoefficient(community_size, 2);
    }
    return h;
}

std::unordered_map<std::int64_t, std::int64_t> moveNodesFast(std::unordered_map<std::int64_t, std::int64_t> &partitions, const mg_graph::GraphView<> &graph) {
    std::queue<std::int64_t> nodes;
    for (std::int64_t i = 0; i < graph.Nodes().size(); i++) {
        nodes.push(i);
    }

    while(!nodes.empty()) {
      auto node_id = nodes.front();
      nodes.pop();
      auto best_community = partitions[node_id];
      auto best_delta = computeCPM(partitions, graph);
      for (const auto &neighbour : graph.Neighbours(node_id)) {
          auto &current_community = partitions[node_id];
          current_community = neighbour.node_id;
          auto delta = computeCPM(partitions, graph);
          if (delta > best_delta) {
              best_delta = delta;
              best_community = neighbour.node_id;
              // find neighbours that are not in this community and add them to the queue
              for (const auto &neighbour : graph.Neighbours(node_id)) {
                  if (partitions[neighbour.node_id] == current_community) {
                      nodes.push(neighbour.node_id);
                  }
              }
          }
          else {
              current_community = best_community;
          }
      }
    }
}

}  // namespace leiden_alg
