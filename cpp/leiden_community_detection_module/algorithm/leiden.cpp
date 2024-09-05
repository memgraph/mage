#include "leiden.hpp"
#include <cstddef>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>
#include "data_structures/graph_view.hpp"

namespace leiden_alg {

struct Graph {
  std::size_t num_nodes;
  std::unordered_map<std::int64_t, std::unordered_set<std::int64_t>> adjacency_list;

  // Add an edge to the graph
  void addEdge(std::int64_t u, std::int64_t v) {
      adjacency_list[u].insert(v);
      adjacency_list[v].insert(u);
  }

  std::size_t size() const {
      return num_nodes;
  }

  const std::unordered_set<std::int64_t> &neighbors(std::int64_t u) const {
      return adjacency_list.at(u);
  }
};

const double gamma = 1.0;

// Function to count edges between a node and a set of nodes in a community
int countEdgesBetween(const Graph &G, int node, const std::unordered_set<int> &community) {
    int count = 0;
    for (int neighbor : G.neighbors(node)) {
        if (community.count(neighbor)) {
            count++;
        }
    }
    return count;
}

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

std::unordered_map<std::int64_t, std::int64_t> moveNodesFast(std::unordered_map<std::int64_t, std::int64_t> &partitions, const mg_graph::GraphView<> &memgraph_graph, const Graph &graph) {
    std::queue<std::int64_t> nodes;
    std::unordered_set<std::int64_t> nodes_set;
    for (std::int64_t i = 0; i < graph.size(); i++) {
        nodes.push(i);
        nodes_set.insert(i);
    }

    while(!nodes.empty()) {
      auto node_id = nodes.front();
      nodes.pop();
      nodes_set.erase(node_id);
      auto best_community = partitions[node_id];
      auto best_delta = computeCPM(partitions, memgraph_graph); // fix later
      for (const auto &neighbour : graph.neighbors(node_id)) {
          auto &current_community = partitions[node_id];
          current_community = neighbour;
          auto delta = computeCPM(partitions, memgraph_graph); // fix later
          if (delta > best_delta) {
              best_delta = delta;
              best_community = neighbour;
              // find neighbours that are not in this community and in the queue and add them
              for (const auto &neighbour2 : graph.neighbors(node_id)) {
                  if (partitions[neighbour2] != best_community && nodes_set.find(neighbour2) == nodes_set.end()) {
                      nodes.push(neighbour2);
                      nodes_set.insert(neighbour2);
                  }
              }
          }
          else {
              current_community = best_community;
          }
      }
    }
}

std::unordered_map<std::int64_t, std::int64_t> singletonPartition(const Graph &graph) {
    std::unordered_map<std::int64_t, std::int64_t> partitions;
    for (std::int64_t i = 0; i < graph.size(); i++) {
        partitions[i] = i;
    }
    return partitions;
}

std::unordered_map<std::int64_t, std::int64_t> mergeNodesSubset(std::unordered_map<std::int64_t, std::int64_t> &partitions, const Graph &graph, std::int64_t subset) {
    // TODO: implement this function
}

std::unordered_map<std::int64_t, std::int64_t> refinePartition(std::unordered_map<std::int64_t, std::int64_t> &partitions, Graph &graph) {
  auto refined_partitions = singletonPartition(graph);
  for (const auto &[node_id, community_id] : partitions) {
      // TODO: merge nodes subset function
  }
  return refined_partitions;
}


void aggregateGraph (std::unordered_map<std::int64_t, std::int64_t> &partitions, Graph &graph) {
  // communities becomes the new nodes
  std::unordered_map<std::int64_t, std::unordered_set<std::int64_t>> new_adjacency_list;
  for (const auto &[node_id, community_id] : partitions) {
      new_adjacency_list[community_id] = std::unordered_set<std::int64_t>();
  }

  for (const auto &[node_id, community_id] : partitions) {
      for (const auto &neighbour : graph.neighbors(node_id)) {
          if (partitions[neighbour] != community_id) {
              new_adjacency_list[community_id].insert(partitions[neighbour]);
          }
      }
  }

  graph.adjacency_list = std::move(new_adjacency_list);
}


}  // namespace leiden_alg
