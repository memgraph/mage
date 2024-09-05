#include "leiden.hpp"
#include <cstddef>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>
#include "data_structures/graph_view.hpp"

namespace leiden_alg {

const double gamma = 1.0; // TODO: user should be able to set this

struct Graph {
  std::size_t num_nodes;
  std::unordered_map<int, std::pair<std::unordered_set<int>, int>> adjacency_list; // node_id -> (neighbors, community_id)
  // Add an edge to the graph
  void addEdge(int u, int v) {
      adjacency_list[u].first.insert(v);
      adjacency_list[v].first.insert(u);
  }

  std::size_t size() const {
      return num_nodes;
  }

  const std::unordered_set<int> &neighbors(int u) const {
      return adjacency_list.at(u).first;
  }
  
   int getCommunityForNode(int u) const {
      return adjacency_list.at(u).second;
  }
};

using Partition = std::unordered_map<int, std::unordered_set<int>>; // community_id -> node_ids within the community

// Function to count edges between a node and a set of nodes in a community
int countEdgesBetween(const Graph &G, int node, Partition &partitions, int &number_of_nodes_in_subset) {
    int count = 0;
    for (const auto &[neighbor, community] : partitions) {
        if (community == partitions[node]) {
            number_of_nodes_in_subset++;
            if (G.adjacency_list.at(node).first.find(neighbor) != G.adjacency_list.at(node).first.end()) {
                count++;
            }
        }
    }
    return count;
}

// TODO: Optimize this function
int BinomialCoefficient(const int n, const int k) {
  std::vector<int> aSolutions(k);
  aSolutions[0] = n - k + 1;

  for (int i = 1; i < k; ++i) {
    aSolutions[i] = aSolutions[i - 1] * (n - k + 1 + i) / (i + 1);
  }

  return aSolutions[k - 1];
}

double computeCPM (Partition &partitions, const Graph &graph) {
    double h = 0.0;

    // compute sum of edges within communities
    for (const auto &[community_id, node_ids] : partitions) {
        int num_edges = 0;
        for (const auto &node_id : node_ids) {
            for (const auto &neighbor : graph.neighbors(node_id)) {
                if (partitions[community_id].find(neighbor) != partitions[community_id].end()) {
                    num_edges++;
                }
            }
        }
        int community_size = static_cast<int>(node_ids.size());
        h += num_edges - gamma * BinomialCoefficient(community_size, 2);
    }

    return h;
}

Partition moveNodesFast(Partition &partitions, const Graph &graph) {
    std::queue<int> nodes;
    std::unordered_set<int> nodes_set;
    for (int i = 0; i < graph.size(); i++) {
        nodes.push(i);
        nodes_set.insert(i);
    }

    while(!nodes.empty()) {
        auto node_id = nodes.front();
        nodes.pop();
        nodes_set.erase(node_id);
        auto best_community = graph.getCommunityForNode(node_id);
        auto best_delta = computeCPM(partitions, graph); // fix later
        for (const auto &neighbor : graph.neighbors(node_id)) {
            auto delta = computeCPM(partitions, graph); // fix later
            if (delta > best_delta) {
                best_delta = delta;
                best_community = neighbor;
                // find neighbors that are not in this community and in the queue and add them
                for (const auto &neighbor2 : graph.neighbors(node_id)) {
                    if (nodes_set.find(neighbor2) == nodes_set.end() && graph.getCommunityForNode(neighbor2) != best_community) {
                        nodes.push(neighbor2);
                        nodes_set.insert(neighbor2);
                    }
                }
            }
        }
        partitions[best_community].insert(node_id);
    }
}

Partition singletonPartition(const Graph &graph) {
  Partition partitions;
  for (int i = 0; i < graph.size(); i++) {
      partitions[i].insert(i);
  }
  return partitions;
}

// Partition mergeNodesSubset(Partition &partitions, const Graph &graph, int subset) {
//   // 1 - find well connected nodes within the subset
//   std::vector<int> well_connected_nodes;
//   for (const auto &[node_id, community_id] : partitions) {
//     if (community_id == subset) {
//       int num_of_nodes_in_subset = 0;
//       int num_edges = countEdgesBetween(graph, node_id, partitions, num_of_nodes_in_subset);
//       std::size_t degree = graph.adjacency_list.at(node_id).size();
//       if (num_edges >= gamma * degree * (num_of_nodes_in_subset - degree)) {
//         well_connected_nodes.push_back(node_id);
//       }
//     }
//     break;
//   }
//   for (const auto &node : well_connected_nodes) {
//     // 2 
//   }
// }

Partition refinePartition(Partition &partitions, Graph &graph) {
  auto refined_partitions = singletonPartition(graph);
  for (const auto &[node_id, community_id] : partitions) {
      // TODO: merge nodes subset function
  }
  return refined_partitions;
}


void aggregateGraph (Partition &partitions, Graph &graph) {
  // communities becomes the new nodes
  std::unordered_map<int, std::pair<std::unordered_set<int>, int>> new_adjacency_list;
  new_adjacency_list.reserve(partitions.size());
  for (const auto &[community_id, _] : partitions) {
      new_adjacency_list[community_id] = {};
  }

  for (const auto &[node_id, _] : graph.adjacency_list) {
    for (const auto &neighbor : graph.neighbors(node_id)) {
        auto community_id_node = graph.adjacency_list.at(node_id).second;
        auto community_id_neighbor = graph.adjacency_list.at(neighbor).second;
        if (community_id_node != community_id_neighbor) {
            new_adjacency_list[node_id] = std::make_pair(std::unordered_set<int>{community_id_neighbor}, community_id_node);
        }
    }
  }

  graph.adjacency_list = std::move(new_adjacency_list);
}


}  // namespace leiden_alg
