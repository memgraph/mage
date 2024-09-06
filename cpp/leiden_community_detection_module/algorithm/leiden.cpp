#include <cstddef>
#include <iterator>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

#include "leiden.hpp"
#include "data_structures/graph_view.hpp"

namespace leiden_alg {

const double gamma = 1.0; // TODO: user should be able to set this
const double theta = 1.0; // TODO: user should be able to set this

struct Graph {
  std::size_t num_nodes = 0;
  std::unordered_map<int, std::pair<std::unordered_set<int>, int>> adjacency_list; // node_id -> (neighbors, community_id)

  // Add an edge to the graph
  void addEdge(int u, int v) {
      adjacency_list[u].first.insert(v);
      adjacency_list[v].first.insert(u);
  }

  bool isVertexInGraph(int u) const {
      return adjacency_list.find(u) != adjacency_list.end();
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

bool isSubset(const std::unordered_set<int> &set1, const std::unordered_set<int> &set2) {
    return std::includes(set2.begin(), set2.end(), set1.begin(), set1.end());
}

// Function to count edges between a node and a set of nodes in a community
int countEdgesBetweenNodeAndCommunity(const Graph &G, int node, const std::unordered_set<int> &community) {
  int count = 0;
  for (const auto &neighbor : G.neighbors(node)) {
    if (community.find(neighbor) != community.end()) {
      count++;
    }
  }
  return count;
}

// E(C, S - C)
int countEdgesBetweenCommunities(const Graph &G, const std::unordered_set<int> &community, const std::unordered_set<int> &subset) {
    std::unordered_set<int> set_intersection;
    std::set_difference(community.begin(), community.end(), subset.begin(), subset.end(), std::inserter(set_intersection, set_intersection.begin()));
    int count = 0;
    for (const auto &node : set_intersection) {
        for (const auto &neighbor : G.neighbors(node)) {
            if (subset.find(neighbor) != subset.end()) {
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

void moveNodesFast(Partition &partitions, const Graph &graph) {
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

bool isInSingletonCommunity(const Partition &partitions, int node_id, const Graph &graph) {
  auto community_id = graph.getCommunityForNode(node_id);
  return partitions.at(community_id).size() == 1;
}

Partition mergeNodesSubset(Partition &partitions, const Graph &graph, int subset) {
    std::vector<double> probabilities; // probability of merging with each community
    std::vector<std::pair<int, int>> node_and_community; // node_id, community_id -> corresponding to the probabilities

    // 1 - find well connected nodes within the subset
    std::vector<int> well_connected_nodes;
    auto nodes_in_subset = partitions[subset];
    auto number_of_nodes_in_subset = static_cast<int>(nodes_in_subset.size());
    for (const auto &node_id : nodes_in_subset) {
        int num_edges = countEdgesBetweenNodeAndCommunity(graph, node_id, partitions[subset]);
        auto node_degree = static_cast<int>(graph.neighbors(node_id).size());
        if (num_edges > gamma * node_degree * (number_of_nodes_in_subset - node_degree)) {
            well_connected_nodes.push_back(node_id);
        }
    }
    
    // 2 - find well connected communities to the subset and calculate their probability of merging
    for (const auto &node_id : well_connected_nodes) {
        if (isInSingletonCommunity(partitions, node_id, graph)) { // TODO: check if this should be called on P_refined
            for (auto &community : partitions) {
                auto cpm_before_merge = computeCPM(partitions, graph);
                if (isSubset(community.second, partitions[subset])) {
                    // check if the community is well connected to the subset
                    auto number_of_edges_between_community_and_subset = countEdgesBetweenCommunities(graph, community.second, partitions[subset]);
                    if (number_of_edges_between_community_and_subset >= gamma * community.second.size() * (partitions[subset].size() - community.second.size())) {
                        // community is well connected to the subset -> calculate probability of merging
                        community.second.insert(node_id);
                        auto cpm_after_merge = computeCPM(partitions, graph);
                        community.second.erase(node_id);
                        auto delta_cpm = cpm_after_merge - cpm_before_merge;
                        if (delta_cpm > 0) {
                            auto probability = std::exp(1 / theta * delta_cpm);
                            probabilities.push_back(probability);
                            node_and_community.emplace_back(node_id, community.first);
                        }
                        else {
                            probabilities.push_back(0.0);
                            node_and_community.emplace_back(node_id, community.first);
                        }
                    }
                }
            }
        }
    }

    // 3 - sample from the probabilities and merge the node with the community
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(probabilities.begin(), probabilities.end());

    auto index = d(gen);
    auto [node_id, community_id] = node_and_community[index];
    partitions[community_id].insert(node_id);

    // 4 - erase node from the subset
    partitions[subset].erase(node_id);

    return partitions;
}

Partition refinePartition(Partition &partitions, Graph &graph) {
    auto refined_partitions = singletonPartition(graph);
    for (const auto &[community_id, node_ids] : partitions) {
        refined_partitions = mergeNodesSubset(refined_partitions, graph, community_id);
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

Partition leiden(const mg_graph::GraphView<> &graph) {
    Graph G;
    for (const auto &node : graph.Nodes()) {
        G.num_nodes++;
        for (const auto &neighbor : graph.Neighbours(node.id)) {
            G.addEdge(node.id, neighbor.node_id);
        }
    }
    
    auto partitions = singletonPartition(G);
    bool done = false;
    while(!done) {
        moveNodesFast(partitions, G);
        done = partitions.size() == G.size();
        if (!done) {
            auto refined_partitions = refinePartition(partitions, G);
            aggregateGraph(refined_partitions, G);

            // get new partitions
            for (const auto &[community_id, node_ids] : partitions) {
                for (const auto &node_id : node_ids) {
                    if (!G.isVertexInGraph(node_id)) {
                        partitions[community_id].erase(node_id);
                    }
                }
            }
        }
    }

    return partitions;
}

std::vector<std::int64_t> GetCommunities(const mg_graph::GraphView<> &graph) {
    auto partitions = leiden(graph);
    std::vector<std::int64_t> communities(graph.Nodes().size());
    for (const auto &[community_id, node_ids] : partitions) {
        for (const auto &node_id : node_ids) {
            communities[node_id] = community_id;
        }
    }
    return communities;
}

}  // namespace leiden_alg
