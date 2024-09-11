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

const double gamma = 0.25; // TODO: user should be able to set this
const double theta = 0.01; // TODO: user should be able to set this

struct Graph {
  std::size_t num_nodes = 0;
  std::vector<std::vector<int>> adjacency_list; // node_id -> neighbors
  // Add an edge to the graph
  void addEdge(int u, int v) {
    if (u >= adjacency_list.size()) {
        adjacency_list.resize(u + 1);
    }
    if (v >= adjacency_list.size()) {
        adjacency_list.resize(v + 1);
    }
    adjacency_list[u].push_back(v);
    adjacency_list[v].push_back(u);
  }

  bool isVertexInGraph(int u) const {
      return u < adjacency_list.size();
    }

  std::size_t size() const {
      return num_nodes;
  }

  const std::vector<int> &neighbors(int u) const {
    return adjacency_list[u];
  }
};

struct Partitions {
    std::vector<std::vector<int>> communities; // community_id -> node_ids within the community
    std::vector<int> community_id; // node_id -> community_id
    std::vector<int> community_weights; // community_id -> weight

    int getCommunityForNode(int node_id) const {
        return community_id[node_id];
    }

    void updateWeightForCommunity(int community_id, int weight_update = 1) {
        community_weights[community_id] += weight_update;
    }
};

bool isSubset(const std::vector<int> &set1, const std::vector<int> &set2) {
    return std::includes(set2.begin(), set2.end(), set1.begin(), set1.end());
}

// |E(node, C \ node)| -> fix this
int countEdgesBetweenNodeAndCommunity(const Graph &graph, int node_id, int community_id, const Partitions &partitions) {
  int count = 0;
  for (const auto &neighbor : graph.neighbors(node_id)) {
    if (partitions.getCommunityForNode(neighbor) == community_id) {
      count++;
    }
  }
  return count;
}

// |E(C, S \ C)|
int countEdgesBetweenCommunities(const Graph &graph, const std::vector<int> &community, const std::vector<int> &subset, const Partitions &partitions) {
    std::vector<int> set_intersection;
    std::set_difference(community.begin(), community.end(), subset.begin(), subset.end(), std::inserter(set_intersection, set_intersection.begin()));
    int count = 0;
    for (const auto &node : set_intersection) {
        count += countEdgesBetweenNodeAndCommunity(graph, node, partitions.getCommunityForNode(node), partitions);
    }
    return count;
}

int getNumOfPossibleEdges(int n) {
    return n * (n - 1) / 2;
}

double computeCPM (Partitions &partitions, int weight_update = 0, int updated_community = -1) {
    double result = 0.0;
    for (auto i = 0; i < partitions.communities.size(); i++) {
        if (updated_community != i) {
            const auto &community_size = static_cast<int>(partitions.communities[i].size());
            result += partitions.community_weights[i] - gamma * getNumOfPossibleEdges(community_size);
        }
        else {
            const auto &community_size = static_cast<int>(partitions.communities[i].size() + 1);
            result += (partitions.community_weights[i] + weight_update) - gamma * getNumOfPossibleEdges(community_size);
        }
    }
    return result;
}

// Move a node to a new community and returns set of neighbors that are not in the new community and the weight update
std::pair<std::vector<int>, int> moveNode(const std::vector<int> &community, int node_id, int new_community, Partitions &partitions, const Graph &graph, std::unordered_set<int> *processed_nodes = nullptr) {
   std::vector<int> neighbors_in_different_community;
   int weight_update = 0;
   neighbors_in_different_community.reserve(community.size());
   for (const auto &neighbor : graph.neighbors(node_id)) {
        if (partitions.getCommunityForNode(neighbor) == new_community) {
            weight_update++;
        }
        else {
            if (processed_nodes && processed_nodes->find(neighbor) == processed_nodes->end()) {
                neighbors_in_different_community.push_back(neighbor);
            }
        }
   }

   return {neighbors_in_different_community, weight_update};
}

std::pair<double, int> computeDeltaCPM(const Partitions &partitions, const int node_id, const int new_community_id, const Graph &graph) {
    double result = 0.0;
    
    const auto current_community_id = partitions.getCommunityForNode(node_id);
    const auto current_community_size = static_cast<int>(partitions.communities[current_community_id].size());
    const auto new_community_size = static_cast<int>(partitions.communities[new_community_id].size());
    const auto current_community_weight = partitions.community_weights[current_community_id];
    const auto new_community_weight = partitions.community_weights[new_community_id];

    const auto source_weight = current_community_weight - gamma * getNumOfPossibleEdges(current_community_size);
    const auto target_weight = new_community_weight - gamma * getNumOfPossibleEdges(new_community_size);

    // TODO: this can be done in one function call -> not that important for now
    const auto num_edges_between_node_and_current_community = countEdgesBetweenNodeAndCommunity(graph, node_id, current_community_id, partitions);
    const auto num_edges_between_node_and_new_community = countEdgesBetweenNodeAndCommunity(graph, node_id, new_community_id, partitions);

    const auto new_source_weight = current_community_weight + num_edges_between_node_and_new_community - gamma * getNumOfPossibleEdges(current_community_size + 1);
    const auto new_target_weight = new_community_weight - num_edges_between_node_and_current_community - gamma * getNumOfPossibleEdges(new_community_size - 1);

    result = new_source_weight + new_target_weight - source_weight - target_weight;
    return {result, num_edges_between_node_and_current_community};
}

void moveNodesFast(Partitions &partitions, Graph &graph) {
    std::deque<int> nodes;
    std::unordered_set<int> nodes_set;
    for (int i = 0; i < graph.size(); i++) {
        nodes.push_back(i);
        nodes_set.insert(i);
    }

    std::shuffle(nodes.begin(), nodes.end(), std::mt19937(std::random_device()()));

    while(!nodes.empty()) {
        auto node_id = nodes.front();
        nodes.pop_front();
        nodes_set.erase(node_id);
        auto best_community = partitions.getCommunityForNode(node_id);
        double best_delta = 0;
        int weight_update = 0;

        for (const auto neighbor_id : graph.neighbors(node_id)) {
            const auto community_id_of_neighbor = partitions.getCommunityForNode(neighbor_id);
            if (community_id_of_neighbor != best_community) {
                const auto result = computeDeltaCPM(partitions, node_id, community_id_of_neighbor, graph);
                if (result.first > best_delta) {
                    best_delta = result.first;
                    best_community = community_id_of_neighbor;
                    weight_update = result.second;
                }
            }
        }

        if (best_delta > 0) {
            partitions.communities[best_community].push_back(node_id);
            partitions.community_id[node_id] = best_community;
            partitions.updateWeightForCommunity(best_community, weight_update);
            partitions.communities[partitions.getCommunityForNode(node_id)].erase(std::remove(partitions.communities[partitions.getCommunityForNode(node_id)].begin(), partitions.communities[partitions.getCommunityForNode(node_id)].end(), node_id), partitions.communities[partitions.getCommunityForNode(node_id)].end());

            // add neighbors to the queue
            for (const auto neighbor_id : graph.neighbors(node_id)) {
                if (nodes_set.find(neighbor_id) != nodes_set.end()) {
                    nodes.push_back(neighbor_id);
                    nodes_set.insert(neighbor_id);
                }
            }
        }
    }
}

Partitions singletonPartition(const Graph &graph) {
  Partitions partitions;
  for (int i = 0; i < graph.size(); i++) {
    partitions.communities.push_back({i});
    partitions.community_id.push_back(i);
    partitions.community_weights.push_back(0);
  }
  return partitions;
}

bool isInSingletonCommunity(const Partitions &partitions, int node_id, const Graph &graph) {
    auto community_id = partitions.getCommunityForNode(node_id);
    return partitions.communities[community_id].size() == 1;
}

bool isWellConnectedCommunity(const std::vector<int> &community, const Partitions &partitions, const Graph &graph, int subset) {
    if (isSubset(community, partitions.communities[subset])) {
        // check if the community is well connected to the subset
        auto number_of_edges_between_community_and_subset = countEdgesBetweenCommunities(graph, community, partitions.communities[subset], partitions);
        if (number_of_edges_between_community_and_subset >= gamma * static_cast<double>(community.size()) * static_cast<double>(partitions.communities[subset].size() - community.size())) {
            return true;
        }
    }
    return false;
}

Partitions mergeNodesSubset(Partitions &refined_partitions, const Graph &graph, int subset, Partitions &partitions) {
    std::vector<double> probabilities; // probability of merging with each community
    std::vector<std::pair<int, int>> node_and_community; // node_id, community_id -> corresponding to the probabilities
    int weight_update = 0;

    // 1 - find well connected nodes within the subset
    std::vector<int> well_connected_nodes;
    auto nodes_in_subset = partitions.communities[subset];
    auto number_of_nodes_in_subset = static_cast<int>(nodes_in_subset.size());
    for (const auto &node_id : nodes_in_subset) {
        const auto num_edges = countEdgesBetweenNodeAndCommunity(graph, node_id, subset, partitions);
        const auto node_degree = static_cast<int>(graph.neighbors(node_id).size());
        if (num_edges >= gamma * node_degree * (number_of_nodes_in_subset - node_degree)) {
            well_connected_nodes.push_back(node_id);
        }
    }

    const auto cpm_before_merge = computeCPM(refined_partitions);
    
    // 2 - find well connected communities to the subset and calculate their probability of merging
    for (const auto &node_id : well_connected_nodes) {
        if (isInSingletonCommunity(refined_partitions, node_id, graph)) { 
            for (auto i = 0; i < refined_partitions.communities.size(); i++) {
                if (isWellConnectedCommunity(refined_partitions.communities[i], partitions, graph, subset)) {
                    auto nodes_for_queue_and_weight_update = moveNode(partitions.communities[subset], node_id, i, partitions, graph);
                    auto cpm = computeCPM(partitions, nodes_for_queue_and_weight_update.second);
                    auto delta_cpm = cpm - cpm_before_merge;
                    if (delta_cpm > 0) {
                        auto probability = std::exp(1 / theta * delta_cpm);
                        probabilities.push_back(probability);
                        node_and_community.emplace_back(node_id, refined_partitions.community_id[i]);
                        weight_update = nodes_for_queue_and_weight_update.second;
                    }
                    else {
                        probabilities.push_back(0.0);
                        node_and_community.emplace_back(node_id, refined_partitions.community_id[i]);
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
    refined_partitions.communities[community_id].push_back(node_id);
    refined_partitions.community_id[node_id] = community_id;
    if (weight_update > 0) {
        refined_partitions.updateWeightForCommunity(community_id, weight_update);
    }

    // 4- remove the node from the subset
    const auto current_community = partitions.getCommunityForNode(node_id);
    refined_partitions.communities[current_community].erase(std::remove(refined_partitions.communities[current_community].begin(), refined_partitions.communities[current_community].end(), node_id), refined_partitions.communities[current_community].end());

    return refined_partitions;
}

Partitions refinePartition(Partitions &partitions, Graph &graph) {
    auto refined_partitions = singletonPartition(graph);
    for (auto i = 0; i < partitions.communities.size(); i++) {
        if (partitions.communities[i].size() > 1) {
            refined_partitions = mergeNodesSubset(refined_partitions, graph, i, partitions);
        }
    }
    return refined_partitions;
}

// communities becomes the new nodes
void aggregateGraph (const Partitions &partitions, Graph &graph) {
  std::vector<std::vector<int>> new_adjacency_list;
  new_adjacency_list.reserve(partitions.communities.size());

  for (auto i = 0; i < partitions.communities.size(); i++) {
    new_adjacency_list.emplace_back();
  }

  for (auto i = 0; i < graph.adjacency_list.size(); i++) {
    for (const auto &neighbor : graph.neighbors(i)) {
        auto community_id_node = partitions.getCommunityForNode(i);
        auto community_id_neighbor = partitions.getCommunityForNode(neighbor);
        if (community_id_node != community_id_neighbor) {
            new_adjacency_list[community_id_node].push_back(community_id_neighbor);
        }
    }
  }

  graph.adjacency_list = std::move(new_adjacency_list);
  graph.num_nodes = partitions.communities.size();
}

int countNonEmptyCommunities(const Partitions &partitions) {
    int count = 0;
    for (const auto &community : partitions.communities) {
        if (!community.empty()) {
            count++;
        }
    }
    return count;
}

Partitions leiden(const mg_graph::GraphView<> &memgraph_graph) {
    Graph graph;
    for (const auto &node : memgraph_graph.Nodes()) {
        graph.num_nodes++;
        for (const auto &neighbor : memgraph_graph.Neighbours(node.id)) {
            graph.addEdge(node.id, neighbor.node_id);
        }
    }
    
    auto partitions = singletonPartition(graph);
    bool done = false;
    while(!done) {
        moveNodesFast(partitions, graph);
        const auto non_empty_communities = countNonEmptyCommunities(partitions);
        done = non_empty_communities == graph.num_nodes;
        if (!done) {
            auto refined_partitions = refinePartition(partitions, graph);
            aggregateGraph(refined_partitions, graph);

            // create new partitions
            Partitions new_partitions;
            for (auto i = 0; i < partitions.communities.size(); i++) {
                for (const auto &node_id : partitions.communities[i]) {
                    if (graph.isVertexInGraph(node_id)) {
                        new_partitions.communities[i].push_back(node_id);
                        new_partitions.community_id[node_id] = i;
                        new_partitions.community_weights[i] = partitions.community_weights[i];
                    }
                }
            }
            partitions = std::move(new_partitions);
        }
    }

    return partitions;
}

std::vector<std::int64_t> GetCommunities(const mg_graph::GraphView<> &graph) {
    auto partitions = leiden(graph);
    std::vector<std::int64_t> communities(graph.Nodes().size());
    for (const auto &node : graph.Nodes()) {
        communities[node.id] = partitions.getCommunityForNode(node.id);
    }
    return communities;
}

}  // namespace leiden_alg
