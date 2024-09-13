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
#include "boost/unordered/unordered_map.hpp"
#include "boost/unordered/unordered_set.hpp"
#include "data_structures/graph_view.hpp"
#include "omp.h"

namespace leiden_alg {

const double gamma = 0.25; // TODO: user should be able to set this
const double theta = 0.01; // TODO: user should be able to set this

struct Graph {
  std::vector<std::vector<int>> adjacency_list; // node_id -> neighbors
  // Add an edge to the graph
  void addEdge(int u, int v) {
    if (u >= adjacency_list.size()) {
        adjacency_list.resize(u + 1);
    }
    adjacency_list[u].push_back(v);
  }

  bool isVertexInGraph(int u) const {
      return u < adjacency_list.size();
    }

  std::size_t size() const {
      return adjacency_list.size();
  }

  const std::vector<int> &neighbors(int u) const {
    return adjacency_list[u];
  }
};

struct Partitions {
    std::vector<std::vector<int>> communities; // community_id -> node_ids within the community
    std::vector<int> community_id; // node_id -> community_id
    std::vector<int> community_weights; // community_id -> weight
    boost::unordered_map<std::pair<int, int>, int> node_and_community_cache; // (node_id, community_id) -> number

    int getCommunityForNode(int node_id) const {
        return community_id[node_id];
    }

    void updateWeightForCommunity(int community_id, int weight_update = 1) {
        community_weights[community_id] += weight_update;
    }

    void clearCache() {
        node_and_community_cache.clear();
    }
};

bool edgeBetweenCommunities(const std::vector<int> &community1, const std::vector<int> &community2, const Graph &graph) {
    for (const auto &node1 : community1) {
        for (const auto &node2 : community2) {
            if (std::find(graph.neighbors(node1).begin(), graph.neighbors(node1).end(), node2) != graph.neighbors(node1).end()) {
                return true;
            }
        }
    }
    return false;
}

bool isSubset(const std::vector<int> &set1, const std::vector<int> &set2) {
    return std::includes(set2.begin(), set2.end(), set1.begin(), set1.end());
}

// |E(node, C \ node)|
int countEdgesBetweenNodeAndCommunity(const Graph &graph, int node_id, int community_id, Partitions &partitions) {
    // check cache first
    if (partitions.node_and_community_cache.find({node_id, community_id}) != partitions.node_and_community_cache.end()) {
        return partitions.node_and_community_cache[{node_id, community_id}];
    }
    int count = 0;
    #pragma omp parallel for reduction(+:count) 
    for (int neighbor : graph.neighbors(node_id)) {
         if (partitions.getCommunityForNode(neighbor) == community_id) {
            count++;
        }
    }
    partitions.node_and_community_cache[{node_id, community_id}] = count;
    return count;
}

// |E(C, S \ C)|
int countEdgesBetweenCommunities(int community_id, int subset, Partitions &refined_partitions, Partitions &partitions, const Graph &graph) {
    std::vector<int> set_intersection;
    const auto &refined_community = refined_partitions.communities[community_id];
    const auto &original_community = partitions.communities[subset];
    std::set_difference(original_community.begin(), original_community.end(), refined_community.begin(), refined_community.end(), std::inserter(set_intersection, set_intersection.begin()));
    int count = 0;
    for (const auto &node : set_intersection) {
        count += countEdgesBetweenNodeAndCommunity(graph, node, community_id, refined_partitions);
    }
    return count;
}

int getNumOfPossibleEdges(int n) {
    return n * (n - 1) / 2;
}

std::pair<double, int> computeDeltaCPM(Partitions &partitions, const int node_id, const int new_community_id, const Graph &graph) {
    double result = 0.0;
    
    const auto current_community_id = partitions.getCommunityForNode(node_id);
    const auto current_community_size = static_cast<int>(partitions.communities[current_community_id].size());
    const auto new_community_size = static_cast<int>(partitions.communities[new_community_id].size());
    const auto current_community_weight = partitions.community_weights[current_community_id];
    const auto new_community_weight = partitions.community_weights[new_community_id];

    const auto source_weight = current_community_weight - gamma * getNumOfPossibleEdges(current_community_size);
    const auto target_weight = new_community_weight - gamma * getNumOfPossibleEdges(new_community_size);
    
    const auto num_edges_between_node_and_current_community = countEdgesBetweenNodeAndCommunity(graph, node_id, current_community_id, partitions);
    const auto num_edges_between_node_and_new_community = countEdgesBetweenNodeAndCommunity(graph, node_id, new_community_id, partitions);

    const auto new_source_weight = current_community_weight - num_edges_between_node_and_current_community - gamma * getNumOfPossibleEdges(current_community_size + 1);
    const auto new_target_weight = new_community_weight + num_edges_between_node_and_new_community - gamma * getNumOfPossibleEdges(new_community_size - 1);

    result = new_source_weight + new_target_weight - source_weight - target_weight;
    return {result, num_edges_between_node_and_new_community};
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
            // remove the node from the current community
            auto iterator = std::find(partitions.communities[partitions.getCommunityForNode(node_id)].begin(), partitions.communities[partitions.getCommunityForNode(node_id)].end(), node_id);
            partitions.communities[partitions.getCommunityForNode(node_id)].erase(iterator);

            partitions.clearCache();
            partitions.communities[best_community].push_back(node_id);
            partitions.community_id[node_id] = best_community;
            partitions.updateWeightForCommunity(best_community, weight_update);

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

bool isInSingletonCommunity(const Partitions &partitions, int node_id) {
    auto community_id = partitions.getCommunityForNode(node_id);
    return partitions.communities[community_id].size() == 1;
}

// this gets called only if the community is in singleton community -> so we can assume that the community has only one node
bool isWellConnectedCommunity(int community_id, int subset, Partitions &refined_partitions, Partitions &partitions, const Graph &graph) {
    const auto &community = refined_partitions.communities[community_id];
    const auto &original_community = partitions.communities[subset];
    if (isSubset(community, original_community) && !community.empty()) {
        // check if the community is well connected to the subset
        auto number_of_edges_between_community_and_subset = countEdgesBetweenCommunities(community_id, subset, refined_partitions, partitions, graph);
        if (number_of_edges_between_community_and_subset >= gamma * static_cast<double>(community.size()) * static_cast<double>(partitions.communities[subset].size() - community.size())) {
            return true;
        }
    }
    return false;
}

Partitions mergeNodesSubset(Partitions &refined_partitions, const Graph &graph, int subset, Partitions &partitions) {
    int weight_update = 0;

    // 1 - find well connected nodes within the subset
    std::vector<int> well_connected_nodes;
    auto nodes_in_subset = partitions.communities[subset];
    auto number_of_nodes_in_subset = static_cast<int>(nodes_in_subset.size());
    for (const auto &node_id : nodes_in_subset) {
        const auto num_edges_between_node_and_community = countEdgesBetweenNodeAndCommunity(graph, node_id, subset, partitions);
        const auto node_degree = static_cast<int>(graph.neighbors(node_id).size());
        if (num_edges_between_node_and_community >= gamma * node_degree * (number_of_nodes_in_subset - node_degree)) {
            well_connected_nodes.push_back(node_id);
        }
    }
    
    // 2 - find well connected communities to the subset and calculate their probability of merging
    for (const auto &node_id : well_connected_nodes) {
        std::vector<double> probability_of_merging(partitions.communities.size(), 0);
        if (isInSingletonCommunity(refined_partitions, node_id)) {
            bool probability_higher_than_zero = false; 
            for (auto i = 0; i < refined_partitions.communities.size(); i++) {
                if (isWellConnectedCommunity(i, subset, refined_partitions, partitions, graph)) {
                    const auto result = computeDeltaCPM(refined_partitions, node_id, i, graph);
                    const auto delta_cpm = result.first;
                    if (delta_cpm > 0) {
                        probability_higher_than_zero = true;
                        auto probability = std::exp(1 / theta * delta_cpm);
                        probability_of_merging[i] = probability;
                        weight_update = result.second;
                    }
                }
            }
            if (!probability_higher_than_zero) continue;

            // remove it from the current community -> we know it is a singleton community
            refined_partitions.communities[node_id].clear();
            refined_partitions.clearCache();

            // sample from the probabilities and merge the node with the community
            std::random_device rd;
            std::mt19937 gen(rd());
            std::discrete_distribution<> d(probability_of_merging.begin(), probability_of_merging.end());

            const auto community_id = d(gen);
            refined_partitions.communities[community_id].push_back(node_id);
            refined_partitions.community_id[node_id] = community_id;
            if (weight_update > 0) {
                refined_partitions.updateWeightForCommunity(community_id, weight_update);
            }    
        }
    }

    return refined_partitions;
}

Partitions refinePartition(Partitions &partitions, const Graph &graph) {
    auto refined_partitions = singletonPartition(graph);
    for (auto i = 0; i < partitions.communities.size(); i++) {
        if (partitions.communities[i].size() > 1) {
            refined_partitions = mergeNodesSubset(refined_partitions, graph, i, partitions);
        }
    }
    return refined_partitions;
}

// communities becomes the new nodes
Partitions aggregateGraph (const Partitions &refined_partitions, Graph &graph, const Partitions &original_partitions) {
    std::vector<std::vector<int>> remapped_communities;
    std::unordered_map<int, int> old_community_to_new_community; // old_community_id -> new_community_id
    std::vector<std::vector<int>> new_adjacency_list;
    int new_community_id = 0;
    Partitions new_partitions;

    for (auto i = 0; i < refined_partitions.communities.size(); i++) {
        const auto &community = refined_partitions.communities[i];
        if (!community.empty()) {
            remapped_communities.push_back(community);
            old_community_to_new_community[i] = new_community_id;
            new_community_id++;
        }
    }
    
    // go through communities of partitions and create new edges where there is an edge between two communities 
    for (auto i = 0; i < remapped_communities.size(); i++) {
        new_adjacency_list.emplace_back();
        new_partitions.community_id.push_back(-1);
        new_partitions.community_weights.push_back(0);
        for (auto j = 0; j < remapped_communities.size(); j++) {
            if (i != j && edgeBetweenCommunities(remapped_communities[i], remapped_communities[j], graph)) {
                new_adjacency_list[i].push_back(j);
            }
        }
    }
    graph.adjacency_list = std::move(new_adjacency_list);
    new_community_id = 0;
    new_partitions.communities.reserve(original_partitions.communities.size());
    std::unordered_set<int> already_added_communities;
    for (const auto & community : original_partitions.communities) {
        if (!community.empty()) {
            for (auto j = 0; j < remapped_communities.size(); j++) {
                if (already_added_communities.find(j) == already_added_communities.end() && isSubset(remapped_communities[j], community)) {
                    if (new_partitions.communities.size() <= new_community_id) {
                        new_partitions.communities.emplace_back();
                    }
                    new_partitions.communities[new_community_id].push_back(j);
                    new_partitions.community_id[j] = new_community_id;
                    already_added_communities.insert(j);
                }
            }
            new_community_id++;
        }
    }

    return new_partitions;
}

bool checkIfDone(const Partitions &partitions, const Graph &graph) {
    std::size_t count_single_node_communities = 0;
    for (const auto &community : partitions.communities) {
        if (community.size() == 1) {
            count_single_node_communities++;
        }
    }
    return count_single_node_communities == graph.size();
}

void recalculateWeights(Partitions &partitions, const Graph &graph) {
    // calculate number of edges in the community
    boost::unordered_set<std::pair<int, int>> edges;
    for (auto i = 0; i < partitions.communities.size(); i++) {
        partitions.community_weights[i] = 0;
        for (const auto &node : partitions.communities[i]) {
            for (const auto &neighbor : graph.neighbors(node)) {
                if (partitions.getCommunityForNode(neighbor) == i && edges.find({node, neighbor}) == edges.end()) {
                    partitions.community_weights[i]++;
                    edges.insert({node, neighbor});
                }
            }
        }
    }
}


Partitions leiden(const mg_graph::GraphView<> &memgraph_graph) {
    Graph graph;
    for (const auto &node : memgraph_graph.Nodes()) {
        int previous_neighbour = -1;
        for (const auto &neighbor : memgraph_graph.Neighbours(node.id)) {
            if (neighbor.node_id != previous_neighbour) {
                graph.addEdge(node.id, neighbor.node_id);
                previous_neighbour = neighbor.node_id;
            }
        }
    }
    auto partitions = singletonPartition(graph);
    bool done = false;
    while(!done) {
        moveNodesFast(partitions, graph);
        done = checkIfDone(partitions, graph) || partitions.communities.size() <= 2;
        if (!done) {
            partitions.clearCache();
            auto refined_partitions = refinePartition(partitions, graph);
            partitions = aggregateGraph(refined_partitions, graph, partitions);
            recalculateWeights(partitions, graph); // TODO: fix this
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
