#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

#include "leiden.hpp"
#include "leiden_utils/leiden_utils.hpp"
#include "data_structures/graph_view.hpp"

namespace leiden_alg {

void moveNodesFast(Partitions &partitions, Graph &graph, const double gamma, std::vector<std::uint64_t> &edge_weights_per_community) {
    std::deque<std::uint64_t> nodes;
    std::unordered_set<std::uint64_t> nodes_set;
    
    for (std::uint64_t i = 0; i < graph.size(); i++) {
        nodes.push_back(i);
        nodes_set.insert(i);
    }

    std::shuffle(nodes.begin(), nodes.end(), std::mt19937(std::random_device()()));

    while(!nodes.empty()) {
        auto node_id = nodes.front();
        nodes.pop_front();
        nodes_set.erase(node_id);
        auto best_community = partitions.getCommunityForNode(node_id);
        const auto current_community = best_community;
        auto best_delta = static_cast<double>(edge_weights_per_community[best_community]) - static_cast<double>(partitions.getCommunityWeight(best_community)) * gamma;

        for (const auto neighbor_id : graph.neighbors(node_id)) {
            const auto community_id_of_neighbor = partitions.getCommunityForNode(neighbor_id);

            // look only at the neighbors that are not in the same community
            if (community_id_of_neighbor != best_community) { 
                const auto delta = static_cast<double>(edge_weights_per_community[community_id_of_neighbor]) - static_cast<double>(partitions.getCommunityWeight(community_id_of_neighbor)) * gamma;
                if (delta > best_delta) {
                    best_delta = delta;
                    best_community = community_id_of_neighbor;
                }
            }
        }

        if (current_community != best_community) {
            // remove the node from the current community
            auto &community = partitions.communities[partitions.getCommunityForNode(node_id)];
            auto iterator = std::find(community.begin(), community.end(), node_id);
            if (iterator == community.end()) {
                throw std::runtime_error("Node not found in the community");
            }
            partitions.communities[partitions.getCommunityForNode(node_id)].erase(iterator);

            partitions.communities[best_community].push_back(node_id);
            partitions.community_id[node_id] = best_community;

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
  for (std::uint64_t i = 0; i < graph.size(); i++) {
    partitions.communities.push_back({i});
    partitions.community_id.push_back(i);
  }
  return partitions;
}

bool isInSingletonCommunity(const Partitions &partitions, std::uint64_t node_id) {
    auto community_id = partitions.getCommunityForNode(node_id);
    return partitions.communities[community_id].size() == 1;
}

void mergeNodesSubset(Partitions &refined_partitions, const Graph &graph, std::uint64_t subset, Partitions &partitions, std::mt19937 &gen, std::discrete_distribution<> &distribution, const double gamma, const double theta) {
    // external_edge_weight_per_cluster_in_subset[i] - tracks the external edge weight between nodes in cluster i and the other clusters in the same subset
    std::vector<std::uint64_t> external_edge_weight_per_cluster_in_subset(partitions.communities[subset].size(), 0);
    std::vector<std::uint64_t> edge_weights(refined_partitions.communities.size(), 0); // number of edges in the community
    std::uint64_t total_node_weight = 0;

    std::vector<std::uint64_t> neighbor_communities;
    for (const auto &node_id : partitions.communities[subset]) {
        total_node_weight++;
        for (const auto &neighbor_id : graph.neighbors(node_id)) {
            if (partitions.getCommunityForNode(neighbor_id) == subset) {
                external_edge_weight_per_cluster_in_subset[node_id]++;
            }
        }
    }

    for (const auto &node_id : partitions.communities[subset]) {
        std::vector<double> probability_of_merging(refined_partitions.communities.size(), 0);
        auto current_community = refined_partitions.getCommunityForNode(node_id);
        auto best_community = current_community;

        if (isInSingletonCommunity(refined_partitions, node_id) && (
                static_cast<double>(external_edge_weight_per_cluster_in_subset[current_community]) >= 
                gamma * static_cast<double>(refined_partitions.getCommunityWeight(current_community)) * static_cast<double>((total_node_weight - refined_partitions.getCommunityWeight(current_community))))) {
            
            neighbor_communities.clear();
            neighbor_communities.reserve(refined_partitions.communities.size() - 1);
            std::vector<bool> visited(refined_partitions.communities.size(), false);

            // find neighbor communities
            for (const auto &neighbor : graph.neighbors(node_id)) {
                if (partitions.getCommunityForNode(neighbor) == subset) {
                    const auto refined_community = refined_partitions.getCommunityForNode(neighbor);
                    edge_weights[refined_community]++;
                    if (!visited[refined_community]) {
                        visited[refined_community] = true;
                        neighbor_communities.push_back(refined_community);
                    }
                }
            }

            auto max_delta = 0.0;

            for (const auto &neighbor_community : neighbor_communities) {
                if (static_cast<double>(external_edge_weight_per_cluster_in_subset[neighbor_community]) >= gamma * static_cast<double>(refined_partitions.getCommunityWeight(neighbor_community))
                    * static_cast<double>((total_node_weight - refined_partitions.getCommunityWeight(neighbor_community)))) {
                    
                    const auto delta = static_cast<double>(edge_weights[neighbor_community]) - static_cast<double>(refined_partitions.getCommunityWeight(neighbor_community)) * gamma;
                    if (delta > max_delta) {
                        max_delta = delta;
                        best_community = neighbor_community;
                    }

                    if (delta > 0) {
                        probability_of_merging[neighbor_community] = std::exp(delta / theta); // check this formula
                    }
                }
                edge_weights[neighbor_community] = 0;
            }

            if (max_delta > 0) {
                distribution.param(std::discrete_distribution<>::param_type(probability_of_merging.begin(), probability_of_merging.end()));
                best_community = distribution(gen);
                refined_partitions.communities[best_community].push_back(node_id);
                refined_partitions.community_id[node_id] = best_community;

                // remove the node from the current community, note that it is a singleton community
                refined_partitions.communities[current_community].clear();

                // update the external edge weight for the new community
                for (const auto &neighbor : graph.neighbors(node_id)) {
                    if (partitions.getCommunityForNode(neighbor) == subset) {
                        if (refined_partitions.getCommunityForNode(neighbor) == best_community) {
                            external_edge_weight_per_cluster_in_subset[neighbor]++;
                        } else {
                            external_edge_weight_per_cluster_in_subset[neighbor]--;
                        }
                    }
                }
            }
        }
    }
}

Partitions refinePartition(Partitions &partitions, const Graph &graph, std::mt19937 &gen, std::discrete_distribution<> &distribution, const double gamma, const double theta) {
    auto refined_partitions = singletonPartition(graph);
    for (std::uint64_t i = 0; i < partitions.communities.size(); i++) {
        if (partitions.communities[i].size() > 1) {
            mergeNodesSubset(refined_partitions, graph, i, partitions, gen, distribution, gamma, theta); 
        }
    }
    return refined_partitions;
}

// communities becomes the new nodes
Partitions aggregateGraph(const Partitions &refined_partitions, Graph &graph, Partitions &original_partitions, std::vector<std::vector<IntermediaryCommunityId>> &intermediary_communities, std::uint64_t current_level) {
    std::vector<std::vector<std::uint64_t>> remapped_communities;
    std::unordered_map<std::uint64_t, std::uint64_t> old_community_to_new_community; // old_community_id -> new_community_id
    std::vector<std::vector<std::uint64_t>> new_adjacency_list;
    std::uint64_t new_community_id = 0;
    Partitions new_partitions;
    
    intermediary_communities.emplace_back();

    for (std::uint64_t i = 0; i < refined_partitions.communities.size(); i++) {
        const auto &community = refined_partitions.communities[i];
        if (!community.empty()) {
            remapped_communities.push_back(community);
            old_community_to_new_community[i] = new_community_id;
            new_community_id++;
        }
    }
    
    // 1. step - go through communities of partitions and create new edges where there is an edge between two communities 
    // 2. step - create new intermediary community ids -> nodes that are in community i are children of the new intermediary community id
    for (std::uint64_t i = 0; i < remapped_communities.size(); i++) {
        // 1. step
        new_adjacency_list.emplace_back();
        new_partitions.community_id.push_back(-1);
        for (std::uint64_t j = 0; j < remapped_communities.size(); j++) {
            if (i != j && edgeBetweenCommunities(remapped_communities[i], remapped_communities[j], graph)) {
                new_adjacency_list[i].push_back(j);
            }
        }

        // 2. step
        auto *new_intermediary_community_id = new IntermediaryCommunityId({i, current_level + 1, nullptr});
        for (const auto &node_id : remapped_communities[i]) {
            intermediary_communities[current_level][node_id].parent = new_intermediary_community_id;    
        }
        intermediary_communities[current_level + 1].push_back(*new_intermediary_community_id);
    }

    graph.adjacency_list = std::move(new_adjacency_list);
    new_community_id = 0;
    new_partitions.communities.reserve(original_partitions.communities.size());
    new_partitions.community_id.reserve(remapped_communities.size());
    new_partitions.community_id = std::vector<std::uint64_t>(remapped_communities.size(), 0);

    std::unordered_set<std::uint64_t> already_added_communities;
    for (auto &community : original_partitions.communities) {
        if (!community.empty()) {
            for (std::uint64_t j = 0; j < remapped_communities.size(); j++) {
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


std::vector<std::vector<IntermediaryCommunityId>> leiden(const mg_graph::GraphView<> &memgraph_graph) {
    Graph graph;
    Partitions partitions;
    std::vector<std::vector<IntermediaryCommunityId>> intermediary_communities; // level -> community_ids
    intermediary_communities.emplace_back();
    std::uint64_t level = 0;

    // random device -> used when sampling from the merging probabilities
    std::random_device random_device;
    std::mt19937 gen(random_device());
    std::discrete_distribution<> distribution;

    std::uint64_t number_of_edges = 0;

    for (const auto &node : memgraph_graph.Nodes()) {
        partitions.communities.push_back({node.id});
        partitions.community_id.push_back(node.id);
        intermediary_communities[level].emplace_back(IntermediaryCommunityId{node.id, level, nullptr});

        std::uint64_t previous_neighbour = node.id; // to avoid adding the same edge twice 
        for (const auto &neighbor : memgraph_graph.Neighbours(node.id)) {
            if (neighbor.node_id != previous_neighbour) {
                graph.addEdge(node.id, neighbor.node_id);
                previous_neighbour = neighbor.node_id;
                number_of_edges++;
            }
        }
    }
    const double gamma = 1.0 / static_cast<double>(number_of_edges); // TODO: user should be able to set this
    const double theta = 0.01; // TODO: user should be able to set this
    bool done = false;
    while(!done) {
        auto edge_weights_per_community = calculateEdgeWeightsPerCommunity(partitions, graph);
        moveNodesFast(partitions, graph, gamma, edge_weights_per_community);
        done = checkIfDone(partitions, graph) || partitions.communities.size() <= 2;
        if (!done) {
            auto refined_partitions = refinePartition(partitions, graph, gen, distribution, gamma, theta);
            partitions = aggregateGraph(refined_partitions, graph, partitions, intermediary_communities, level);
            level++;
        }
    }

    return intermediary_communities;
}

std::vector<std::vector<std::uint64_t>> getCommunities(const mg_graph::GraphView<> &graph) {
    std::vector<std::vector<std::uint64_t>> node_and_community_hierarchy; // node_id -> list of community_ids
    auto communities_hierarchy = leiden(graph);
    for (const auto &node : communities_hierarchy[0]) {
        std::vector<std::uint64_t> community_ids;
        const auto *current_community = &node;
        while (current_community != nullptr) {
            community_ids.push_back(current_community->community_id);
            current_community = current_community->parent;
        }
        node_and_community_hierarchy.emplace_back(std::move(community_ids));
    }

    return node_and_community_hierarchy;
}

}  // namespace leiden_alg
