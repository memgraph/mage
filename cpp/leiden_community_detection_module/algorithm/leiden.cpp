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

const double MAX_DOUBLE = std::numeric_limits<double>::max();

void moveNodesFast(Partitions &partitions, Graph &graph, const double gamma, std::vector<std::uint64_t> &edge_weights_per_community) {
    std::deque<std::uint64_t> nodes;
    std::unordered_set<std::uint64_t> nodes_set;
    nodes_set.reserve(graph.size());
    
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
            std::iter_swap(iterator, community.end() - 1);
            community.pop_back();

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

void mergeNodesSubset(Partitions &refined_partitions, const Graph &graph, std::uint64_t subset, Partitions &partitions, const double gamma, const double theta, std::unordered_set<std::uint64_t> &refined_communities_within_subset) {
    // external_edge_weight_per_cluster_in_subset[i] - tracks the external edge weight between nodes in cluster i and the other clusters in the same subset
    std::vector<std::uint64_t> external_edge_weight_per_cluster_in_subset(refined_partitions.communities.size(), 0);
    std::vector<std::uint64_t> edge_weights(refined_partitions.communities.size(), 0); // number of edges in the community

    std::vector<std::uint64_t> neighbor_communities;
    for (const auto &node_id : partitions.communities[subset]) {
        refined_communities_within_subset.insert(node_id); // at the beginning, all nodes are in its own community
        for (const auto &neighbor_id : graph.neighbors(node_id)) {
            if (partitions.getCommunityForNode(neighbor_id) == subset) {
                external_edge_weight_per_cluster_in_subset[node_id]++;
            }
        }
    }

    for (const auto &node_id : partitions.communities[subset]) {
        const auto current_community = refined_partitions.getCommunityForNode(node_id);

        if (isInSingletonCommunity(refined_partitions, node_id) && (
                static_cast<double>(external_edge_weight_per_cluster_in_subset[current_community]) >= 
                gamma * static_cast<double>(refined_partitions.getCommunityWeight(current_community)) * static_cast<double>((partitions.communities[subset].size() - refined_partitions.getCommunityWeight(current_community))))) {
            
            std::vector<double> probability_of_merging;
            probability_of_merging.reserve(neighbor_communities.size());
            auto total_cum_sum = 0.0;

            neighbor_communities.clear();
            neighbor_communities.reserve(refined_partitions.communities.size() - 1);
            std::vector<bool> visited(refined_partitions.communities.size(), false);
            double max_delta = 0.0;
            auto best_community = current_community;

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

            for (const auto neighbor_community : neighbor_communities) {
                 if (static_cast<double>(external_edge_weight_per_cluster_in_subset[neighbor_community]) >= gamma * static_cast<double>(refined_partitions.getCommunityWeight(neighbor_community))
                    * static_cast<double>((partitions.communities[subset].size() - refined_partitions.getCommunityWeight(neighbor_community)))) {
                    
                    const auto delta = static_cast<double>(edge_weights[neighbor_community]) - static_cast<double>(refined_partitions.getCommunityWeight(neighbor_community)) * gamma;

                    if (delta > 0) {
                        total_cum_sum += std::exp(delta / theta);
                    }

                    if (delta > max_delta) {
                        max_delta = delta;
                        best_community = neighbor_community;
                    }
                }
                probability_of_merging.push_back(total_cum_sum);
            }
            if (total_cum_sum < MAX_DOUBLE) {
                static std::minstd_rand gen(std::random_device{}());
                std::uniform_real_distribution<double> dis(0, total_cum_sum);
                const auto random_number = dis(gen); 
                
                // use lower_bound to find the first element that is greater than random_number
                const auto best_community_index = std::lower_bound(probability_of_merging.begin(), probability_of_merging.end(), random_number) - probability_of_merging.begin();
                best_community = neighbor_communities[best_community_index];
                refined_partitions.communities[best_community].push_back(node_id);
            }

            // refined_partitions.communities[best_community].push_back(node_id);
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

Partitions refinePartition(Partitions &partitions, const Graph &graph, const double gamma, const double theta, std::vector<std::unordered_set<std::uint64_t>> &subset_to_refined_communities) {
    auto refined_partitions = singletonPartition(graph);
    for (std::uint64_t i = 0; i < partitions.communities.size(); i++) {
        if (partitions.communities[i].size() > 1) {
            mergeNodesSubset(refined_partitions, graph, i, partitions, gamma, theta, subset_to_refined_communities[i]);
        }
        if (partitions.communities[i].size() == 1) {
            subset_to_refined_communities[i].insert(i);
        }
    }
    return refined_partitions;
}

// communities becomes the new nodes
Partitions aggregateGraph(const Partitions &refined_partitions, Graph &graph, Partitions &original_partitions, std::vector<std::vector<IntermediaryCommunityId>> &intermediary_communities, std::uint64_t current_level, const std::vector<std::unordered_set<std::uint64_t>> &subset_to_refined_communities) {
    std::vector<std::vector<std::uint64_t>> remapped_communities; // nodes and communities should go from 0 to n
    std::unordered_map<std::uint64_t, std::uint64_t> old_community_to_new_community; // old_community_id -> new_community_id
    std::unordered_map<std::uint64_t, std::uint64_t> new_community_to_old_community; // new_community_id -> old_community_id
    std::vector<std::vector<std::uint64_t>> new_adjacency_list;
    std::uint64_t new_community_id = 0;
    Partitions new_partitions;
    
    intermediary_communities.emplace_back();

    for (std::uint64_t i = 0; i < refined_partitions.communities.size(); i++) {
        const auto &community = refined_partitions.communities[i];
        if (!community.empty()) {
            remapped_communities.push_back(community);
            old_community_to_new_community[i] = new_community_id;
            new_community_to_old_community[new_community_id] = i;
            new_community_id++;
        }
    }
    
    // create new adjacency list -> if there is an edge between two communities, add it to the new adjacency list
    std::vector<std::vector<bool>> edge_exists(refined_partitions.communities.size(), std::vector<bool>(refined_partitions.communities.size(), false));
    new_adjacency_list = std::vector<std::vector<std::uint64_t>>(remapped_communities.size(), std::vector<std::uint64_t>());
    for (std::uint64_t i = 0; i < graph.adjacency_list.size(); i++) {
        const auto community_id = refined_partitions.getCommunityForNode(i);
        const auto new_community_id = old_community_to_new_community[community_id];
        for (const auto &neighbor : graph.adjacency_list[i]) {
            const auto neighbor_community_id = refined_partitions.getCommunityForNode(neighbor);
            const auto new_neighbor_community_id = old_community_to_new_community[neighbor_community_id];
            if (edge_exists[new_community_id][new_neighbor_community_id] || new_community_id == new_neighbor_community_id) continue;

            new_adjacency_list[new_community_id].push_back(new_neighbor_community_id);
            new_adjacency_list[new_neighbor_community_id].push_back(new_community_id);
            edge_exists[new_community_id][new_neighbor_community_id] = true;
            edge_exists[new_neighbor_community_id][new_community_id] = true;
        }
    }

    // create new intermediary community ids -> nodes that are in community i are children of the new intermediary community id
    for (std::uint64_t i = 0; i < remapped_communities.size(); i++) {
        auto *new_intermediary_community_id = new IntermediaryCommunityId({i, current_level + 1, nullptr});
        for (const auto &node_id : remapped_communities[i]) {
            intermediary_communities[current_level][node_id].parent = new_intermediary_community_id;    
        }
        intermediary_communities[current_level + 1].push_back(*new_intermediary_community_id);
    }

    graph.adjacency_list = std::move(new_adjacency_list);
    new_community_id = 0;
    new_partitions.communities.reserve(graph.size());
    new_partitions.community_id.reserve(graph.size());
    new_partitions.community_id = std::vector<std::uint64_t>(graph.size(), -1);

    // create new partitions -> communities that are a subset of the original community are added to the new partitions as a part of the same community
    for (const auto &community_list : original_partitions.communities) {
        bool new_community_created = false;
        for (const auto &community_id : community_list) {
            // if community was added to remapped communities that means it wasn't empty
            const auto remapped_community_id = old_community_to_new_community.find(community_id);
            if (remapped_community_id != old_community_to_new_community.end()) {
                if (!new_community_created) {
                    new_partitions.communities.emplace_back();
                    new_community_created = true;
                }
                new_partitions.communities[new_community_id].push_back(remapped_community_id->second);
                new_partitions.community_id[remapped_community_id->second] = new_community_id;
            }
        }
        if (new_community_created) {
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

void sanityCheck (const Partitions &partitions, std::uint64_t level) {
    for (std::uint64_t i = 0; i < partitions.communities.size(); i++) {
        for (const auto &node_id : partitions.communities[i]) {
            if (partitions.getCommunityForNode(node_id) != i) {
                throw std::runtime_error("Community id does not match at level " + std::to_string(level) + " for node " + std::to_string(node_id) + " expected " + std::to_string(i) + " but got " + std::to_string(partitions.getCommunityForNode(node_id)));
            }
        }
    }
    for (const auto &community_id : partitions.community_id) {
        if (community_id >= partitions.communities.size()) {
            throw std::runtime_error("Community id is out of bounds at level " + std::to_string(level) + " for node " + std::to_string(community_id));
        }
        if (community_id == -1) {
            throw std::runtime_error("Community id is -1 at level " + std::to_string(level));
        }
    }
}

std::vector<std::vector<IntermediaryCommunityId>> leiden(const mg_graph::GraphView<> &memgraph_graph) {
    Graph graph;
    Partitions partitions;
    std::vector<std::vector<IntermediaryCommunityId>> intermediary_communities; // level -> community_ids
    intermediary_communities.emplace_back();
    std::uint64_t level = 0;

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
            std::vector<std::unordered_set<std::uint64_t>> subset_to_refined_communities(partitions.communities.size(), std::unordered_set<std::uint64_t>()); // subset_id -> refined_community_ids
            const auto refined_partitions = refinePartition(partitions, graph, gamma, theta, subset_to_refined_communities);
            partitions = aggregateGraph(refined_partitions, graph, partitions, intermediary_communities, level, subset_to_refined_communities);
            sanityCheck(partitions, level);
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
