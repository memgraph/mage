#include <filesystem>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <future>

#include "leiden.hpp"
#include "leiden_utils/leiden_utils.hpp"
#include "data_structures/graph_view.hpp"



namespace leiden_alg {

const double MAX_DOUBLE = std::numeric_limits<double>::max();

void moveNodesFast(Partitions &partitions, Graph &graph, double gamma, double resolution_parameter) {
    std::deque<std::uint64_t> nodes;
    std::unordered_set<std::uint64_t> nodes_set;
    std::vector<double> edge_weights_per_community(partitions.communities.size(), 0);
    std::vector<char> visited(partitions.communities.size(), false);
    std::vector<std::uint64_t> neighbor_communities;
    neighbor_communities.reserve(partitions.communities.size());
    nodes_set.reserve(graph.size());
    
    for (std::uint64_t i = 0; i < graph.size(); i++) {
        nodes.push_back(i);
        nodes_set.insert(i);
    }

    std::shuffle(nodes.begin(), nodes.end(), std::mt19937(42));

    while(!nodes.empty()) {
        auto node_id = nodes.front();
        nodes.pop_front();
        nodes_set.erase(node_id);
        auto best_community = partitions.getCommunityForNode(node_id);
        const auto current_community = best_community;

        std::uint64_t number_of_neighbor_communities = 0;

        for (const auto &[neighbor_id, weight] : graph.neighbors(node_id)) {
            const auto community_id_of_neighbor = partitions.getCommunityForNode(neighbor_id);
            if (!visited[community_id_of_neighbor]) {
                visited[community_id_of_neighbor] = 1;
                neighbor_communities[number_of_neighbor_communities++] = community_id_of_neighbor;
            }
            edge_weights_per_community[community_id_of_neighbor] += weight;
        }
        const auto current_delta = static_cast<double>(edge_weights_per_community[best_community]) - (static_cast<double>(partitions.getCommunityWeight(best_community)) -1) * gamma;
        auto best_delta = current_delta;
        for (std::uint64_t i = 0; i < number_of_neighbor_communities; i++) {
            const auto community_id_of_neighbor = partitions.getCommunityForNode(neighbor_communities[i]);

            // look only at the neighbors that are not in the same community
            if (community_id_of_neighbor != best_community) { 
                const auto delta = edge_weights_per_community[community_id_of_neighbor] - static_cast<double>(partitions.getCommunityWeight(community_id_of_neighbor)) * gamma;
                if (delta > best_delta + resolution_parameter) {
                    best_delta = delta;
                    best_community = community_id_of_neighbor;
                }
            }

            edge_weights_per_community[community_id_of_neighbor] = 0.0;
            visited[community_id_of_neighbor] = 0;
        }

        if (current_community != best_community) {
            // remove the node from the current community
            auto &community = partitions.communities[partitions.getCommunityForNode(node_id)];
            auto iterator = std::find(community.begin(), community.end(), node_id);
            
            std::iter_swap(iterator, community.end() - 1);
            community.pop_back();

            partitions.communities[best_community].push_back(node_id);
            partitions.community_id[node_id] = best_community;

            // add neighbors to the queue
            for (const auto &[neighbor_id, _] : graph.neighbors(node_id)) {
                if (nodes_set.find(neighbor_id) == nodes_set.end() && partitions.getCommunityForNode(neighbor_id) != best_community) {
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

void mergeNodesSubset(Partitions &refined_partitions, const Graph &graph, std::uint64_t subset, Partitions &partitions, double gamma, double theta, double resolution_parameter) {
    // external_edge_weight_per_cluster_in_subset[i] - tracks the external edge weight between nodes in cluster i and the other clusters in the same subset
    std::vector<double> external_edge_weight_per_cluster_in_subset(refined_partitions.communities.size(), 0);
    std::vector<std::uint64_t> neighbor_communities;
    std::vector<char> visited(refined_partitions.communities.size(), false);
    std::vector<double> edge_weights(refined_partitions.communities.size(), 0); // edge weight of community
    std::vector<double> probability_of_merging;      

    probability_of_merging.reserve(refined_partitions.communities.size());
    neighbor_communities.reserve(refined_partitions.communities.size());

    // at the beginning all nodes are in singleton communities                                 
    for (const auto &node_id : partitions.communities[subset]) {                                
        for (const auto &[neighbor_id, edge_weight] : graph.neighbors(node_id)) {
            if (partitions.getCommunityForNode(neighbor_id) == subset) {
                external_edge_weight_per_cluster_in_subset[refined_partitions.getCommunityForNode(node_id)] += edge_weight;
            }                                                                       
        }                                                                                                                       
    }                                                                                                                         

    for (const auto &node_id : partitions.communities[subset]) {                                                                                                                                                                                                                                                                                                                                                                                       
        const auto current_community = refined_partitions.getCommunityForNode(node_id);
        std::uint64_t number_of_neighbor_communities = 0;

        if (isInSingletonCommunity(refined_partitions, node_id) && 
                (external_edge_weight_per_cluster_in_subset[current_community] >= gamma * static_cast<double>(refined_partitions.getCommunityWeight(current_community)) * 
                static_cast<double>((partitions.communities[subset].size() - refined_partitions.getCommunityWeight(current_community))))) {
            auto total_cum_sum = 0.0;

            auto max_delta = 0.0;
            auto best_community = current_community;

            // find neighbor communities
            for (const auto &[neighbor_id, weight] : graph.neighbors(node_id)) {
                if (partitions.getCommunityForNode(neighbor_id) == subset) {
                    const auto refined_community = refined_partitions.getCommunityForNode(neighbor_id);
                    edge_weights[refined_community] += weight;
                    if (!visited[refined_community]) {
                        visited[refined_community] = 1;
                        neighbor_communities[number_of_neighbor_communities++] = refined_community;
                    }
                }
            }

            // it's important that this goes from 0 to neighbor_communities.size() because we need to update first neighbor_communities.size() elements of probability_of_merging
            for (std::uint64_t j = 0; j < neighbor_communities.size(); j++) {
                const auto neighbor_community = neighbor_communities[j];
                if (refined_partitions.getCommunityWeight(neighbor_community) == 0) continue; // skip empty communities

                if (external_edge_weight_per_cluster_in_subset[neighbor_community] >= gamma * static_cast<double>(refined_partitions.getCommunityWeight(neighbor_community))
                    * static_cast<double>((partitions.communities[subset].size() - refined_partitions.getCommunityWeight(neighbor_community)))) {
                    
                    const auto delta = edge_weights[neighbor_community] - static_cast<double>(refined_partitions.getCommunityWeight(neighbor_community)) * gamma;

                    if (delta > resolution_parameter) {
                        total_cum_sum += std::exp(delta / theta);
                    }

                    if (delta > max_delta + resolution_parameter) {
                        max_delta = delta;
                        best_community = neighbor_community;
                    }
                }
                probability_of_merging[j] = total_cum_sum;
                edge_weights[neighbor_community] = 0.0;
                visited[neighbor_community] = 0;
            }
            if (total_cum_sum < MAX_DOUBLE) {
                static std::minstd_rand gen(42);
                std::uniform_real_distribution<double> dis(0, total_cum_sum);
                const auto random_number = dis(gen); 
                
                const auto last = probability_of_merging.begin() + static_cast<std::int64_t>(neighbor_communities.size());
                const auto best_community_index = std::lower_bound(probability_of_merging.begin(), last, random_number) - probability_of_merging.begin();
                best_community = neighbor_communities[best_community_index];
            }

            refined_partitions.communities[best_community].push_back(node_id);
            refined_partitions.community_id[node_id] = best_community;

            // remove the node from the current community, note that it is a singleton community
            refined_partitions.communities[current_community].clear();

            // update the external edge weight for the new community and edge weights for the communities
            for (const auto &[neighbor_id, weight] : graph.neighbors(node_id)) {
                const auto neighbor_community = refined_partitions.getCommunityForNode(neighbor_id);
                if (neighbor_community == subset) {
                    // update external edge weight for the new community
                    if (neighbor_community == best_community) {
                        external_edge_weight_per_cluster_in_subset[best_community] -= weight;
                    } else {
                        external_edge_weight_per_cluster_in_subset[best_community] += weight;
                    }
                }
            }
        }
    }
}

Partitions refinePartition(Partitions &partitions, const Graph &graph, double gamma, double theta, double resolution_parameter) {
    auto refined_partitions = singletonPartition(graph);
    for (std::uint64_t i = 0; i < partitions.communities.size(); i++) {
        if (partitions.communities[i].size() > 1) {
            mergeNodesSubset(refined_partitions, graph, i, partitions, gamma, theta, resolution_parameter);
        }
    }
    return refined_partitions;
}

// communities becomes the new nodes
Partitions aggregateGraph(const Partitions &refined_partitions, Graph &graph, Partitions &original_partitions, Dendrogram &intermediary_communities, std::uint64_t current_level) {
    std::vector<std::vector<std::uint64_t>> remapped_communities; // nodes and communities should go from 0 to n
    std::unordered_map<std::uint64_t, std::uint64_t> old_community_to_new_community; // old_community_id -> new_community_id
    std::vector<std::vector<std::pair<std::uint64_t, double>>> new_adjacency_list;
    new_adjacency_list.reserve(refined_partitions.communities.size());
    std::uint64_t new_community_id = 0;
    Partitions new_partitions;
    new_partitions.communities.reserve(refined_partitions.communities.size());
    new_partitions.community_id.reserve(graph.size());
    intermediary_communities.emplace_back();

    for (std::uint64_t i = 0; i < refined_partitions.communities.size(); i++) {
        const auto &community = refined_partitions.communities[i];
        if (!community.empty()) {
            remapped_communities.push_back(community);
            old_community_to_new_community[i] = new_community_id;
            new_community_id++;
        }
    }

    auto future = std::async(std::launch::async, createIntermediaryCommunities, std::ref(intermediary_communities), std::ref(remapped_communities), current_level);
    
    // create new adjacency list -> if there is an edge between two communities, add it to the new adjacency list
    std::unordered_map<std::uint64_t, std::unordered_set<std::uint64_t>> edge_exists;
    new_adjacency_list = std::vector<std::vector<std::pair<std::uint64_t, double>>>(remapped_communities.size(), std::vector<std::pair<std::uint64_t, double>>());
    for (std::uint64_t i = 0; i < graph.adjacency_list.size(); i++) {
        const auto community_id = refined_partitions.getCommunityForNode(i);
        const auto new_community_id = old_community_to_new_community[community_id];
        for (const auto &[neighbor_id, weight] : graph.adjacency_list[i]) {
            const auto neighbor_community_id = refined_partitions.getCommunityForNode(neighbor_id);
            const auto new_neighbor_community_id = old_community_to_new_community[neighbor_community_id];

            if (new_community_id == new_neighbor_community_id) continue;
            auto edge_exists_iter = edge_exists.find(new_community_id);
            if (edge_exists_iter != edge_exists.end()) {
                if (edge_exists_iter->second.find(new_neighbor_community_id) != edge_exists_iter->second.end()) continue;
                edge_exists_iter->second.insert(new_neighbor_community_id);
                edge_exists[new_neighbor_community_id].insert(new_community_id);
                new_adjacency_list[new_community_id].emplace_back(new_neighbor_community_id, weight);
                new_adjacency_list[new_neighbor_community_id].emplace_back(new_community_id, weight);
            } else {
                edge_exists[new_community_id].insert(new_neighbor_community_id);
                edge_exists[new_neighbor_community_id].insert(new_community_id);
                new_adjacency_list[new_community_id].emplace_back(new_neighbor_community_id, weight);
                new_adjacency_list[new_neighbor_community_id].emplace_back(new_community_id, weight);
            }
        }
    }

    graph.adjacency_list = std::move(new_adjacency_list);
    new_community_id = 0;
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

    future.wait();

    return new_partitions;
}

bool checkIfDone(const Partitions &partitions) {
    return std::ranges::all_of(partitions.communities, [](const auto &community) {
        return community.size() == 1;
    });
}

bool onlySingleCommunity(const Partitions &partitions) {
    const auto number_of_empty_communities = std::ranges::count_if(partitions.communities, [](const auto &community) {
        return community.empty();
    });
    return partitions.communities.size() == number_of_empty_communities + 1;
}

Dendrogram leiden(const mg_graph::GraphView<> &memgraph_graph, double gamma, double theta, double resolution_parameter) {
    Graph graph;
    Partitions partitions;
    Dendrogram intermediary_communities; // level -> community_ids
    intermediary_communities.emplace_back();
    std::uint64_t level = 0;
    double sum_of_weights = 0.0;

    for (const auto &[id, from, to] : memgraph_graph.Edges()) {
        const auto edge_weight = memgraph_graph.IsWeighted()
                                           ? memgraph_graph.GetWeight(id)
                                           : 1.0;  // Make it positive and cast to Double, fixed to 1.0
        // always add the edge, because the algorithm works on undirected graphs
        graph.addEdge(from, to, edge_weight);
        graph.addEdge(to, from, edge_weight);
        sum_of_weights += edge_weight;
    }
    gamma /= sum_of_weights;

    // initialize partitions and leafs of the dendrogram
    partitions = singletonPartition(graph);
    intermediary_communities[0].reserve(memgraph_graph.Nodes().size());
    for (std::uint64_t i = 0; i < memgraph_graph.Nodes().size(); i++) {
        intermediary_communities[0].push_back(std::make_shared<IntermediaryCommunityId>(IntermediaryCommunityId{i, 0, nullptr}));
    }

    bool done = false;
    while(!done) {
        moveNodesFast(partitions, graph, gamma, resolution_parameter);
        done = checkIfDone(partitions);
        if (!done) {
            auto refined_partitions = refinePartition(partitions, graph, gamma, theta, resolution_parameter);
            // if (onlySingleCommunity(refined_partitions)) {
            //     done = true;
            //     continue;
            // }
            if (checkIfDone(refined_partitions)) {
                refined_partitions = std::move(partitions);
                partitions = aggregateGraph(refined_partitions, graph, refined_partitions, intermediary_communities, level);
            }
            else {
                partitions = aggregateGraph(refined_partitions, graph, partitions, intermediary_communities, level);
            }
            level++;
        }
        if (partitions.communities.size() == 1) {
            done = true;
        }
    }

    return intermediary_communities;
}

std::vector<std::vector<std::uint64_t>> getCommunities(const mg_graph::GraphView<> &graph, double gamma, double theta, double resolution_parameter) {
    std::vector<std::vector<std::uint64_t>> node_and_community_hierarchy; // node_id -> list of community_ids
    const auto communities_hierarchy = leiden(graph, gamma, theta, resolution_parameter);
    for (const auto &node : communities_hierarchy[0]) {
        std::vector<std::uint64_t> community_ids;
        auto current_community = node;
        while (current_community != nullptr) {
            community_ids.push_back(current_community->community_id);
            current_community = current_community->parent;
        }
        node_and_community_hierarchy.emplace_back(std::move(community_ids));
    }

    return node_and_community_hierarchy;
}

}  // namespace leiden_alg
