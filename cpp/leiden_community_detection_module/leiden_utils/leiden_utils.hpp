#ifndef LEIDEN_UTILS_HPP
#define LEIDEN_UTILS_HPP

#include <atomic>
#include <memory>
#include <vector>

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

namespace leiden_alg {

struct Graph {
  std::vector<std::vector<std::pair<std::uint64_t, double>>> adjacency_list; // node_id -> (neighbor_id, edge_weight)

  inline void addEdge(std::uint64_t u, std::uint64_t v, double edge_weight = 1.0) {
    if (u >= adjacency_list.size()) {
        adjacency_list.resize(u + 1);
    }
    adjacency_list[u].emplace_back(v, edge_weight);
  }

  inline bool isVertexInGraph(std::uint64_t u) const {
      return u < adjacency_list.size();
    }

  inline std::size_t size() const {
      return adjacency_list.size();
  }

  inline const std::vector<std::pair<std::uint64_t, double>> &neighbors(std::uint64_t node_id) const {
      return adjacency_list[node_id];
  }
};

struct Partitions {
    std::vector<std::vector<std::uint64_t>> communities; // community_id -> node_ids within the community
    std::vector<std::uint64_t> community_id; // node_id -> community_id

    inline std::uint64_t getCommunityForNode(std::uint64_t node_id) const {
        return community_id[node_id];
    }

    inline std::uint64_t getCommunityWeight(std::uint64_t community_id) const {
        return communities[community_id].size();
    }
};
struct IntermediaryCommunityId {
    std::uint64_t community_id;
    std::uint64_t level;
    std::shared_ptr<IntermediaryCommunityId> parent;
};

using Dendrogram = std::vector<std::vector<std::shared_ptr<IntermediaryCommunityId>>>;

std::vector<double> calculateEdgeWeightsPerCommunity(const Partitions &partitions, const Graph &graph);
void createIntermediaryCommunities(Dendrogram &intermediary_communities, const std::vector<std::vector<std::uint64_t>> &communities, std::uint64_t current_level);

}  // namespace leiden_alg

#endif // LEIDEN_UTILS_HPP