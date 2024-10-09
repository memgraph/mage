#ifndef LEIDEN_UTILS_HPP
#define LEIDEN_UTILS_HPP

#include <memory>
#include <vector>

#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>

namespace leiden_alg {

struct Graph {
  std::vector<std::vector<std::uint64_t>> adjacency_list; // node_id -> neighbors

  // Add an edge to the graph
  inline void addEdge(std::uint64_t u, std::uint64_t v) {
    if (u >= adjacency_list.size()) {
        adjacency_list.resize(u + 1);
    }
    adjacency_list[u].push_back(v);
  }

  inline bool isVertexInGraph(std::uint64_t u) const {
      return u < adjacency_list.size();
    }

  inline std::size_t size() const {
      return adjacency_list.size();
  }

  inline const std::vector<std::uint64_t> &neighbors(std::uint64_t u) const {
    return adjacency_list[u];
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

std::vector<std::uint64_t> calculateEdgeWeightsPerCommunity(const Partitions &partitions, const Graph &graph);
void createIntermediaryCommunities(std::vector<std::vector<std::shared_ptr<IntermediaryCommunityId>>> &intermediary_communities, const std::vector<std::vector<std::uint64_t>> &communities, std::uint64_t current_level);

}  // namespace leiden_alg

#endif // LEIDEN_UTILS_HPP