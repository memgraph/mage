#ifndef LEIDEN_UTILS_HPP
#define LEIDEN_UTILS_HPP

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <omp.h>
#include <utility>
#include <set>
#include <iterator>

#include <boost/unordered/unordered_map.hpp>
#include <boost/unordered/unordered_set.hpp>

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
    std::vector<std::uint64_t> community_weights; // community_id -> weight
    boost::unordered_map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> node_and_community_cache; // (node_id, community_id) -> number

    inline std::uint64_t getCommunityForNode(std::uint64_t node_id) const {
        return community_id[node_id];
    }

    inline void updateWeightForCommunity(std::uint64_t community_id, std::uint64_t weight_update = 1) {
        community_weights[community_id] += weight_update;
    }

    inline void clearCache() {
        node_and_community_cache.clear();
    }
};

struct IntermediaryCommunityId {
    std::uint64_t community_id;
    std::uint64_t level;
    IntermediaryCommunityId *parent;
};

bool edgeBetweenCommunities(const std::vector<std::uint64_t>& community1, const std::vector<std::uint64_t>& community2, const Graph& graph);
bool isSubset(std::vector<std::uint64_t>& set1, std::vector<std::uint64_t>& set2);
std::uint64_t countEdgesBetweenNodeAndCommunity(const Graph& graph, std::uint64_t node_id, std::uint64_t community_id, Partitions& partitions);
std::uint64_t countEdgesBetweenCommunities(std::uint64_t community_id, std::uint64_t subset, Partitions& refined_partitions, Partitions& partitions, const Graph& graph);
std::uint64_t getNumOfPossibleEdges(std::uint64_t n);
std::pair<double, std::uint64_t> computeDeltaCPM(Partitions& partitions, std::uint64_t node_id, std::uint64_t new_community_id, const Graph& graph, double gamma = 1.0);

}  // namespace leiden_alg

#endif // LEIDEN_UTILS_HPP