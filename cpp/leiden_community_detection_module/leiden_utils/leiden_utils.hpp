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
  std::vector<std::vector<int>> adjacency_list; // node_id -> neighbors

  // Add an edge to the graph
  inline void addEdge(int u, int v) {
    if (u >= adjacency_list.size()) {
        adjacency_list.resize(u + 1);
    }
    adjacency_list[u].push_back(v);
  }

  inline bool isVertexInGraph(int u) const {
      return u < adjacency_list.size();
    }

  inline std::size_t size() const {
      return adjacency_list.size();
  }

  inline const std::vector<int> &neighbors(int u) const {
    return adjacency_list[u];
  }
};

struct Partitions {
    std::vector<std::vector<int>> communities; // community_id -> node_ids within the community
    std::vector<int> community_id; // node_id -> community_id
    std::vector<int> community_weights; // community_id -> weight
    boost::unordered_map<std::pair<int, int>, int> node_and_community_cache; // (node_id, community_id) -> number

    inline int getCommunityForNode(int node_id) const {
        return community_id[node_id];
    }

    inline void updateWeightForCommunity(int community_id, int weight_update = 1) {
        community_weights[community_id] += weight_update;
    }

    inline void clearCache() {
        node_and_community_cache.clear();
    }
};

struct IntermediaryCommunityId {
    int community_id;
    int level;
    IntermediaryCommunityId *parent;
};

bool edgeBetweenCommunities(const std::vector<int>& community1, const std::vector<int>& community2, const Graph& graph);
bool isSubset(std::vector<int>& set1, std::vector<int>& set2);
int countEdgesBetweenNodeAndCommunity(const Graph& graph, int node_id, int community_id, Partitions& partitions);
int countEdgesBetweenCommunities(int community_id, int subset, Partitions& refined_partitions, Partitions& partitions, const Graph& graph);
int getNumOfPossibleEdges(int n);
std::pair<double, int> computeDeltaCPM(Partitions& partitions, int node_id, int new_community_id, const Graph& graph, double gamma = 1.0);

}  // namespace leiden_alg

#endif // LEIDEN_UTILS_HPP