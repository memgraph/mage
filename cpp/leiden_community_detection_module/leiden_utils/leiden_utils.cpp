#include "leiden_utils.hpp"
#include <cstdint>
namespace leiden_alg {

std::vector<std::uint64_t> calculateEdgeWeightsPerCommunity(const Partitions &partitions, const Graph &graph) {
    std::vector<std::uint64_t> edge_weights_per_community(partitions.communities.size(), 0);
    for (const auto & community : partitions.communities) {
        for (const auto &node : community) {
            edge_weights_per_community[partitions.getCommunityForNode(node)] += graph.neighbors(node).size();
        }
    }
    return edge_weights_per_community;
}

}  // namespace leiden_alg
