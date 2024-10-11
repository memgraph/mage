#include "leiden_utils.hpp"
#include <cstdint>
namespace leiden_alg {

// create new intermediary community ids -> nodes that are in community i are children of the new intermediary community id
void CreateIntermediaryCommunities(Dendrogram &intermediary_communities, const std::vector<std::vector<std::uint64_t>> &communities, std::uint64_t current_level) {
    for (std::uint64_t i = 0; i < communities.size(); i++) {
        const auto new_intermediary_community_id = 
            std::make_shared<IntermediaryCommunityId>(IntermediaryCommunityId{i, current_level + 1, nullptr});
        for (const auto &node_id : communities[i]) {
            intermediary_communities[current_level][node_id]->parent = new_intermediary_community_id;    
        }
        intermediary_communities[current_level + 1].push_back(new_intermediary_community_id);
    }
}

}  // namespace leiden_alg
