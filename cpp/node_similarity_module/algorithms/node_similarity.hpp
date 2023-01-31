#include <mgp.hpp>

namespace node_similarity_algs {

std::unordered_map<uint64_t, std::unordered_set<uint64_t>> GetNeighbours(const auto graph) {
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> neighbors;
    for (const auto node: graph.Nodes()) {
        std::unordered_set<uint64_t> neighbors_id;
        for (const auto relationship: node.OutRelationships()) {
            neighbors_id.insert(relationship.To().Id().AsUint());
        }
        for (const auto relationship: node.InRelationships()) {
            neighbors_id.insert(relationship.From().Id().AsUint());
        }
        neighbors[node.Id().AsUint()] = neighbors_id;
    }
    return neighbors;
}

}  // namespace node_similarity_algs

namespace node_similarity_util {

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (std::pair<T1, T2> const &pair) const
    {
        std::size_t h1 = std::hash<T1>()(pair.first);
        std::size_t h2 = std::hash<T2>()(pair.second);

        return h1 ^ h2;
    }
};

}  // namespace node_similarity_util