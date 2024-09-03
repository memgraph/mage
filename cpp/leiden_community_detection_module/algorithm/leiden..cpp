#include "leiden.hpp"

namespace leiden_alg {


constexpr int kReplaceMap = 0;
constexpr int kThreadsOpt = 1;
constexpr int kNumColors = 16;

std::vector<std::int64_t> GetCommunities(const mg_graph::GraphView<> &graph) {
    auto number_of_nodes = graph.Nodes().size();


    return std::vector<std::int64_t>();
}

}  // namespace leiden_alg
