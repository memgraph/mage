#pragma once

#include <sys/types.h>
#include <mgp.hpp>

namespace KWeightedShortestPath {

/* KWeightedShortestPath constants */
constexpr const char *kProcedure = "k_weighted_shortest_path";
constexpr const char *kArgumentStartNode = "start_node";
constexpr const char *kArgumentEndNode = "end_node";
constexpr const char *kArgumentNumberOfWeightedShortestPaths = "number_of_weighted_shortest_paths";
constexpr const char *kResult = "result";
constexpr const int64_t kDefaultNumberOfWeightedShortestPaths = 5;

void KWeightedShortestPath(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace KWeightedShortestPath
