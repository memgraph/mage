#pragma once

#include <mgp.hpp>

namespace Path {

constexpr std::string_view kReturnSubgraphAll = "path";
constexpr std::string_view kProcedureSubgraphAll = "subgraph_all";
constexpr std::string_view kArgumentsStart = "start_node";
constexpr std::string_view kArgumentsConfig = "config";
constexpr std::string_view kResultSubgraphAll = "path";

void SubgraphAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Path
