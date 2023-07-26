#pragma once

#include <mgp.hpp>
#include <string_view>

namespace Map {

constexpr const std::string_view kProcedureFromNodes = "fromNodes";

constexpr const std::string_view kFromNodesArg1 = "label";
constexpr const std::string_view kFromNodesArg2 = "property";

constexpr const std::string_view kResultFromNodes = "map";

void FromNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Map
