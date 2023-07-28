#pragma once

#include <mgp.hpp>
#include <string_view>

namespace Map {

constexpr const std::string_view kProcedureFromValues = "from_values";

constexpr const std::string_view kFromNodesArg1 = "values";

constexpr const std::string_view kResultFromValues = "map";

void FromValues(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Map
