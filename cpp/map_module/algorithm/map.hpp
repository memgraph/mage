#pragma once

#include <mgp.hpp>
#include <string_view>

namespace Map {

/* from_values constants */
constexpr const std::string_view kProcedureFromValues = "from_values";
constexpr const std::string_view kFromValuesArg1 = "values";
constexpr const std::string_view kResultFromValues = "map";

/* set_key constants */
constexpr const std::string_view kProcedureSetKey = "set_key";
constexpr const std::string_view kSetKeyArg1 = "map";
constexpr const std::string_view kSetKeyArg2 = "key";
constexpr const std::string_view kSetKeyArg3 = "value";
constexpr const std::string_view kResultSetKey = "map";

void FromValues(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void SetKey(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Map
