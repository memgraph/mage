#pragma once

#include <mgp.hpp>

namespace Map {
constexpr std::string_view kProcedureFromLists = "from_lists";
constexpr std::string_view kArgumentListKeysFromLists = "list_keys";
constexpr std::string_view kArgumentListValuesFromLists = "list_values";
constexpr std::string_view kReturnListFromLists = "result";
void FromLists(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Map
