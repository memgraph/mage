#pragma once

#include <mgp.hpp>
#include <string>

namespace Map {
constexpr std::string_view kReturnValueFlatten = "result";
constexpr std::string_view kProcedureFlatten = "flatten";
constexpr std::string_view kArgumentMapFlatten = "map";
constexpr std::string_view kArgumentDelimiterFlatten = "delimiter";
void Flatten(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void flattenRecursion(mgp::Map &result, const mgp::Map &input, const std::string &key, const std::string &delimiter);
constexpr std::string_view kProcedureFromLists = "from_lists";
constexpr std::string_view kArgumentListKeysFromLists = "list_keys";
constexpr std::string_view kArgumentListValuesFromLists = "list_values";
constexpr std::string_view kReturnListFromLists = "result";
void FromLists(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Map
