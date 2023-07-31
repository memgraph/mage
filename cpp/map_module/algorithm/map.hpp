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
}  // namespace Map
