#pragma once

#include <mgp.hpp>

namespace Map {

constexpr std::string_view kReturnRemoveKey = "removed";
constexpr std::string_view kProcedureRemoveKey = "remove_key";
constexpr std::string_view kArgumentsInputMap = "input_map";
constexpr std::string_view kArgumentsKey = "key";
constexpr std::string_view kArgumentsIsRecursive = "recursive_map";
constexpr std::string_view kResultRemoveKey = "removed";

constexpr std::string_view kReturnFromPairs = "map";
constexpr std::string_view kProcedureFromPairs = "from_pairs";
constexpr std::string_view kArgumentsInputList = "input_list";
constexpr std::string_view kResultFromPairs = "map";

constexpr std::string_view kReturnMerge = "merged";
constexpr std::string_view kProcedureMerge = "merge";
constexpr std::string_view kArgumentsInputMap1 = "input_map1";
constexpr std::string_view kArgumentsInputMap2 = "input_map2";
constexpr std::string_view kResultMerge = "merged";

void RemoveKey(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void RemoveRecursion(mgp::Map &result, bool recursive, std::string_view key);
void FromPairs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void Merge(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Map
