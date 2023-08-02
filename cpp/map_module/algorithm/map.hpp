#pragma once

#include <mgp.hpp>
#include <string>
#include <string_view>
#include <unordered_set>

namespace Map {

/* from_nodes constants */
constexpr const std::string_view kProcedureFromNodes = "from_nodes";
constexpr const std::string_view kFromNodesArg1 = "label";
constexpr const std::string_view kFromNodesArg2 = "property";
constexpr const std::string_view kResultFromNodes = "map";

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

/* remove_key constants */
constexpr std::string_view kReturnRemoveKey = "removed";
constexpr std::string_view kProcedureRemoveKey = "remove_key";
constexpr std::string_view kArgumentsInputMap = "input_map";
constexpr std::string_view kArgumentsKey = "key";
constexpr std::string_view kArgumentsIsRecursive = "recursive_map";
constexpr std::string_view kResultRemoveKey = "removed";

/* from_pairs constants */
constexpr std::string_view kReturnFromPairs = "map";
constexpr std::string_view kProcedureFromPairs = "from_pairs";
constexpr std::string_view kArgumentsInputList = "input_list";
constexpr std::string_view kResultFromPairs = "map";

/* merge constants */
constexpr std::string_view kReturnMerge = "merged";
constexpr std::string_view kProcedureMerge = "merge";
constexpr std::string_view kArgumentsInputMap1 = "input_map1";
constexpr std::string_view kArgumentsInputMap2 = "input_map2";
constexpr std::string_view kResultMerge = "merged";

/* flatten constants */
constexpr std::string_view kReturnValueFlatten = "result";
constexpr std::string_view kProcedureFlatten = "flatten";
constexpr std::string_view kArgumentMapFlatten = "map";
constexpr std::string_view kArgumentDelimiterFlatten = "delimiter";

/* from_lists constants */
constexpr std::string_view kProcedureFromLists = "from_lists";
constexpr std::string_view kArgumentListKeysFromLists = "list_keys";
constexpr std::string_view kArgumentListValuesFromLists = "list_values";
constexpr std::string_view kReturnListFromLists = "result";

/* remove_keys constants */
constexpr std::string_view kReturnRemoveKeys = "result";
constexpr std::string_view kProcedureRemoveKeys = "remove_keys";
constexpr std::string_view kArgumentsInputMapRemoveKeys = "input_map";
constexpr std::string_view kArgumentsKeysListRemoveKeys = "keys_list";
constexpr std::string_view kArgumentsRecursiveRemoveKeys = "recursive";

void FromNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void FromValues(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void SetKey(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void RemoveKey(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void RemoveRecursion(mgp::Map &result, bool recursive, std::string_view key);

void FromPairs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Merge(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Flatten(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void FlattenRecursion(mgp::Map &result, const mgp::Map &input, const std::string &key, const std::string &delimiter);

void FromLists(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void RemoveRecursionSet(mgp::Map &result, bool recursive, std::unordered_set<std::string_view> &set);

void RemoveKeys(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Map
