#pragma once

#include <mgp.hpp>
#include <string>
#include <unordered_set>
namespace Map {
constexpr std::string_view kReturnRemoveKeys = "result";

constexpr std::string_view kProcedureRemoveKeys = "remove_keys";

constexpr std::string_view kArgumentsInputMapRemoveKeys = "input_map";
constexpr std::string_view kArgumentsKeysListRemoveKeys = "keys_list";
constexpr std::string_view kArgumentsRecursiveRemoveKeys = "recursive";
void removeRecursionSet(mgp::Map &result, bool recursive, std::unordered_set<std::string_view> &set);
void RemoveKeys(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Map
