#pragma once

#include <mgp.hpp>

namespace Map {

constexpr std::string_view kReturnRemoveKey = "removed";

constexpr std::string_view kProcedureRemoveKey = "remove_key";

constexpr std::string_view kArgumentsInputMap = "input_map";
constexpr std::string_view kArgumentsKey = "key";
constexpr std::string_view kArgumentsRecursiveMap = "recursive_map";

constexpr std::string_view kResultRemoveKey = "removed";

void RemoveKey(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void removeRecursion(mgp::Map &removed, bool recursive, std::string_view key);

}  // namespace Map
