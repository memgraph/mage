#pragma once

#include <mgp.hpp>

namespace Create {
constexpr std::string_view kProcedureRemoveProperties = "remove_properties";
constexpr std::string_view kArgumentNodeRemoveProperties = "node";
constexpr std::string_view kArgumentKeysRemoveProperties = "list_keys";
constexpr std::string_view kReturntRemoveProperties = "node";
void RemoveProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Create
