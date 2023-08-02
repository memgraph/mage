#pragma once

#include <mgp.hpp>

#include <string>
namespace Create {
constexpr std::string_view kProcedureSetProperty = "set_property";
constexpr std::string_view kArgumentNodeSetProperty = "node";
constexpr std::string_view kArgumentKeySetProperty = "key";
constexpr std::string_view kArgumentValueSetProperty = "value";
constexpr std::string_view kReturntSetProperty = "node";

constexpr std::string_view kProcedureRemoveProperties = "remove_properties";
constexpr std::string_view kArgumentNodeRemoveProperties = "node";
constexpr std::string_view kArgumentKeysRemoveProperties = "list_keys";
constexpr std::string_view kReturntRemoveProperties = "node";

void SetProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void RemoveProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Create
