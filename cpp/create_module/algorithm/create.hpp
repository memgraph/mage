#pragma once

#include <mgp.hpp>
#include <string>
namespace Create {
constexpr std::string_view kProcedureSetProperty = "set_property";
constexpr std::string_view kArgumentNodeSetProperty = "node";
constexpr std::string_view kArgumentKeySetProperty = "key";
constexpr std::string_view kArgumentValueSetProperty = "value";
constexpr std::string_view kReturntSetProperty = "node";
void SetProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Create
