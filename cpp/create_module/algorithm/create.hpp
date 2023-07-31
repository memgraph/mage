#pragma once

#include <mgp.hpp>

namespace Create {
constexpr std::string_view kProcedureNodes = "nodes";
constexpr std::string_view kArgumentLabelsNodes = "labels";
constexpr std::string_view kArgumentPropertiesNodes = "properties";
constexpr std::string_view kReturnNodes = "node";
void Nodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Create
