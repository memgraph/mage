#pragma once

#include <mgp.hpp>

namespace Create {

constexpr std::string_view kReturnNode = "node";

constexpr std::string_view kProcedureNode = "node";

constexpr std::string_view kArgumentsLabelsList = "labels";
constexpr std::string_view kArgumentsProperties = "properties";

constexpr std::string_view kResultNode = "node";

void Node(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Create
