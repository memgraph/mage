#pragma once

#include <mgp.hpp>

namespace Create {

constexpr std::string_view kReturnRemoveLabels = "nodes";

constexpr std::string_view kProcedureRemoveLabels = "remove_labels";

constexpr std::string_view kArgumentsNodes = "input_nodes";
constexpr std::string_view kArgumentsLabels = "labels";

constexpr std::string_view kResultRemoveLabels = "nodes";

void RemoveLabels(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Create
