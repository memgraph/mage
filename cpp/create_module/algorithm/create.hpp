#pragma once

#include <mgp.hpp>
#include <unordered_set>

namespace Create {

constexpr std::string_view kReturnRemoveLabels = "nodes";

constexpr std::string_view kProcedureRemoveLabels = "remove_labels";

constexpr std::string_view kArgumentsNodes = "input_nodes";
constexpr std::string_view kArgumentsLabels = "labels";

constexpr std::string_view kResultRemoveLabels = "nodes";

void ProcessElement(std::unordered_set<mgp::Id> &result_set, const mgp::Value &element);
const std::unordered_set<mgp::Id> GetIds(const mgp::Value &argument);
void RemoveLabels(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Create
