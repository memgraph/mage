#pragma once

#include <mgp.hpp>
#include <unordered_set>

namespace Create {

constexpr std::string_view kReturnRemoveLabels = "nodes";

constexpr std::string_view kProcedureRemoveLabels = "remove_labels";

constexpr std::string_view kArgumentsNodes = "input_nodes";
constexpr std::string_view kArgumentsLabels = "labels";

constexpr std::string_view kResultRemoveLabels = "nodes";

void RemoveElementLabels(mgp::Node &element, const mgp::List &labels, const mgp::RecordFactory &record_factory);
void ProcessElement(const mgp::Value &element, const mgp::Graph graph, const mgp::List &labels,
                    const mgp::RecordFactory &record_factory);
void RemoveLabels(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Create
