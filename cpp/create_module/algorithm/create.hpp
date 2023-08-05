#pragma once

#include <mgp.hpp>
#include <unordered_set>

namespace Create {

constexpr std::string_view kReturnProperties = "nodes";

constexpr std::string_view kProcedureSetProperties = "set_properties";

constexpr std::string_view kArgumentsNodes = "input_nodes";
constexpr std::string_view kArgumentsKeys = "input_keys";
constexpr std::string_view kArgumentsValues = "input_values";

constexpr std::string_view kResultProperties = "nodes";

void SetElementProp(mgp::Node &element, const mgp::List &prop_key_list, const mgp::List &prop_value_list,
                    const mgp::RecordFactory &record_factory);
void ProcessElement(const mgp::Value &element, const mgp::Graph graph, const mgp::List &prop_key_list,
                    const mgp::List &prop_value_list, const mgp::RecordFactory &record_factory);
void SetProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Create
