#pragma once

#include <mgp.hpp>
#include <unordered_set>

namespace Create {

constexpr std::string_view kReturnRelProp = "relationship";

constexpr std::string_view kProcedureSetRelProp = "set_rel_property";

constexpr std::string_view kArgumentsRelationship = "input_rel";
constexpr std::string_view kArgumentsKey = "input_key";
constexpr std::string_view kArgumentsValue = "input_value";

constexpr std::string_view kResultRelProp = "relationship";

void SetElementProp(mgp::Relationship &element, const mgp::List &prop_key_list, const mgp::List &prop_value_list,
                    const mgp::RecordFactory &record_factory);
void ProcessElement(const mgp::Value &element, const mgp::Graph graph, const mgp::List &prop_key_list,
                    const mgp::List &prop_value_list, const mgp::RecordFactory &record_factory,
                    std::unordered_set<mgp::Id> &relIds);
void SetRelProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Create
