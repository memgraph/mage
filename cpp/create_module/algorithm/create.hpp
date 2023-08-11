#pragma once

#include <mgp.hpp>
#include <unordered_set>

namespace Create {

constexpr std::string_view kReturnNode = "node";
constexpr std::string_view kProcedureNode = "node";
constexpr std::string_view kArgumentsLabelsList = "labels";
constexpr std::string_view kArgumentsProperties = "properties";
constexpr std::string_view kResultNode = "node";

constexpr std::string_view kReturnProperties = "nodes";
constexpr std::string_view kProcedureSetProperties = "set_properties";
constexpr std::string_view kArgumentsNodes = "input_nodes";
constexpr std::string_view kArgumentsKeys = "input_keys";
constexpr std::string_view kArgumentsValues = "input_values";
constexpr std::string_view kResultProperties = "nodes";

constexpr std::string_view kReturnRemoveLabels = "nodes";
constexpr std::string_view kProcedureRemoveLabels = "remove_labels";
constexpr std::string_view kArgumentsLabels = "labels";
constexpr std::string_view kResultRemoveLabels = "nodes";
  
constexpr std::string_view kReturnRelProp = "relationship";
constexpr std::string_view kProcedureSetRelProp = "set_rel_property";
constexpr std::string_view kArgumentsRelationship = "input_rel";
constexpr std::string_view kArgumentsKey = "input_key";
constexpr std::string_view kArgumentsValue = "input_value";
constexpr std::string_view kResultRelProp = "relationship";

void Node(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void SetElementProp(mgp::Node &element, const mgp::List &prop_key_list, const mgp::List &prop_value_list,
                    const mgp::RecordFactory &record_factory);
void ProcessElement(const mgp::Value &element, const mgp::Graph graph, const mgp::List &prop_key_list,
                    const mgp::List &prop_value_list, const mgp::RecordFactory &record_factory);
void SetProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void RemoveElementLabels(mgp::Node &element, const mgp::List &labels, const mgp::RecordFactory &record_factory);
void ProcessElement(const mgp::Value &element, const mgp::Graph graph, const mgp::List &labels,
                    const mgp::RecordFactory &record_factory);
void RemoveLabels(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
  
void SetElementProp(mgp::Relationship &element, const mgp::List &prop_key_list, const mgp::List &prop_value_list,
                    const mgp::RecordFactory &record_factory);
void ProcessElement(const mgp::Value &element, const mgp::Graph graph, const mgp::List &prop_key_list,
                    const mgp::List &prop_value_list, const mgp::RecordFactory &record_factory,
                    std::unordered_set<mgp::Id> &relIds);
void SetRelProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Create
