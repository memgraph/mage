#pragma once

#include <mgp.hpp>

#include <string>

namespace Create {
/*nodes constants*/
constexpr std::string_view kProcedureNodes = "nodes";
constexpr std::string_view kArgumentLabelsNodes = "labels";
constexpr std::string_view kArgumentPropertiesNodes = "properties";
constexpr std::string_view kReturnNodes = "node";

/*set_property constants*/
constexpr std::string_view kProcedureSetProperty = "set_property";
constexpr std::string_view kArgumentNodeSetProperty = "nodes";
constexpr std::string_view kArgumentKeySetProperty = "key";
constexpr std::string_view kArgumentValueSetProperty = "value";
constexpr std::string_view kReturntSetProperty = "node";

/*remove_properties constants*/
constexpr std::string_view kProcedureRemoveProperties = "remove_properties";
constexpr std::string_view kArgumentNodeRemoveProperties = "nodes";
constexpr std::string_view kArgumentKeysRemoveProperties = "list_keys";
constexpr std::string_view kReturntRemoveProperties = "node";

void SetProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void StartSetProperty(mgp::Node &node, const std::string &string, mgp::Value &value,
                      const mgp::RecordFactory &record_factory);

void RemoveProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void StartRemoveProperties(mgp::Node &graph_node, const mgp::List &list_keys, const mgp::RecordFactory &record_factory);

void Nodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Create
