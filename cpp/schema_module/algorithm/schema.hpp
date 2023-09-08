#pragma once

#include <mgp.hpp>

namespace Schema {

/*NodeTypeProperties and RelTypeProperties constants*/
constexpr std::string_view kProcedureNodeType = "node_type_properties";
constexpr std::string_view kProcedureRelType = "rel_type_properties";
constexpr std::string_view kReturnLabels = "labels";
constexpr std::string_view kReturnRelType = "rel_type";
constexpr std::string_view kReturnPropertyName = "property_name";
constexpr std::string_view kReturnPropertyType = "property_type";
constexpr std::string_view kReturnMandatory = "mandatory";

std::string TypeOf(const mgp::Type &type);
void ProcessPropertiesNode(mgp::Record &record, const mgp::List &labels, const std::string &propertyName,
                           const std::string &propertyType, const bool &mandatory);
void ProcessPropertiesRel(mgp::Record &record, const std::string_view &type, const std::string &propertyName,
                          const std::string &propertyType, const bool &mandatory);
void NodeTypeProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void RelTypeProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Schema
