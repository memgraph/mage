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
void NodeTypeProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void RelTypeProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Schema