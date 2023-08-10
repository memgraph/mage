#pragma once

#include <mgp.hpp>
#include <string_view>

namespace Create {

/* remove_rel_properties constants */
constexpr const std::string_view kProcedureRemoveRelProperties = "remove_rel_properties";
constexpr const std::string_view kRemoveRelPropertiesArg1 = "relationships";
constexpr const std::string_view kRemoveRelPropertiesArg2 = "keys";
constexpr const std::string_view kResultRemoveRelProperties = "relationship";

/* relationship constants */
constexpr const std::string_view kProcedureRelationship = "relationship";
constexpr const std::string_view kRelationshipArg1 = "from";
constexpr const std::string_view kRelationshipArg2 = "relationshipType";
constexpr const std::string_view kRelationshipArg3 = "properties";
constexpr const std::string_view kRelationshipArg4 = "to";
constexpr const std::string_view kResultRelationship = "relationship";

void RemoveRelProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Relationship(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Create
