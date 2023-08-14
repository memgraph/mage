#pragma once

#include <mgp.hpp>
#include <string_view>

namespace Create {

/* remove_rel_properties constants */
constexpr std::string_view kProcedureRemoveRelProperties = "remove_rel_properties";
constexpr std::string_view kRemoveRelPropertiesArg1 = "relationships";
constexpr std::string_view kRemoveRelPropertiesArg2 = "keys";
constexpr std::string_view kResultRemoveRelProperties = "relationship";

/* relationship constants */
constexpr std::string_view kProcedureRelationship = "relationship";
constexpr std::string_view kRelationshipArg1 = "from";
constexpr std::string_view kRelationshipArg2 = "relationshipType";
constexpr std::string_view kRelationshipArg3 = "properties";
constexpr std::string_view kRelationshipArg4 = "to";
constexpr std::string_view kResultRelationship = "relationship";

void RemoveRelProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Relationship(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Create
