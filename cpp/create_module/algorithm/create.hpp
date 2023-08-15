#pragma once

#include <mgp.hpp>
#include <string_view>

namespace Create {

/* set_rel_properties constants */
constexpr std::string_view kProcedureSetRelProperties = "set_rel_properties";
constexpr std::string_view kSetRelPropertiesArg1 = "relationships";
constexpr std::string_view kSetRelPropertiesArg2 = "keys";
constexpr std::string_view kSetRelPropertiesArg3 = "values";
constexpr std::string_view kResultSetRelProperties = "relationship";

/* relationship constants */
constexpr std::string_view kProcedureRelationship = "relationship";
constexpr std::string_view kRelationshipArg1 = "from";
constexpr std::string_view kRelationshipArg2 = "relationshipType";
constexpr std::string_view kRelationshipArg3 = "properties";
constexpr std::string_view kRelationshipArg4 = "to";
constexpr std::string_view kResultRelationship = "relationship";

void SetRelProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Relationship(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Create
