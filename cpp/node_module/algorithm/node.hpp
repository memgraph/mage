#pragma once

#include <unordered_set>

#include <mgp.hpp>

namespace Node {


/* relationship.exists constants */
constexpr std::string_view kReturnRelExists = "exists";
constexpr std::string_view kProcedureRelExists = "relationship_exists";
constexpr std::string_view kArgumentsNode = "node";
constexpr std::string_view kArgumentsPattern = "pattern";
constexpr std::string_view kResultRelExists = "exists";

bool FindRelationship(std::unordered_set<std::string_view> types, mgp::Relationships relationships);
void RelExists(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
  
/* relationship.types constants */
constexpr const std::string_view kProcedureRelationshipTypes = "relationship_types";
constexpr const std::string_view kRelationshipTypesArg1 = "node";
constexpr const std::string_view kRelationshipTypesArg2 = "types";
constexpr const std::string_view kResultRelationshipTypes = "relationship_types";

void RelationshipTypes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Node
