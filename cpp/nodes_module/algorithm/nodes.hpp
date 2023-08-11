#pragma once

#include <mgp.hpp>
#include <string>


namespace Nodes {

/* relationship_types constants */
constexpr const std::string_view kProcedureRelationshipTypes = "relationship_types";
constexpr const std::string_view kRelationshipTypesArg1 = "nodes";
constexpr const std::string_view kRelationshipTypesArg2 = "types";
constexpr const std::string_view kResultRelationshipTypes = "relationship_types";

/* delete constants */
constexpr const std::string_view kProcedureDelete = "delete";
constexpr const std::string_view kDeleteArg1 = "nodes";

/*link constants*/
constexpr size_t minimumNodeListSize = 2;
constexpr std::string_view kProcedureLink = "link";
constexpr std::string_view kArgumentNodesLink = "nodes";
constexpr std::string_view kArgumentTypeLink = "type";

void RelationshipTypes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Delete(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Link(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Nodes
