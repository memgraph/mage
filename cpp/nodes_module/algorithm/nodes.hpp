#pragma once

#include <mgp.hpp>

namespace Nodes {

/* relationship_types constants */
constexpr const std::string_view kProcedureRelationshipTypes = "relationship_types";
constexpr const std::string_view kRelationshipTypesArg1 = "nodes";
constexpr const std::string_view kRelationshipTypesArg2 = "types";
constexpr const std::string_view kResultRelationshipTypes = "relationship_types";

/* delete constants */
constexpr const std::string_view kProcedureDelete = "delete";
constexpr const std::string_view kDeleteArg1 = "nodes";

void RelationshipTypes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Delete(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Nodes
