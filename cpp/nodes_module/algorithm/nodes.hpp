#pragma once

#include <mgp.hpp>

namespace Nodes {

constexpr const std::string_view kProcedureRelationshipTypes = "relationship_types";
constexpr const std::string_view kRelationshipTypesArg1 = "nodes";
constexpr const std::string_view kRelationshipTypesArg2 = "types";
constexpr const std::string_view kResultRelationshipTypes = "relationship_types";

void RelationshipTypes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Nodes
