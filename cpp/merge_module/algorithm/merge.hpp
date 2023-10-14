#pragma once

#include <mgp.hpp>

namespace Merge {

/* relationship constants */
constexpr const std::string_view kProcedureRelationship = "relationship";
constexpr const std::string_view kRelationshipArg1 = "startNode";
constexpr const std::string_view kRelationshipArg2 = "relationshipType";
constexpr const std::string_view kRelationshipArg3 = "identProps";
constexpr const std::string_view kRelationshipArg4 = "props";
constexpr const std::string_view kRelationshipArg5 = "endNode";
constexpr const std::string_view kRelationshipArg6 = "onMatchProps";
constexpr const std::string_view kRelationshipResult = "rel";

void Relationship(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Merge