#pragma once

#include <mgp.hpp>

namespace Node {

/* relationship.exists constants */
constexpr std::string_view kReturnRelExists = "rel_exists";
constexpr std::string_view kProcedureRelExists = "relationship_exists";
constexpr std::string_view kArgumentsNode = "node";
constexpr std::string_view kArgumentsPattern = "pattern";
constexpr std::string_view kResultRelExists = "rel_exists";

bool FindRelationship(std::string_view type, mgp::Relationships relationships);
void RelExists(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Node
