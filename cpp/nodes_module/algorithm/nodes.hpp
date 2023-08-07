#pragma once

#include <mgp.hpp>
#include <string>

namespace Nodes {

/*relationships_exist constants*/
constexpr std::string_view kProcedureRelationshipsExist= "relationships_exist";
constexpr std::string_view kReturnRelationshipsExist= "result";
constexpr std::string_view kArgumentNodesRelationshipsExist= "nodes";
constexpr std::string_view kArgumentRelationshipsRelationshipsExist= "relationships";
void RelationshipsExist(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
bool RelationshipExist(const mgp::Node &node, std::string &rel_type);
} //namespace Nodesž
