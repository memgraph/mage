#pragma once

#include <mgp.hpp>
#include <string>

namespace Refactor {

constexpr std::string_view kProcedureCollapseNode = "collapse_node";
constexpr std::string_view kArgumentNodesCollapseNode = "nodes";
constexpr std::string_view kArgumentTypeCollapseNode = "type";
constexpr std::string_view kReturnIdCollapseNode = "id_collapsed";
constexpr std::string_view kReturnRelationshipCollapseNode = "new_relationship";

void TransferProperties(const mgp::Node &node, mgp::Relationship &rel);
void Collapse(mgp::Graph &graph, const mgp::Node &node, const std::string &type,
              const mgp::RecordFactory &record_factory);
void CollapseNode(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Refactor
