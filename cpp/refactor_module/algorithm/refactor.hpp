#pragma once

#include <mgp.hpp>

namespace Refactor {
/* clone_nodes constants */

constexpr const std::string_view kProcedureCloneNodes = "clone_nodes";
constexpr const std::string_view kReturnClonedNodeId = "cloned_node_id";
constexpr const std::string_view kReturnNewNode = "new_node";
constexpr const std::string_view kArgumentsNodesToClone = "nodes";
constexpr const std::string_view kArgumentsCloneRels = "clone_rels";
constexpr const std::string_view kArgumentsSkipPropClone = "skip_props";
constexpr const std::string_view kResultClonedNodeId = "cloned_node_id";
constexpr const std::string_view kResultNewNode = "new_node";

void InsertCloneNodesRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const int cycle_id,
                            const int node_id);
void CloneNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Refactor
