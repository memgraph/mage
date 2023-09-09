#pragma once

#include <mgp.hpp>

namespace Refactor {
/* categorize constants */

constexpr const std::string_view kProcedureCategorize = "categorize";
constexpr const std::string_view kReturnCategorize = "status";

constexpr const std::string_view kArgumentsCatSourceKey = "original_prop_key";
constexpr const std::string_view kArgumentsCatRelType = "rel_type";
constexpr const std::string_view kArgumentsCatRelOutgoing = "is_outgoing";
constexpr const std::string_view kArgumentsCatLabelName = "new_label";
constexpr const std::string_view kArgumentsCatPropKey = "new_prop_name_key";
constexpr const std::string_view kArgumentsCopyPropKeys = "copy_props_list";

constexpr const std::string_view kResultCategorize = "status";

void Categorize(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

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

/* clone_subgraph_from_paths constants */

constexpr const std::string_view kProcedureCSFP = "clone_subgraph_from_paths";
constexpr const std::string_view kReturnClonedNodeId = "cloned_node_id";
constexpr const std::string_view kReturnNewNode = "new_node";
constexpr const std::string_view kArgumentsPath = "paths";
constexpr const std::string_view kArgumentsConfigMap = "config";
constexpr const std::string_view kResultClonedNodeId = "cloned_node_id";
constexpr const std::string_view kResultNewNode = "new_node";

/* clone_subgraph constants */

constexpr const std::string_view kProcedureCloneSubgraph = "clone_subgraph";
constexpr const std::string_view kArgumentsNodes = "nodes";
constexpr const std::string_view kArgumentsRels = "rels";

void InsertCloneNodesRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const int cycle_id,
                            const int node_id);
void CloneNodesAndRels(mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory,
                       const std::vector<mgp::Node> &nodes, const std::vector<mgp::Relationship> &rels,
                       const mgp::Map &config_map);
void CloneSubgraphFromPaths(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void CloneSubgraph(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Refactor
