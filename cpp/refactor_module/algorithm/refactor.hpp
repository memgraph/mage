#pragma once

#include <mgp.hpp>
#include <string>
#include <string_view>
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

/* clone_nodes constants */
constexpr const std::string_view kProcedureCloneNodes = "clone_nodes";
constexpr const std::string_view kReturnClonedNodeId = "cloned_node_id";
constexpr const std::string_view kReturnNewNode = "new_node";
constexpr const std::string_view kArgumentsNodesToClone = "nodes";
constexpr const std::string_view kArgumentsCloneRels = "clone_rels";
constexpr const std::string_view kArgumentsSkipPropClone = "skip_props";
constexpr const std::string_view kResultClonedNodeId = "cloned_node_id";
constexpr const std::string_view kResultNewNode = "new_node";

/* clone_subgraph_from_paths constants */
constexpr const std::string_view kProcedureCSFP = "clone_subgraph_from_paths";
constexpr const std::string_view kArgumentsPath = "paths";
constexpr const std::string_view kArgumentsConfigMap = "config";

/* clone_subgraph constants */
constexpr const std::string_view kProcedureCloneSubgraph = "clone_subgraph";
constexpr const std::string_view kArgumentsNodes = "nodes";
constexpr const std::string_view kArgumentsRels = "rels";

/* from constants */
constexpr const std::string_view kProcedureFrom = "from";
constexpr const std::string_view kFromArg1 = "relationship";
constexpr const std::string_view kFromArg2 = "new_from";
constexpr const std::string_view kFromResult = "relationship";

/* to constants */
constexpr const std::string_view kProcedureTo = "to";
constexpr const std::string_view kToArg1 = "relationship";
constexpr const std::string_view kToArg2 = "new_to";
constexpr const std::string_view kToResult = "relationship";

/* rename_label constants */
constexpr std::string_view kProcedureRenameLabel = "rename_label";
constexpr std::string_view kRenameLabelArg1 = "old_label";
constexpr std::string_view kRenameLabelArg2 = "new_label";
constexpr std::string_view kRenameLabelArg3 = "nodes";
constexpr std::string_view kRenameLabelResult = "nodes_changed";

/* rename_node_property constants */
constexpr std::string_view kProcedureRenameNodeProperty = "rename_node_property";
constexpr std::string_view kRenameNodePropertyArg1 = "old_property";
constexpr std::string_view kRenameNodePropertyArg2 = "new_property";
constexpr std::string_view kRenameNodePropertyArg3 = "nodes";
constexpr std::string_view kRenameNodePropertyResult = "nodes_changed";

/*collapse constants*/
constexpr std::string_view kProcedureCollapseNode = "collapse_node";
constexpr std::string_view kArgumentNodesCollapseNode = "nodes";
constexpr std::string_view kArgumentTypeCollapseNode = "type";
constexpr std::string_view kReturnIdCollapseNode = "id_collapsed";
constexpr std::string_view kReturnRelationshipCollapseNode = "new_relationship";

/*invert constants*/
constexpr std::string_view kProcedureInvert = "invert";
constexpr std::string_view kArgumentRelationship = "relationship";
constexpr std::string_view kReturnRelationshipInvert = "relationship";
constexpr std::string_view kReturnIdInvert = "id_inverted";

/* delete_and_reconnect constants */
constexpr std::string_view kProcedureDeleteAndReconnect = "delete_and_reconnect";
constexpr std::string_view kDeleteAndReconnectArg1 = "path";
constexpr std::string_view kDeleteAndReconnectArg2 = "nodes";
constexpr std::string_view kDeleteAndReconnectArg3 = "config";
constexpr std::string_view kReturnDeleteAndReconnect1 = "nodes";
constexpr std::string_view kReturnDeleteAndReconnect2 = "relationships";

void From(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void To(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void RenameLabel(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void RenameNodeProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Categorize(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void InsertCloneNodesRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const int cycle_id,
                            const int node_id);

void CloneNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void CloneNodesAndRels(mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory,
                       const std::vector<mgp::Node> &nodes, const std::vector<mgp::Relationship> &rels,
                       const mgp::Map &config_map);

void CloneSubgraphFromPaths(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void CloneSubgraph(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void TransferProperties(const mgp::Node &node, mgp::Relationship &rel);

void Collapse(mgp::Graph &graph, const mgp::Node &node, const std::string &type,
              const mgp::RecordFactory &record_factory);

void CollapseNode(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Invert(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void InvertRel(mgp::Graph &graph, mgp::Relationship &rel);

void DeleteAndReconnect(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Refactor
