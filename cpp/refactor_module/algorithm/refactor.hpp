#pragma once

#include <mgp.hpp>

namespace Refactor {

/* from constants */
constexpr const std::string_view kProcedureFrom = "from";
constexpr const std::string_view kFromArg1 = "relationship";
constexpr const std::string_view kFromArg2 = "new_from";

/* to constants */
constexpr const std::string_view kProcedureTo = "to";
constexpr const std::string_view kToArg1 = "relationship";
constexpr const std::string_view kToArg2 = "new_to";

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

void From(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void To(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void RenameLabel(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void RenameNodeProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Refactor
