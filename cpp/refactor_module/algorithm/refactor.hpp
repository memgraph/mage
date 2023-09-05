#pragma once

#include <mgp.hpp>
#include <string_view>

namespace Refactor {

/* rename_label constants */
constexpr std::string_view kProcedureRenameLabel = "rename_label";
constexpr std::string_view kRenameLabelArg1 = "old_label";
constexpr std::string_view kRenameLabelArg2 = "new_label";
constexpr std::string_view kRenameLabelArg3 = "nodes";

/* rename_node_property constants */
constexpr std::string_view kProcedureRenameNodeProperty = "rename_node_property";
constexpr std::string_view kRenameNodePropertyArg1 = "old_property";
constexpr std::string_view kRenameNodePropertyArg2 = "new_property";
constexpr std::string_view kRenameNodePropertyArg3 = "nodes";

void RenameLabel(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void RenameNodeProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Refactor
