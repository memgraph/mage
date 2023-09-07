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
}  // namespace Refactor
