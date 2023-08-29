#pragma once

#include <mgp.hpp>

namespace Label {

/* exists constants */
constexpr std::string_view kReturnExists = "exists";
constexpr std::string_view kProcedureExists = "exists";
constexpr std::string_view kArgumentsNode = "node";
constexpr std::string_view kArgumentsLabel = "label";
constexpr std::string_view kResultExists = "exists";

void Exists(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Label
