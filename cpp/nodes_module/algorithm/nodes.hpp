#pragma once

#include <mgp.hpp>
#include <string>

namespace Nodes {

/*link constants*/
constexpr std::string_view kProcedureLink = "link";
constexpr std::string_view kArgumentNodesLink = "nodes";
constexpr std::string_view kArgumentTypeLink = "type";

void Link(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Nodes
