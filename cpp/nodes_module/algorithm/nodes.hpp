#pragma once

#include <mgp.hpp>

namespace Nodes {

/* delete constants */
constexpr const std::string_view kProcedureDelete = "delete";
constexpr const std::string_view kDeleteArg1 = "nodes";

void Delete(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Nodes
