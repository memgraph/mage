#pragma once

#include <mgp.hpp>
#include <string>

namespace Refactor {
    void CollapseNode(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Refactor
