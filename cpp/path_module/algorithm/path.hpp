#pragma once

#include <mgp.hpp>
#include <unordered_set>
namespace Path {
    constexpr std::string_view kProcedureExpand = "expand";
    void Expand(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Path