#pragma once

#include <mgp.hpp>

namespace Text {

/* join constants */
constexpr const std::string_view kProcedureJoin = "test";
constexpr const std::string_view kJoinArg1 = "edge";
constexpr const std::string_view kJoinArg2 = "node";
constexpr const std::string_view kResultJoin = "node";

void Join(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Text
