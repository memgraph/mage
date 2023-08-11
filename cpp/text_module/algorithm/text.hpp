#pragma once

#include <mgp.hpp>

namespace Text {

/* join constants */
constexpr const std::string_view kProcedureJoin = "join";
constexpr const std::string_view kJoinArg1 = "strings";
constexpr const std::string_view kJoinArg2 = "delimiter";
constexpr const std::string_view kResultJoin = "string";

void Join(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Text
