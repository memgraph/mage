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

void From(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void To(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Refactor
