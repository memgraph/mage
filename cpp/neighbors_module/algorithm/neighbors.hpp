#pragma once

#include <mgp.hpp>

namespace Neighbors {

constexpr std::string_view kReturnAtHop = "node";

constexpr std::string_view kProcedureAtHop = "at_hop";

constexpr std::string_view kArgumentsNode = "node";
constexpr std::string_view kArgumentsRelType = "rel_type";
constexpr std::string_view kArgumentsDistance = "distance";

constexpr std::string_view kResultAtHop = "node";

void AtHop(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Neighbors
