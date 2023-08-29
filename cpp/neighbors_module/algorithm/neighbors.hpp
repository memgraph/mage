#pragma once

#include <mgp.hpp>

namespace Neighbors {

constexpr std::string_view kReturnAtHop = "nodes";
constexpr std::string_view kProcedureAtHop = "at_hop";
  
constexpr std::string_view kReturnByHop = "nodes";
constexpr std::string_view kProcedureByHop = "by_hop";

constexpr std::string_view kArgumentsNode = "node";
constexpr std::string_view kArgumentsRelType = "rel_type";
constexpr std::string_view kArgumentsDistance = "distance";

constexpr std::string_view kResultAtHop = "nodes";
constexpr std::string_view kResultByHop = "nodes";

void AtHop(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void ByHop(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Neighbors
