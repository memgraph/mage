#pragma once

#include <mgp.hpp>
#include <string_view>

namespace Meta {

/* update constants */
constexpr const std::string_view kProcedureUpdate = "update";
constexpr const std::string_view kStatsArg1 = "createdObjects";
constexpr const std::string_view kStatsArg2 = "deletedObjects";
constexpr const std::string_view kStatsArg3 = "removedVertexProperties";
constexpr const std::string_view kStatsArg4 = "removedEdgeProperties";
constexpr const std::string_view kStatsArg5 = "setVertexLabels";
constexpr const std::string_view kStatsArg6 = "removedVertexLabels";

/* stats constants */
constexpr const std::string_view kProcedureStats = "stats";
constexpr const std::string_view kReturnStats = "stats";

void Update(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Stats(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Meta
