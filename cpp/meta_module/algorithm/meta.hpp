#pragma once

#include <mgp.hpp>
#include <string_view>

namespace Meta {

/* update constants */
constexpr const std::string_view kProcedureUpdate = "update";
constexpr const std::string_view kUpdateArg1 = "createdObjects";
constexpr const std::string_view kUpdateArg2 = "deletedObjects";
constexpr const std::string_view kUpdateArg3 = "removedVertexProperties";
constexpr const std::string_view kUpdateArg4 = "removedEdgeProperties";
constexpr const std::string_view kUpdateArg5 = "setVertexLabels";
constexpr const std::string_view kUpdateArg6 = "removedVertexLabels";

/* stats constants */
constexpr const std::string_view kProcedureStatsOnline = "stats_online";
constexpr const std::string_view kProcedureStatsOffline = "stats_offline";
constexpr const std::string_view kStatsOnlineArg1 = "update_stats";
constexpr const std::string_view kReturnStats1 = "labelCount";
constexpr const std::string_view kReturnStats2 = "relationshipTypeCount";
constexpr const std::string_view kReturnStats3 = "propertyKeyCount";
constexpr const std::string_view kReturnStats4 = "nodeCount";
constexpr const std::string_view kReturnStats5 = "relationshipCount";
constexpr const std::string_view kReturnStats6 = "labels";
constexpr const std::string_view kReturnStats7 = "relationshipTypes";
constexpr const std::string_view kReturnStats8 = "relationshipTypesCount";
constexpr const std::string_view kReturnStats9 = "stats";

/* reset constants */
constexpr const std::string_view kProcedureReset = "reset";

void Update(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void StatsOnline(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void StatsOffline(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Reset(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Meta
