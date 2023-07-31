#pragma once

#include <mgp.hpp>

#include <string_view>

namespace Collections {

/* sum_longs constants */
constexpr const std::string_view kProcedureSumLongs = "sum_longs";
constexpr const std::string_view kSumLongsArg1 = "numbers";
constexpr const std::string_view kResultSumLongs = "sum";

/* avg constants */
constexpr const std::string_view kProcedureAvg = "avg";
constexpr const std::string_view kAvgArg1 = "numbers";
constexpr const std::string_view kResultAvg = "average";

/* contains_all constants */
constexpr const std::string_view kProcedureContainsAll = "contains_all";
constexpr const std::string_view kContainsAllArg1 = "collection";
constexpr const std::string_view kContainsAllArg2 = "values";
constexpr const std::string_view kResultContainsAll = "contained";

/* intersection constants */
constexpr const std::string_view kProcedureIntersection = "intersection";
constexpr const std::string_view kIntersectionArg1 = "first";
constexpr const std::string_view kIntersectionArg2 = "second";
constexpr const std::string_view kResultIntersection = "intersection";

void SumLongs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Avg(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void ContainsAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Intersection(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
