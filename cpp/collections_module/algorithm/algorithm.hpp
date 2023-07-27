#pragma once

#include <mgp.hpp>

#include <string_view>

namespace Collections {

constexpr const std::string_view kProcedureSumLongs = "sumLongs";
constexpr const std::string_view kProcedureAvg = "avg";

constexpr const std::string_view kReturnSumLongs = "sum";
constexpr const std::string_view kReturnAvg = "average";

constexpr const std::string_view kNumbersList = "list_of_numbers";

constexpr const std::string_view kResultSumLongs = "sum";
constexpr const std::string_view kResultAverage = "average";

void SumLongs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Avg(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
