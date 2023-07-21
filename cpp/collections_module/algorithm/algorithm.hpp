#pragma once

#include <mgp.hpp>

namespace Collections {

constexpr const std::string_view kProcedureSumLongs = "sumLongs";

constexpr const std::string_view kReturnSumLongs = "sum";

constexpr const std::string_view kNumbersList = "list_of_numbers";

constexpr const char *kResultSumLongs = "sum";

void SumLongs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
