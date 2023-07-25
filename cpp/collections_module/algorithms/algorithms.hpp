#pragma once

#include <mgp.hpp>

namespace Collections {

constexpr std::string_view kReturnPairs = "pairs";

constexpr std::string_view kProcedurePairs = "pairs";

constexpr std::string_view kInputList = "inputList";

constexpr std::string_view kResultPairs = "pairs";

void Pairs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
