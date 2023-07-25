#pragma once

#include <mgp.hpp>

namespace Map {

constexpr std::string_view kReturnFromPairs = "map";

constexpr std::string_view kProcedureFromPairs = "from_pairs";

constexpr std::string_view kArgumentsInputList = "input_list";

constexpr std::string_view kResultFromPairs = "map";

void FromPairs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Map
