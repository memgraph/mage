#pragma once

#include <algorithm>
#include <list>
#include <vector>

#include <mgp.hpp>

namespace Collections {

constexpr std::string_view kReturnSort = "sorted";

constexpr std::string_view kProcedureSort = "sort";

constexpr std::string_view kArgumentsInputList = "input_list";

constexpr std::string_view kResultSort = "sorted";

void Sort(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
