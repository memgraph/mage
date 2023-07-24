#pragma once

#include <mgp.hpp>

namespace Collections {

constexpr std::string_view kReturnCS = "contains";

constexpr std::string_view kProcedureCS = "contains_sorted";

constexpr std::string_view kArgumentInputList = "input_list";
constexpr std::string_view kArgumentElement = "element";

constexpr std::string_view kResultCS = "contains";

void ContainsSorted(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
