#pragma once

#include <mgp.hpp>

namespace Collections {

constexpr std::string_view kReturnSum = "sum";

constexpr std::string_view kProcedureSum = "sum";

constexpr std::string_view kInputList = "input_list";

constexpr std::string_view kResultSum = "sum";

void Sum(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
