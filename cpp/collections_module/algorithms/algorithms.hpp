#pragma once

#include <mgp.hpp>

namespace Collections {

constexpr std::string_view kReturnMax = "max";

constexpr std::string_view kProcedureMax = "max";

constexpr std::string_view kArgumentsInputList = "input_list";

constexpr std::string_view kResultMax = "max";

void Max(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
