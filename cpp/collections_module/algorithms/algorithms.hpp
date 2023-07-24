#pragma once

#include <mgp.hpp>

namespace Collections {

constexpr std::string_view kReturnUnion = "union";

constexpr std::string_view kProcedureUnion = "union";

constexpr std::string_view kArgumentsInputList1 = "input_list1";
constexpr std::string_view kArgumentsInputList2 = "input_list2";

constexpr std::string_view kResultUnion = "union";

void Union(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
