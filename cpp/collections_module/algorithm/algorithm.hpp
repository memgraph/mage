#pragma once
#include <mgp.hpp>
#include <string>

namespace Collections {
constexpr std::string_view kReturnValueUnionAll = "return_list";
constexpr std::string_view kProcedureUnionAll = "unionAll";
constexpr std::string_view kArgumentList1UnionAll = "list1";
constexpr std::string_view kArgumentList2UnionAll = "list2";

void unionAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
