#pragma once

#include <mgp.hpp>
#include <string>
#include <sstream>

namespace Collections {
//Contains constants
constexpr std::string_view kReturnValueContains = "output";
constexpr std::string_view kProcedureContains = "contains";
constexpr std::string_view kArgumentListContains = "list";
constexpr std::string_view kArgumentValueContains = "value";
//UnionAll constants
constexpr std::string_view kReturnValueUnionAll = "return_list";
constexpr std::string_view kProcedureUnionAll = "unionAll";
constexpr std::string_view kArgumentList1UnionAll = "list1";
constexpr std::string_view kArgumentList2UnionAll = "list2";
//Min constants
constexpr std::string_view kReturnValueMin = "min";
constexpr std::string_view kProcedureMin = "min";
constexpr std::string_view kArgumentListMin = "list";
//ToSet constants
constexpr std::string_view kReturnToSet = "result";
constexpr std::string_view kProcedureToSet = "to_set";
constexpr std::string_view kArgumentListToSet = "list";
//Partition constants
constexpr std::string_view kReturnValuePartition = "result";
constexpr std::string_view kProcedurePartition = "partition";
constexpr std::string_view kArgumentListPartition = "list";
constexpr std::string_view kArgumentSizePartition = "partition_size";

void Contains(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void UnionAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void Min(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void ToSet(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void Partition(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections

