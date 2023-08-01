#pragma once

#include <mgp.hpp>
#include <string>
#include <sstream>

namespace Collections {

constexpr std::string_view kProcedureRemoveAll = "remove_all";
constexpr std::string_view kReturnRemoveAll = "removed";
constexpr std::string_view kArgumentsInputList = "input_list";
constexpr std::string_view kArgumentsRemoveList = "to_remove_list";
constexpr std::string_view kResultRemoveAll = "removed";

constexpr std::string_view kReturnSum = "sum";
constexpr std::string_view kProcedureSum = "sum";
constexpr std::string_view kInputList = "input_list";
constexpr std::string_view kResultSum = "sum";

constexpr std::string_view kReturnUnion = "union";
constexpr std::string_view kProcedureUnion = "union";
constexpr std::string_view kArgumentsInputList1 = "input_list1";
constexpr std::string_view kArgumentsInputList2 = "input_list2";
constexpr std::string_view kResultUnion = "union";

constexpr std::string_view kReturnSort = "sorted";
constexpr std::string_view kProcedureSort = "sort";
constexpr std::string_view kResultSort = "sorted";

constexpr std::string_view kReturnCS = "contains";
constexpr std::string_view kProcedureCS = "contains_sorted";
constexpr std::string_view kArgumentInputList = "input_list";
constexpr std::string_view kArgumentElement = "element";
constexpr std::string_view kResultCS = "contains";

constexpr std::string_view kReturnMax = "max";
constexpr std::string_view kProcedureMax = "max";
constexpr std::string_view kResultMax = "max";

constexpr std::string_view kProcedureSplit = "split";
constexpr std::string_view kReturnSplit = "splitted";
constexpr std::string_view kArgumentDelimiter = "delimiter";
constexpr std::string_view kResultSplit = "splitted";

constexpr std::string_view kReturnPairs = "pairs";
constexpr std::string_view kProcedurePairs = "pairs";
constexpr std::string_view kResultPairs = "pairs";

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

void RemoveAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void Sum(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void Union(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void Sort(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void ContainsSorted(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void Max(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void Split(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void Pairs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Contains(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void UnionAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void Min(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void ToSet(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void Partition(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);


}  // namespace Collections
