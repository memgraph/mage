#pragma once

#include <mgp.hpp>

namespace Collections {

constexpr std::string_view kReturnSort = "sorted";

constexpr std::string_view kProcedureSort = "sort";

constexpr std::string_view kArgumentsInputList = "input_list";

constexpr std::string_view kResultSort = "sorted";

void Sort(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

  
constexpr std::string_view kReturnCS = "contains";

constexpr std::string_view kProcedureCS = "contains_sorted";

constexpr std::string_view kArgumentInputList = "input_list";
constexpr std::string_view kArgumentElement = "element";

constexpr std::string_view kResultCS = "contains";

void ContainsSorted(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

  
constexpr std::string_view kReturnMax = "max";

constexpr std::string_view kProcedureMax = "max";

constexpr std::string_view kArgumentsInputList = "input_list";

constexpr std::string_view kResultMax = "max";

void Max(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

  
constexpr std::string_view kProcedureSplit = "split";

constexpr std::string_view kReturnSplit = "splitted";

constexpr std::string_view kArgumentInputList = "inputList";
constexpr std::string_view kArgumentDelimiter = "delimiter";

constexpr std::string_view kResultSplit = "splitted";

void Split(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

  
constexpr std::string_view kReturnPairs = "pairs";

constexpr std::string_view kProcedurePairs = "pairs";

constexpr std::string_view kInputList = "inputList";

constexpr std::string_view kResultPairs = "pairs";

void Pairs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
