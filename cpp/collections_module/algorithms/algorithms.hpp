#pragma once

#include <mgp.hpp>

namespace Collections {

constexpr std::string_view kProcedureSplit = "split";

constexpr std::string_view kReturnSplit = "splitted";

constexpr std::string_view kArgumentInputList = "inputList";
constexpr std::string_view kArgumentDelimiter = "delimiter";

constexpr std::string_view kResultSplit = "splitted";

void Split(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
