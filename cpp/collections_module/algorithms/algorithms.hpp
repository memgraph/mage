#pragma once

#include <mgp.hpp>

namespace Collections {

constexpr std::string_view kProcedureRemoveAll = "remove_all";

constexpr std::string_view kReturnRemoveAll = "removed";

constexpr std::string_view kArgumentsInputList = "input_list";
constexpr std::string_view kArgumentsRemoveList = "to_remove_list";

constexpr std::string_view kResultRemoveAll = "removed";

void RemoveAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
