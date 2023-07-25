#pragma once

#include <mgp.hpp>
#include <string_view>

namespace Collections {

constexpr const std::string_view kProcedureContainsAll = "containsAll";

constexpr const std::string_view kReturnContainsAll = "contained";

constexpr const std::string_view kAnyList = "list_of_any";

constexpr const std::string_view kResultContainsAll = "contained";

void ContainsAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
