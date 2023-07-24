#pragma once

#include <mgp.hpp>
#include <string_view>

namespace Collections {

constexpr const std::string_view kProcedureIntersection = "intersection";

constexpr const std::string_view kReturnIntersection = "list_of_any";

constexpr const std::string_view kAnyList = "list_of_any";

constexpr const std::string_view kResultIntersection = "intersection";

void Intersection(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
