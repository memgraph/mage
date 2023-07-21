#pragma once

#include <mgp.hpp>

#include <exception>
#include <list>
#include <stdexcept>
#include <string_view>

namespace Collections {

constexpr const std::string_view kProcedureAvg = "avg";

constexpr const std::string_view kReturnAvg = "average";

constexpr const std::string_view kNumbersList = "list_of_numbers";

constexpr const char *kResultAverage = "average";

void Avg(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Collections
