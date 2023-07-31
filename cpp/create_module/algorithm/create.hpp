#pragma once

#include <mgp.hpp>
#include <string_view>

namespace Create {

/* set_rel_properties constants */
constexpr const std::string_view kProcedureSetRelProperties = "set_rel_properties";
constexpr const std::string_view kSetRelPropertiesArg1 = "relationships";
constexpr const std::string_view kSetRelPropertiesArg2 = "keys";
constexpr const std::string_view kSetRelPropertiesArg3 = "values";
constexpr const std::string_view kResultSetRelProperties = "relationship";

void SetRelProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Create
