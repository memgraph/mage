#pragma once

#include <mgp.hpp>

namespace Create {

constexpr std::string_view kReturnRelProp = "relationship";

constexpr std::string_view kProcedureSetRelProp = "set_rel_property";

constexpr std::string_view kArgumentsRelationship = "input_rel";
constexpr std::string_view kArgumentsKey = "input_key";
constexpr std::string_view kArgumentsValue = "input_value";

constexpr std::string_view kResultRelProp = "relationship";

void SetRelProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Create
