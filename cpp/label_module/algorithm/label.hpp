#pragma once

#include <mgp.hpp>

namespace Label {

/* exists constants */
constexpr std::string_view kFunctionExists = "exists";
constexpr std::string_view kArgumentsNode = "node";
constexpr std::string_view kArgumentsLabel = "label";

void Exists(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory);

}  // namespace Label
