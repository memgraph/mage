
#pragma once

#include <mgp.hpp>

namespace Util {
/*md5 constants*/
constexpr std::string_view kProcedureMd5 = "md5";
constexpr std::string_view kArgumentValuesMd5 = "values";
constexpr std::string_view kArgumentResultMd5 = "result";
constexpr std::string_view kArgumentStringToHash = "stringToHash";

void Md5Procedure(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void Md5Function(mgp_list *args, mgp_func_context *func_context, mgp_func_result *res, mgp_memory *memory);
std::string Md5(mgp::List arguments);
}  // namespace Util
