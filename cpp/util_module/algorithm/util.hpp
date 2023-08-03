
#pragma once

#include <mgp.hpp>

namespace Util {
    /*md5 constants*/
    constexpr std::string_view kProcedureMd5 = "md5";
    constexpr std::string_view kArgumentValuesMd5 = "values";
    constexpr std::string_view kArgumentResultMd5= "result";

    void Md5(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Create

