#pragma once

#include <mgp.hpp>
#include <unordered_set>

namespace Algo {

/* cover constants */
constexpr std::string_view kProcedureCover = "cover";
constexpr std::string_view kCoverArg1 = "nodes";
constexpr std::string_view kCoverRet1 = "rel";

void Cover(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Algo
