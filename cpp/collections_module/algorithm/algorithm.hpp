#pragma once

#include <mgp.hpp>

namespace Collections{

constexpr const char *kResultSumLongs = "sum";

void SumLongs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}