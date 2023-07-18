#pragma once

#include <mgp.hpp>

#include <exception>
#include <list>
#include <stdexcept>

namespace Collections{

constexpr const char *kResultAverage = "average";

void Avg(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}