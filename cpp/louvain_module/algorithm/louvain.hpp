#pragma once

#include <vector>

#include <mg_utils.hpp>

#include "basic_comm.h"
#include "basic_util.h"
#include "color_comm.h"
#include "defs.h"

namespace louvain_util {

std::vector<std::uint64_t> GrappoloCommunityDetection(graph *grappoloGraph, bool coloring, std::uint64_t min_graph_size,
                                                      double threshold, double coloringThreshold);

void LoadUndirectedEdges(const mg_graph::GraphView<> &memgraph_graph, graph *grappolo_graph);

}  // namespace louvain_util

namespace louvain_alg {

std::vector<std::uint64_t> GetCommunities(const mg_graph::GraphView<> &memgraph_graph, bool coloring = false,
                                          std::uint64_t min_graph_size = 100000, double threshold = 0.000001,
                                          double coloring_threshold = 0.01);

}