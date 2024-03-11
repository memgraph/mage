#pragma once

#include <memory>
#include <vector>

#include <mg_procedure.h>
#include <mg_exceptions.hpp>
#include <mg_graph.hpp>

// This import should go before below imports since it is a dependency for the subsequent ones
#include "defs.h"
///////////////////////
#include "basic_comm.h"
#include "basic_util.h"
#include "color_comm.h"

namespace louvain_alg {

// Used for removing the ambiguity of existing graph instances
using GrappoloGraph = graph;

namespace {

/**
 * Grappolo community detection algorithm. Implementation by: https://github.com/Exa-Graph/grappolo
 * This function takes graph and calls a parallel clustering using the Louvain method as the serial template.
 * Depending on the <coloring> variable, instance of a graph algorithm with graph coloring is called. Next three
 * parameters are used for stopping the algorithm. If less than <minGraphSize> nodes are left in the graph, algorithm
 * stops. The algorithm will stop the iterations in the current phase when the gain in modularity is less than
 * <threshold> or <coloringThreshold>, depending on the algorithm type.
 *
 * @param grappoloGraph Grappolo graph instance
 * @param coloring If true, graph coloring is applied
 * @param min_graph_size Determines when multi-phase operations should stop. Execution stops when the coarsened graph
 * has collapsed the current graph to a fewer than `minGraphSize` nodes.
 * @param threshold The algorithm will stop the iterations in the current phase when the gain in modularity is less than
 * `threshold`
 * @param coloringThreshold The algorithm will stop the iterations in the current phase of coloring algorithm when the
 * gain in modularity is less than `coloringThreshold`
 * @return Vector of community indices
 */
std::vector<std::int64_t> GrappoloCommunityDetection(GrappoloGraph &grappolo_graph, mgp_graph *graph, bool coloring,
                                                     std::uint64_t min_graph_size, double threshold,
                                                     double coloring_threshold);

/**
 * Method for loading Grappolo graph from the instance of Memgraph graph.
 *
 * @param memgraph_graph Memgraph graph instance
 * @param grappolo_graph Grappolo graph instance
 */
void LoadUndirectedEdges(const mg_graph::GraphView<> &memgraph_graph, GrappoloGraph &grappolo_graph);

}  // namespace

std::vector<std::int64_t> GetCommunities(const mg_graph::GraphView<> &memgraph_graph, mgp_graph *graph, bool coloring = false,
                                         std::uint64_t min_graph_shrink = 100000, double threshold = 0.000001,
                                         double coloring_threshold = 0.01);

}  // namespace louvain_alg
