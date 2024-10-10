#pragma once

#include <cstdint>
#include <vector>


#include <mg_procedure.h>
#include <mg_exceptions.hpp>
#include <mg_graph.hpp>

namespace leiden_alg {
///
///@brief A method that performs the Leiden community detection algorithm on the given graph.
///
///@param graph Graph for exploration
///@return A vector that contains community identifiers placed on indices that correspond to the identifiers of the nodes.
///
std::vector<std::vector<std::uint64_t>> getCommunities(const mg_graph::GraphView<> &graph, double gamma, double theta, double resolution_parameter);

}  // namespace leiden_alg