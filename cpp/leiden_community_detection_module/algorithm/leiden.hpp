#pragma once

#include <cstdint>
#include <stack>
#include <vector>

#include <memory>

#include <mg_procedure.h>
#include <mg_exceptions.hpp>
#include <mg_graph.hpp>

namespace leiden_alg {

const double gamma = 0.25; // TODO: user should be able to set this
const double theta = 0.01; // TODO: user should be able to set this

///
///@brief A method that performs the Leiden community detection algorithm on the given graph.
///
///@param graph Graph for exploration
///@return A vector that contains community identifiers placed on indices that correspond to the identifiers of the nodes.
///
std::vector<std::vector<int>> getCommunities(const mg_graph::GraphView<> &graph);

}  // namespace leiden_alg