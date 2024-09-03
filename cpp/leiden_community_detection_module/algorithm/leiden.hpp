#pragma once

#include <cstdint>
#include <stack>
#include <vector>

#include <memory>
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
std::vector<std::int64_t> GetCommunities(const mg_graph::GraphView<> &graph);

}  // namespace leiden_alg