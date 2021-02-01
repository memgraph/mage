#pragma once

#include <mg_procedure.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <vector>

namespace utils {
/// Data structure for saving both mapping ID-Index and Index-ID
template <typename T, typename U>
using PairMap = std::optional<std::pair<std::map<T, U>, std::map<U, T>>>;

struct GraphMapping {
  long numberOfVertices;
  long numberOfEdges;
  std::vector<std::pair<uint32_t, uint32_t>> edges;
};

/// Method for making Memgraph ID to iter index and vice-versa pairs for mapping
/// the Memgraph instance nodes to the algorithm's third-party library nodes.
///
/// @param graph Memgraph graph instance
/// @param result Memgraph result storage
/// @param memory Memgraph memory storage
/// @return Pair of ID to iter and iter to ID mapping
PairMap<uint32_t, uint32_t>
VertexIdMapping(const mgp_graph *graph, mgp_result *result, mgp_memory *memory);

/// Method for making Memgraph ID to iter index and vice-versa pairs for mapping
/// the Memgraph instance nodes to the algorithm's third-party library nodes.
///
/// @param graph Memgraph graph instance
/// @param result Memgraph result storage
/// @param memory Memgraph memory storage
/// @return Pair of ID to iter and iter to ID mapping
PairMap<uint32_t, uint32_t> VertexIdMappingSubgraph(const mgp_list *nodes,
                                                    const mgp_graph *graph,
                                                    mgp_result *result,
                                                    mgp_memory *memory);

std::set<int64_t> GetAdjacentVertexId(const mgp_vertex *start_vertex,
                                      mgp_memory *memory);

std::set<int64_t> FindPattern(const std::vector<std::string> &pattern,
                              const mgp_vertex *start_vertex,
                              const mgp_graph *graph, mgp_memory *memory);

GraphMapping MapMemgraphGraph(const std::map<uint32_t, uint32_t> &idToIter,
                              const mgp_graph *memgraphGraph,
                              mgp_memory *memory);

GraphMapping MapMemgraphGraphWithPatterns(
    const mgp_list *nodes, std::vector<std::vector<std::string>> patterns,
    const std::map<uint32_t, uint32_t> &idToIter,
    const mgp_graph *memgraphGraph, mgp_memory *memory);

std::vector<std::vector<std::string>>
ListToPattern(const mgp_list *edge_patterns);
} // namespace utils
