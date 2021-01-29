#include <mg_procedure.h>

#include <map>
#include <optional>
#include <queue>
#include <utils/mg_utils.hpp>

#include "pagerank/pagerank.hpp"

/// Memgraph query module implementation of parallel pagerank_module algorithm.
/// PageRank is the algorithm for measuring influence of connected nodes.
///
/// @param args Memgraph module arguments
/// @param memgraphGraph Memgraph graph instance
/// @param result Memgraph result storage
/// @param memory Memgraph memory storage
static void ParallelPagerank(const mgp_list *args, const mgp_graph *graph,
                             mgp_result *result, mgp_memory *memory,
                             bool pattern) {
  try {
    std::map<uint32_t, uint32_t> iterToId;
    std::map<uint32_t, uint32_t> idToIter;
    GraphMapping *graphMapping;

    if (!pattern) {
      auto idMapping = VertexIdMapping(graph, result, memory);

      if (!idMapping) {
        mgp_result_set_error_msg(result, "Not enough memory");
        return;
      }
      iterToId = idMapping->first;
      idToIter = idMapping->second;

      graphMapping = MapMemgraphGraph(idToIter, graph, memory);

    } else {
      // Get id mapping.
      const mgp_list *nodes = mgp_value_get_list(mgp_list_at(args, 0));

      if (nodes == nullptr || !mgp_list_size(nodes)) {
        throw std::runtime_error("Nodes list is empty.");
      }

      const mgp_list *edge_patterns = mgp_value_get_list(mgp_list_at(args, 1));
      if (edge_patterns == nullptr || !mgp_list_size(edge_patterns)) {
        throw std::runtime_error("Labels list is empty");
      }

      auto patterns = ListToPattern(edge_patterns);

      auto idMapping = VertexIdMappingSubgraph(nodes, graph, result, memory);

      if (!idMapping) {
        mgp_result_set_error_msg(result, "Not enough memory");
        return;
      }
      iterToId = idMapping->first;
      idToIter = idMapping->second;

      graphMapping = MapMemgraphGraphWithPatterns(nodes, patterns, idToIter,
                                                  graph, memory);
    }

    uint32_t number_of_nodes = graphMapping->numberOfVertices;
    uint32_t number_of_edges = graphMapping->numberOfEdges;
    std::vector<std::pair<uint32_t, uint32_t>> edges = graphMapping->edges;
    // Make Pagerank graph.
    pagerank::PageRankGraph pagerankGraph(number_of_nodes, number_of_edges,
                                          edges);
    // Call pagerank_module calculation.
    auto pagerankResult = pagerank::ParallelIterativePageRank(pagerankGraph);

    // Write results.
    for (uint32_t i = 0; i < number_of_nodes; ++i) {
      mgp_result_record *record = mgp_result_new_record(result);
      if (record == nullptr) {
        mgp_result_set_error_msg(result, "Not enough memory");
        return;
      }

      mgp_value *rank_value = mgp_value_make_double(pagerankResult[i], memory);
      if (rank_value == nullptr) {
        mgp_result_set_error_msg(result, "Not enough memory");
        return;
      }

      mgp_vertex *vertex = mgp_graph_get_vertex_by_id(
          graph, mgp_vertex_id{.as_int = iterToId[i]}, memory);
      mgp_value *vertex_value = mgp_value_make_vertex(vertex);
      if (vertex_value == nullptr) {
        mgp_result_set_error_msg(result, "Not enough memory");
        return;
      }

      int rank_inserted = mgp_result_record_insert(record, "rank", rank_value);
      mgp_value_destroy(rank_value);

      if (!rank_inserted) {
        mgp_result_set_error_msg(result, "Not enough memory");
        return;
      }

      int value_inserted =
          mgp_result_record_insert(record, "node", vertex_value);
      mgp_value_destroy(vertex_value);

      if (!value_inserted) {
        mgp_result_set_error_msg(result, "Not enough memory");
      }
    }

  } catch (const std::exception &e) {
    mgp_result_set_error_msg(result, e.what());
    return;
  }
}

void PagerankWrapper(const mgp_list *args, const mgp_graph *graph,
                     mgp_result *result, mgp_memory *memory) {
  ParallelPagerank(args, graph, result, memory, false);
}

void PatternPagerankWrapper(const mgp_list *args, const mgp_graph *graph,
                            mgp_result *result, mgp_memory *memory) {
  ParallelPagerank(args, graph, result, memory, true);
}

extern "C" int mgp_init_module(struct mgp_module *module,
                               struct mgp_memory *memory) {
  struct mgp_proc *pagerank_proc =
      mgp_module_add_read_procedure(module, "pagerank", PagerankWrapper);

  if (!pagerank_proc) return 1;
  if (!mgp_proc_add_result(pagerank_proc, "node", mgp_type_node())) return 1;
  if (!mgp_proc_add_result(pagerank_proc, "rank", mgp_type_float())) return 1;

  struct mgp_proc *subgraph_proc = mgp_module_add_read_procedure(
      module, "pattern_pagerank", PatternPagerankWrapper);

  if (!subgraph_proc) return 1;
  if (!mgp_proc_add_opt_arg(
          subgraph_proc, "transform_nodes", mgp_type_list(mgp_type_node()),
          mgp_value_make_list(mgp_list_make_empty(0, memory))))
    return 1;

  if (!mgp_proc_add_opt_arg(
          subgraph_proc, "transform_patterns",
          mgp_type_list(mgp_type_list(mgp_type_string())),
          mgp_value_make_list(mgp_list_make_empty(0, memory))))
    return 1;
  if (!mgp_proc_add_result(subgraph_proc, "node", mgp_type_node())) return 1;
  if (!mgp_proc_add_result(subgraph_proc, "rank", mgp_type_float())) return 1;
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
