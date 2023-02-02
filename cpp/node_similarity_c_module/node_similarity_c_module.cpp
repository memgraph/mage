#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

const char *kProcedureGet = "get";
const char *kFieldNode1 = "node1";
const char *kFieldNode2 = "node2";
const char *kSimilarity = "similarity";
  

void Jaccard(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    auto *vertices_it = mgp::graph_iter_vertices(memgraph_graph, memory);  // Safe vertex iterator creation
    for (auto *node1 = mgp::vertex_copy(mgp::vertices_iterator_get(vertices_it), memory); node1;
        node1 = mgp::vertices_iterator_next(vertices_it)) {
        int node1_id = mgp::vertex_get_id(node1).as_int;
        auto *vertices_it2 = mgp::graph_iter_vertices(memgraph_graph, memory);
        for (auto *node2 = mgp::vertex_copy(mgp::vertices_iterator_get(vertices_it2), memory); node2;
            node2 = mgp::vertices_iterator_next(vertices_it2)) {
            auto *record = mgp::result_new_record(result);
            mg_utility::InsertNodeValueResult(memgraph_graph, record, kFieldNode1, node1_id, memory);
            mg_utility::InsertNodeValueResult(memgraph_graph, record, kFieldNode2, mgp::vertex_get_id(node2).as_int, memory);
            mg_utility::InsertDoubleValueResult(record, kSimilarity, 0.0, memory);
        }
         mgp::vertices_iterator_destroy(vertices_it2);
    }
    mgp::vertices_iterator_destroy(vertices_it);
}


  // Each module needs to define mgp_init_module function.
// Here you can register multiple procedures your module supports.
extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  try {
    {
      auto *proc = mgp::module_add_read_procedure(module, kProcedureGet, Jaccard);

      mgp::proc_add_result(proc, kFieldNode1, mgp::type_node());
      mgp::proc_add_result(proc, kFieldNode2, mgp::type_node());
      mgp::proc_add_result(proc, kSimilarity, mgp::type_float());
    }
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

// This is an optional function if you need to release any resources before the
// module is unloaded. You will probably need this if you acquired some
// resources in mgp_init_module.
extern "C" int mgp_shutdown_module() {
  // Return 0 to indicate success.
  return 0;
}
