#include <queue>
#include <unordered_map>

#include <mg_procedure.h>
#include <utils/on_scope_exit.hpp>

/// Finds weakly connected components of a graph.
///
/// Time complexity: O(|V|+|E|)
static void weak(const mgp_list *args, const mgp_graph *graph,
                 mgp_result *result, mgp_memory *memory) {
  std::unordered_map<int64_t, int64_t> vertex_component;
  mgp_vertices_iterator *vertices_iterator =
      mgp_graph_iter_vertices(graph, memory);
  if (vertices_iterator == nullptr) {
    mgp_result_set_error_msg(result, "Unable to create vertices iterator");
    return;
  }
  utils::OnScopeExit vertex_iterator_deleter([&vertices_iterator] {
    if (vertices_iterator != nullptr) {
      mgp_vertices_iterator_destroy(vertices_iterator);
    }
  });

  int64_t curr_component = 0;
  for (const mgp_vertex *vertex = mgp_vertices_iterator_get(vertices_iterator);
       vertex != nullptr;
       vertex = mgp_vertices_iterator_next(vertices_iterator)) {
    mgp_vertex_id vertex_id = mgp_vertex_get_id(vertex);
    if (vertex_component.find(vertex_id.as_int) != vertex_component.end())
      continue;

    // Run BFS from current vertex.
    std::queue<int64_t> q;
    q.push(vertex_id.as_int);
    vertex_component[vertex_id.as_int] = curr_component;
    while (!q.empty()) {
      mgp_vertex *v = mgp_graph_get_vertex_by_id(graph, {q.front()}, memory);
      if (v == nullptr) {
        mgp_result_set_error_msg(result, "Unable to get vertex");
        return;
      }
      utils::OnScopeExit v_deleter([&v] {
        if (v != nullptr) {
          mgp_vertex_destroy(v);
        }
      });

      q.pop();

      // Iterate over inbound edges.
      mgp_edges_iterator *edges_iterator = mgp_vertex_iter_in_edges(v, memory);
      if (edges_iterator == nullptr) {
        mgp_result_set_error_msg(result, "Unable to create edges iterator");
        return;
      }
      for (const mgp_edge *edge = mgp_edges_iterator_get(edges_iterator);
           edge != nullptr; edge = mgp_edges_iterator_next(edges_iterator)) {
        mgp_vertex_id next_id = mgp_vertex_get_id(mgp_edge_get_from(edge));
        if (vertex_component.find(next_id.as_int) != vertex_component.end())
          continue;
        vertex_component[next_id.as_int] = curr_component;
        q.push(next_id.as_int);
      }

      // Iterate over outbound edges.
      mgp_edges_iterator_destroy(edges_iterator);
      edges_iterator = mgp_vertex_iter_out_edges(v, memory);
      if (edges_iterator == nullptr) {
        mgp_result_set_error_msg(result, "Unable to create edges iterator");
        return;
      }
      for (const mgp_edge *edge = mgp_edges_iterator_get(edges_iterator);
           edge != nullptr; edge = mgp_edges_iterator_next(edges_iterator)) {
        mgp_vertex_id next_id = mgp_vertex_get_id(mgp_edge_get_to(edge));
        if (vertex_component.find(next_id.as_int) != vertex_component.end())
          continue;
        vertex_component[next_id.as_int] = curr_component;
        q.push(next_id.as_int);
      }

      mgp_edges_iterator_destroy(edges_iterator);
    }

    ++curr_component;
  }

  for (const auto &p : vertex_component) {
    mgp_result_record *record = mgp_result_new_record(result);
    if (record == nullptr) {
      mgp_result_set_error_msg(result, "Unable to allocate record");
      return;
    }

    mgp_value *id_value = mgp_value_make_int(p.first, memory);
    if (id_value == nullptr) {
      mgp_result_set_error_msg(result, "Unable to allocated id");
      return;
    }
    utils::OnScopeExit id_value_deleter([&id_value] {
      if (id_value != nullptr) {
        mgp_value_destroy(id_value);
      }
    });

    mgp_value *comp_value = mgp_value_make_int(p.second, memory);
    if (comp_value == nullptr) {
      mgp_result_set_error_msg(result, "Unable to allocate value");
      return;
    }
    utils::OnScopeExit comp_value_deleter([&comp_value] {
      if (comp_value != nullptr) {
        mgp_value_destroy(comp_value);
      }
    });

    int mem_id_inserted = mgp_result_record_insert(record, "id", id_value);
    int comp_inserted =
        mgp_result_record_insert(record, "component", comp_value);
    if (!mem_id_inserted || !comp_inserted) {
      mgp_result_set_error_msg(result,
                               "Unable to insert id and value to record");
      return;
    }
  }
}

extern "C" int mgp_init_module(struct mgp_module *module,
                               struct mgp_memory *memory) {
  struct mgp_proc *wcc_proc =
      mgp_module_add_read_procedure(module, "weak", weak);
  if (!mgp_proc_add_result(wcc_proc, "id", mgp_type_int())) return 1;
  if (!mgp_proc_add_result(wcc_proc, "component", mgp_type_int())) return 1;
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
