#include <queue>
#include <unordered_map>

#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

namespace {

constexpr char const *kProcedureGet = "get";

constexpr char const *kFieldVertex = "node";
constexpr char const *kFieldComponentId = "component_id";

void InsertWeaklyComponentResult(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const int component_id,
                                 const int vertex_id) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertNodeValueResult(graph, record, kFieldVertex, vertex_id, memory);
  mg_utility::InsertIntValueResult(record, kFieldComponentId, component_id, memory);
}

/// Finds weakly connected components of a graph.
///
/// Time complexity: O(|V|+|E|)
void Weak(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph);

    std::unordered_map<std::uint64_t, std::uint64_t> vertex_component;
    std::uint64_t curr_component = 0;
    for (auto vertex : graph->Nodes()) {
      if (vertex_component.find(vertex.id) != vertex_component.end()) continue;

      // Run BFS from current vertex.
      std::queue<std::uint64_t> q;

      q.push(vertex.id);
      vertex_component[vertex.id] = curr_component;
      while (!q.empty()) {
        auto v_id = q.front();
        q.pop();

        // Iterate over inbound edges.
        for (auto neighbor : graph->Neighbours(v_id)) {
          auto next_id = neighbor.node_id;

          if (vertex_component.find(next_id) != vertex_component.end()) {
            continue;
          }
          vertex_component[next_id] = curr_component;
          q.push(next_id);
        }
      }

      ++curr_component;
    }

    for (const auto [vertex_id, component_id] : vertex_component) {
      // Insert each weakly component record
      InsertWeaklyComponentResult(memgraph_graph, result, memory, component_id, graph->GetMemgraphNodeId(vertex_id));
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    auto *wcc_proc = mgp::module_add_read_procedure(module, kProcedureGet, Weak);

    mgp::proc_add_result(wcc_proc, kFieldVertex, mgp::type_node());
    mgp::proc_add_result(wcc_proc, kFieldComponentId, mgp::type_int());
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
