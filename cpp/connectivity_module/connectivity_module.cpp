#include <queue>
#include <unordered_map>

#include <mage.hpp>
#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

namespace {

constexpr char const *kProcedureGet = "get";

constexpr char const *kFieldVertex = "node";
constexpr char const *kFieldComponentId = "component_id";

/// Finds the weakly connected components of the graph.
///
/// Time complexity: O(|V|+|E|)
void Weak(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto graph = mage::Graph(memgraph_graph, memory);
    auto graph2 = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph);

    std::unordered_map<std::int64_t, std::int64_t> vertex_component;
    std::int64_t curr_component = 0;

    for (auto vertex : graph.vertices()) {
      if (vertex_component.find(vertex.id().AsInt()) != vertex_component.end()) continue;

      // Run BFS from current vertex.
      std::queue<std::int64_t> q;

      q.push(vertex.id().AsInt());
      vertex_component[vertex.id().AsInt()] = curr_component;
      while (!q.empty()) {
        auto v_id = q.front();
        q.pop();

        // Iterate over inbound edges.
        std::vector<std::int64_t> neighbor_ids;
        for (auto out_edge : graph.GetVertexById(v_id).out_edges()) {
          auto destination = out_edge.to();
          neighbor_ids.push_back(destination.id().AsInt());
        }
        for (auto neighbor_id : neighbor_ids) {
          if (vertex_component.find(neighbor_id) != vertex_component.end()) {
            continue;
          }
          vertex_component[neighbor_id] = curr_component;
          q.push(neighbor_id);
        }
      }
      ++curr_component;
    }

    auto record_factory = mage::RecordFactory(result, memory);

    for (const auto [vertex_id, component_id] : vertex_component) {
      // Insert each weakly component record
      auto record = record_factory.NewRecord();
      record.Insert(kFieldVertex, graph.GetVertexById(vertex_id));
      record.Insert(kFieldComponentId, component_id);
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
