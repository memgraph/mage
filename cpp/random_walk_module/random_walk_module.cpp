#include <mg_utils.hpp>
#include <iostream>

void RandomWalk(mgp_list *args, mgp_graph *memgraph_graph,
                     mgp_result *result, mgp_memory *memory);

void InsertStepRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory,
                      const int step, const int node_id);

extern "C" int mgp_init_module(struct mgp_module *module,
                               struct mgp_memory *memory);

extern "C" int mgp_shutdown_module();


void RandomWalk(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result,
                mgp_memory *memory) {
  try {
    std::cout << "+" << std::endl;
    const auto start = mgp::value_get_vertex(mgp::list_at(args, 0));
    const auto n_steps = mgp::value_get_int(mgp::list_at(args, 1));

    srand(time(NULL));

    const auto start_id = mgp::vertex_get_id(start).as_int;
    const auto graph = mg_utility::GetGraphView(
        memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph);

    int step = 0;
    auto current_node = graph->GetNode(start_id);
    InsertStepRecord(memgraph_graph, result, memory, step++, current_node.id);

    while (step <= n_steps) {
      const auto neighbors = graph->Neighbours(current_node.id);
      if (neighbors.empty()) break;

      const auto next_node = neighbors[rand() % neighbors.size()];
      current_node = graph->GetNode(next_node.node_id);
      // record the output
      InsertStepRecord(memgraph_graph, result, memory, step++, current_node.id);
    }
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void InsertStepRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory,
                      const int step, const int node_id) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertIntValueResult(record, "step", step, memory);
  mg_utility::InsertNodeValueResult(graph, record, "node", node_id, memory);
}

extern "C" int mgp_init_module(struct mgp_module *module,
                               struct mgp_memory *memory) {
  {
    try {
      auto *rw_proc = mgp::module_add_read_procedure(module, "get", RandomWalk);

      // optional parameters require a default value
      auto default_steps = mgp::value_make_int(10, memory);

      mgp::proc_add_arg(rw_proc, "start", mgp::type_node());
      mgp::proc_add_opt_arg(rw_proc, "steps", mgp::type_int(), default_steps);

      mgp::proc_add_result(rw_proc, "step", mgp::type_int());
      mgp::proc_add_result(rw_proc, "node", mgp::type_node());

      mgp_value_destroy(default_steps);
    } catch (const std::exception &e) {
      return 1;
    }
  }
  return 0;
}

extern "C" int mgp_shutdown_module() {
   return 0;
}