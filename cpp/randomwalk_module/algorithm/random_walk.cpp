#include <mg_utils.hpp>  // this line contains declarations of the public C API

void RandomWalk(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result,
                mgp_memory *memory);  // declaration of over our function


void InsertStepRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const int step, const int node_id);


extern "C" int mgp_init_module(
    struct mgp_module *module,
    struct mgp_memory *memory);  // by using extern, c++ compiler will not add argument/parameter type information to
                                 // the name used for linkage
// this is because C doesn't support function name overloading while C++ does
// mgp_init_modules must be used so we can register procedures and in this way they can be called from openCypher

extern "C" int mgp_shutdown_module() { return 0; }  // with this ypu can reset any global states or release some global resources

// Important thing is that exceptions shouldn't cross the module boundary
// don't allocate any global resource with memory argument - if you need to set up a global state, it is possible to do
// it in the mgp_init_module

void RandomWalk(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    const auto start = mgp::value_get_vertex(
        mgp::list_at(args, 0));  // idea is you have a list of arguments: 1. is starting node and second is length
    const auto n_steps = mgp::value_get_int(mgp::list_at(args, 1));  // number of steps

    srand(time(NULL));  // set random seed

    const auto start_id = mgp::vertex_get_id(start).as_int;  // get id of a starting vertex
    const auto graph = mg_utility::GetGraphView(
        memgraph_graph, result, memory,
        mg_graph::GraphType::kUndirectedGraph);  // do you have this functionality when writing Python code

    int step = 0;
    auto current_node = graph->GetNode(start_id);

    InsertStepRecord(memgraph_graph, result, memory, step++, current_node.id);

    while (step <= n_steps) {
      const auto neighbors = graph->Neighbours(current_node.id);
      if (neighbors.empty()) break;  // if no neighbours, hello it was nice to meet you, bye bye

      const auto next_node = neighbors[rand() % neighbors.size()];  // smart
      current_node = graph->GetNode(next_node.node_id);             // get by id

      InsertStepRecord(memgraph_graph, result, memory, step++, current_node.id);
    }

  } catch (std::exception &e) {
    mgp::result_set_error_msg(result, e.what());  // if you fail, then set an error message to a result
    return;
  }
}

void InsertStepRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const int step, const int node_id) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertIntValueResult(record, "step", step, memory);
  mg_utility::InsertNodeValueResult(graph, record, "node", node_id, memory);
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  {
    try {
      auto *rw_proc = mgp::module_add_read_procedure(module, "get", RandomWalk);

      // optional parameters require a default value
      auto default_steps = mgp::value_make_int(10, memory);

      mgp::proc_add_arg(rw_proc, "start", mgp::type_node());
      mgp::proc_add_opt_arg(rw_proc, "steps", mgp::type_int(), default_steps);

      // What are these step and node?
      mgp::proc_add_result(rw_proc, "step", mgp::type_int());
      mgp::proc_add_result(rw_proc, "node", mgp::type_node());

      mgp_value_destroy(default_steps);

    } catch (const std::exception &e) {
      return 1;
    }
  }
  return 0;
}
