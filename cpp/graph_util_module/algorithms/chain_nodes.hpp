#include <mgp.hpp>

const char *kResultChainNodes = "connections";

void ChainNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};
    const auto arguments = mgp::List(args);
    auto graph = mgp::Graph(memgraph_graph);
    auto factory = mgp::RecordFactory(result);

    const auto list_of_nodes = arguments[0].ValueList();
    const auto edge_type = arguments[1].ValueString();

    auto connections = mgp::List();

    const auto list_size = list_of_nodes.Size();
    for (auto i = 0; i < list_size - 1; i++) {
      const auto begin_node = list_of_nodes[i].ValueNode();
      const auto end_node = list_of_nodes[i+1].ValueNode();

      const auto relationship = graph.CreateRelationship(begin_node, end_node, edge_type);
      connections.AppendExtend(mgp::Value(relationship));
    }

    auto result = factory.NewRecord();
    result.Insert(kResultChainNodes, connections);

  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
