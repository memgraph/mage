#include "nodes.hpp"


void Nodes::Link(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph = mgp::Graph(memgraph_graph);
    const mgp::List list_nodes = arguments[0].ValueList();
    const std::string_view type{arguments[1].ValueString()};
    const size_t size = list_nodes.Size();

    for(size_t i = 0; i < size -1; i++){
        graph.CreateRelationship(list_nodes[i].ValueNode(),list_nodes[i+1].ValueNode(), type);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}