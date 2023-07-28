#include "create.hpp"

void Create::SetProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph = mgp::Graph(memgraph_graph);
    const mgp::Node node_input = arguments[0].ValueNode();
    const std::string string(arguments[1].ValueString());
    auto record = record_factory.NewRecord();
    mgp::Value value = arguments[2];
    const int64_t id = node_input.Id().AsInt();
    for (auto node : graph.Nodes()) {
      if (node.Id().AsInt() == id) {
        node.SetProperty(string, std::move(value));
        record.Insert(std::string(Create::kReturntSetProperty).c_str(), node);
      }
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
