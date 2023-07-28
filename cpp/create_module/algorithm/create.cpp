#include "create.hpp"

void Create::Node(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto labels = arguments[0].ValueList();
    const auto properties = arguments[1].ValueMap();

    mgp::Node node = graph.CreateNode();

    for (const auto label : labels) {
      node.AddLabel(label.ValueString());
    }

    for (const auto property : properties) {
      node.SetProperty((std::string)property.key, property.value);
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultNode).c_str(), std::move(node));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
