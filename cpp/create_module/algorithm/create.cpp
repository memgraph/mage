#include "create.hpp"

void Create::Nodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph = mgp::Graph(memgraph_graph);
    const mgp::List labels = arguments[0].ValueList();
    const mgp::List properties = arguments[1].ValueList();
    const int64_t num_of_nodes = properties.Size();
    for (auto i = 0; i < num_of_nodes; i++) {
      mgp::Node node = graph.CreateNode();
      for (auto label : labels) {
        node.AddLabel(label.ValueString());
      }
      const mgp::Map node_properties = properties[i].ValueMap();
      for (auto item : node_properties) {
        node.SetProperty(std::string(item.key), std::move(item.value));
      }
      auto record = record_factory.NewRecord();
      record.Insert(std::string(Create::kReturnNodes).c_str(), node);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
