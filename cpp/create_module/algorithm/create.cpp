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

void Create::SetProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph = mgp::Graph(memgraph_graph);
    const mgp::Node node_input = arguments[0].ValueNode();
    const std::string string(arguments[1].ValueString());
    mgp::Value value = arguments[2];
    const mgp::Id id = node_input.Id();
    mgp::Node node = graph.GetNodeById(id);
    node.SetProperty(string, std::move(value));
    auto record = record_factory.NewRecord();
    record.Insert(std::string(Create::kReturntSetProperty).c_str(), node);
     
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}


void Create::RemoveProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph = mgp::Graph(memgraph_graph);
    const mgp::Node node = arguments[0].ValueNode();
    const mgp::List list_keys = arguments[1].ValueList();
    const mgp::Id node_id = node.Id();

    mgp::Node graph_node = graph.GetNodeById(node_id);
    for (auto key : list_keys) {
      std::string key_str(key.ValueString());
      graph_node.RemoveProperty(key_str);
    }
    auto record = record_factory.NewRecord();
    record.Insert(std::string(Create::kReturntRemoveProperties).c_str(), graph_node);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
