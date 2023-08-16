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
void Create::StartSetProperty(mgp::Node &node, const std::string &string, mgp::Value &value,
                              const mgp::RecordFactory &record_factory) {
  node.SetProperty(string, value);
  auto record = record_factory.NewRecord();
  record.Insert(std::string(Create::kReturntSetProperty).c_str(), node);
}
void Create::SetProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph = mgp::Graph(memgraph_graph);
    mgp::Value start_value = arguments[0];
    const std::string string(arguments[1].ValueString());
    mgp::Value value = arguments[2];
    if (start_value.IsNode()) {
      mgp::Node node = start_value.ValueNode();
      StartSetProperty(node, string, value, record_factory);

    } else if (start_value.IsInt()) {
      mgp::Node node = graph.GetNodeById(mgp::Id::FromInt(start_value.ValueInt()));
      StartSetProperty(node, string, value, record_factory);

    } else if (start_value.IsList()) {
      for (auto val : start_value.ValueList()) {
        if (val.IsNode()) {
          mgp::Node node = val.ValueNode();
          StartSetProperty(node, string, value, record_factory);

        } else if (val.IsInt()) {
          mgp::Node node = graph.GetNodeById(mgp::Id::FromInt(val.ValueInt()));
          StartSetProperty(node, string, value, record_factory);
        } else {
          throw mgp::ValueException("All elements of the list must be nodes or ID's");
        }
      }
    } else {
      throw mgp::ValueException("Input argument must be either node, ID or list of nodes and ID's");
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
void Create::StartRemoveProperties(mgp::Node &graph_node, const mgp::List &list_keys,
                                   const mgp::RecordFactory &record_factory) {
  for (auto key : list_keys) {
    std::string key_str(key.ValueString());
    graph_node.RemoveProperty(key_str);
  }
  auto record = record_factory.NewRecord();
  record.Insert(std::string(Create::kReturntRemoveProperties).c_str(), graph_node);
}

void Create::RemoveProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph = mgp::Graph(memgraph_graph);
    mgp::Value start_value = arguments[0];
    const mgp::List list_keys = arguments[1].ValueList();

    if (start_value.IsNode()) {
      mgp::Node graph_node = start_value.ValueNode();
      StartRemoveProperties(graph_node, list_keys, record_factory);

    } else if (start_value.IsInt()) {
      mgp::Node graph_node = graph.GetNodeById(mgp::Id::FromInt(start_value.ValueInt()));
      StartRemoveProperties(graph_node, list_keys, record_factory);

    } else if (start_value.IsList()) {
      for (auto val : start_value.ValueList()) {
        if (val.IsNode()) {
          mgp::Node graph_node = val.ValueNode();
          StartRemoveProperties(graph_node, list_keys, record_factory);

        } else if (val.IsInt()) {
          mgp::Node graph_node = graph.GetNodeById(mgp::Id::FromInt(val.ValueInt()));
          StartRemoveProperties(graph_node, list_keys, record_factory);

        } else {
          throw mgp::ValueException("All elements of the list must be nodes or ID's");
        }
      }
    } else {
      throw mgp::ValueException("Input argument must be either node, ID or list of nodes and ID's");
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
