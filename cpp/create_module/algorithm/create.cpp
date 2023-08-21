#include "create.hpp"

void Create::SetElementProp(mgp::Relationship &element, const mgp::List &prop_key_list,
                            const mgp::List &prop_value_list, const mgp::RecordFactory &record_factory) {
  for (size_t i = 0; i < prop_key_list.Size(); i++) {
    element.SetProperty(std::string(prop_key_list[i].ValueString()), prop_value_list[i]);
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRelProp).c_str(), std::move(element));
  }
}

void Create::ProcessElement(const mgp::Value &element, const mgp::Graph graph, const mgp::List &prop_key_list,
                            const mgp::List &prop_value_list, const mgp::RecordFactory &record_factory,
                            std::unordered_set<mgp::Id> &relIds) {
  if (!(element.IsRelationship() || element.IsInt())) {
    throw mgp::ValueException("First argument must be a relationship, id or a list of those.");
  }
  if (element.IsRelationship()) {
    auto rel = element.ValueRelationship();
    SetElementProp(rel, prop_key_list, prop_value_list, record_factory);
    return;
  }
  relIds.insert(mgp::Id::FromInt(element.ValueInt()));
}

void Create::SetRelProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::List prop_key_list = mgp::List();
    prop_key_list.AppendExtend(arguments[1]);
    mgp::List prop_value_list = mgp::List();
    prop_value_list.AppendExtend(arguments[2]);

    std::unordered_set<mgp::Id> relIds;

    if (arguments[0].IsList()) {
      for (const auto element : arguments[0].ValueList()) {
        ProcessElement(element, graph, prop_key_list, prop_value_list, record_factory, relIds);
      }
    } else {
      ProcessElement(arguments[0], graph, prop_key_list, prop_value_list, record_factory, relIds);
    }
    if (relIds.empty()) {
      return;
    }
    for (auto rel : graph.Relationships()) {
      if (relIds.contains(rel.Id())) {
        SetElementProp(rel, prop_key_list, prop_value_list, record_factory);
      }
    }
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Create::RemoveElementProperties(mgp::Node &element, const mgp::List &properties_keys,
                                     const mgp::RecordFactory &record_factory) {
  for (auto key : properties_keys) {
    std::string key_str(key.ValueString());
    element.RemoveProperty(key_str);
  }
  auto record = record_factory.NewRecord();
  record.Insert(std::string(Create::kReturntRemoveProperties).c_str(), element);
}

void Create::RemoveElementLabels(mgp::Node &element, const mgp::List &labels,
                                 const mgp::RecordFactory &record_factory) {
  for (auto label : labels) {
    element.RemoveLabel(label.ValueString());
  }
  auto record = record_factory.NewRecord();
  record.Insert(std::string(kResultRemoveLabels).c_str(), std::move(element));
}

void Create::ProcessElement(const mgp::Value &element, const mgp::Graph graph, const mgp::List &list_keys,
                            const bool labels_or_props, const mgp::RecordFactory &record_factory) {
  if (!(element.IsNode() || element.IsInt())) {
    throw mgp::ValueException("First argument must be type node, id or a list of those.");
  }
  if (element.IsNode()) {
    auto node = element.ValueNode();
    if (labels_or_props == 0) {
      RemoveElementLabels(node, list_keys, record_factory);
    } else {
      RemoveElementProperties(node, list_keys, record_factory);
    }
    return;
  }
  auto node = graph.GetNodeById(mgp::Id::FromInt(element.ValueInt()));
  if (labels_or_props == 0) {
    RemoveElementLabels(node, list_keys, record_factory);
  } else {
    RemoveElementProperties(node, list_keys, record_factory);
  }
}

void Create::RemoveLabels(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto labels = arguments[1].ValueList();

    if (arguments[0].IsList()) {
      for (const auto element : arguments[0].ValueList()) {
        ProcessElement(element, graph, labels, 0, record_factory);
      }
      return;
    }
    ProcessElement(arguments[0], graph, labels, 0, record_factory);

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
    const auto list_keys = arguments[1].ValueList();

    if (arguments[0].IsList()) {
      for (const auto element : arguments[0].ValueList()) {
        ProcessElement(element, graph, list_keys, 1, record_factory);
      }
      return;
    }
    ProcessElement(arguments[0], graph, list_keys, 1, record_factory);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Create::SetElementProp(mgp::Node &element, const mgp::List &prop_key_list, const mgp::List &prop_value_list,
                            const mgp::RecordFactory &record_factory) {
  for (size_t i = 0; i < prop_key_list.Size(); i++) {
    element.SetProperty(std::string(prop_key_list[i].ValueString()), prop_value_list[i]);
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultProperties).c_str(), std::move(element));
  }
}

void Create::ProcessElement(const mgp::Value &element, const mgp::Graph graph, const mgp::List &prop_key_list,
                            const mgp::List &prop_value_list, const mgp::RecordFactory &record_factory) {
  if (element.IsNode()) {
    auto node = element.ValueNode();
    SetElementProp(node, prop_key_list, prop_value_list, record_factory);
    return;
  }
  if (element.IsInt()) {
    auto node = graph.GetNodeById(mgp::Id::FromInt(element.ValueInt()));
    SetElementProp(node, prop_key_list, prop_value_list, record_factory);
    return;
  }
  throw mgp::ValueException("First argument must be a node, id or a list of those.");
}

void Create::SetProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto prop_key_list = arguments[1].ValueList();
    const auto prop_value_list = arguments[2].ValueList();

    if (prop_key_list.Size() != prop_value_list.Size()) {
      throw mgp::IndexException("Key and value lists must be the same size.");
    }

    if (!arguments[0].IsList()) {
      ProcessElement(arguments[0], graph, prop_key_list, prop_value_list, record_factory);
      return;
    }
    for (const auto element : arguments[0].ValueList()) {
      ProcessElement(element, graph, prop_key_list, prop_value_list, record_factory);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

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
    mgp::List key;
    key.AppendExtend(arguments[1]);
    mgp::List value;
    value.AppendExtend(arguments[2]);

    if (!arguments[0].IsList()) {
      ProcessElement(arguments[0], graph, key, value, record_factory);
      return;
    }
    for (const auto element : arguments[0].ValueList()) {
      ProcessElement(element, graph, key, value, record_factory);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
