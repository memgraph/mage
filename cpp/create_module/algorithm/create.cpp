#include "create.hpp"

void Create::RemoveElementLabels(mgp::Node &element, const mgp::List &labels,
                                 const mgp::RecordFactory &record_factory) {
  for (auto label : labels) {
    element.RemoveLabel(label.ValueString());
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRemoveLabels).c_str(), std::move(element));
  }
}

void Create::ProcessElement(const mgp::Value &element, const mgp::Graph graph, const mgp::List &labels,
                            const mgp::RecordFactory &record_factory) {
  if (!(element.IsNode() || element.IsInt())) {
    throw mgp::ValueException("First argument must be type node, id or a list of those.");
  }
  if (element.IsNode()) {
    auto node = element.ValueNode();
    RemoveElementLabels(node, labels, record_factory);
    return;
  }
  auto node = graph.GetNodeById(mgp::Id::FromInt(element.ValueInt()));
  RemoveElementLabels(node, labels, record_factory);
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
        ProcessElement(element, graph, labels, record_factory);
      }
      return;
    }
    ProcessElement(arguments[0], graph, labels, record_factory);

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
