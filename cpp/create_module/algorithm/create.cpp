#include "create.hpp"

void Create::ProcessElement(std::unordered_set<mgp::Id> &result_set, const mgp::Value &element) {
  if (element.IsNode()) {
    result_set.insert(std::move(element.ValueNode().Id()));
  } else if (element.IsRelationship()) {
    result_set.insert(std::move(element.ValueRelationship().Id()));
  } else if (element.IsInt()) {
    result_set.insert(std::move(mgp::Id::FromInt(element.ValueInt())));
  } else {
    throw mgp::ValueException("First argument must be node, relationship, id or a list of those.");
  }
}

const std::unordered_set<mgp::Id> Create::GetIds(const mgp::Value &argument) {
  std::unordered_set<mgp::Id> result_set;
  if (argument.IsList()) {
    for (const auto element : argument.ValueList()) {
      ProcessElement(result_set, element);
    }
    return result_set;
  }
  ProcessElement(result_set, argument);
  return result_set;
}

void Create::RemoveLabels(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto nodeIds = GetIds(arguments[0]);

    const auto labels = arguments[1].ValueList();

    for (auto nodeId : nodeIds) {
      auto node = graph.GetNodeById(nodeId);
      for (auto label : labels) {
        node.RemoveLabel(label.ValueString());
        auto record = record_factory.NewRecord();
        record.Insert(std::string(kResultRemoveLabels).c_str(), std::move(node));
      }
    }
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
