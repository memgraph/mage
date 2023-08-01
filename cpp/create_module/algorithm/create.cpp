#include "create.hpp"

const std::unordered_set<mgp::Id> Create::GetIds(const mgp::Value &argument) {
  std::unordered_set<mgp::Id> result_set;
  if (argument.IsList()) {
    for (const auto element : argument.ValueList()) {
      if (element.IsNode()) {
        result_set.insert(std::move(element.ValueNode().Id()));
      } else if (element.IsInt()) {
        result_set.insert(std::move(mgp::Id::FromInt(element.ValueInt())));
      } else {
        throw mgp::ValueException("First argument must be a node, node's id or a list of those.");
      }
    }
  } else if (argument.IsNode()) {
    result_set.insert(std::move(argument.ValueNode().Id()));
  } else if (argument.IsInt()) {
    result_set.insert(std::move(mgp::Id::FromInt(argument.ValueInt())));
  } else {
    throw mgp::ValueException("First argument must be a node, node's id or a list of those.");
  }
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
