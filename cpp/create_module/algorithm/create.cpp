#include "create.hpp"
#include <unordered_set>

void Create::RemoveLabels(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    std::set<mgp::Id> nodeIds;

    if (arguments[0].IsList()) {
      for (const auto element : arguments[0].ValueList()) {
        if (element.IsNode()) {
          nodeIds.insert(std::move(element.ValueNode().Id()));
        } else if (element.IsInt()) {
          nodeIds.insert(std::move(mgp::Id::FromInt(element.ValueInt())));
        } else {
          throw mgp::ValueException("First argument must be a node, node's id or a list of those.");
        }
      }
    } else if (arguments[0].IsNode()) {
      nodeIds.insert(std::move(arguments[0].ValueNode().Id()));
    } else if (arguments[0].IsInt()) {
      nodeIds.insert(std::move(mgp::Id::FromInt(arguments[0].ValueInt())));
    } else {
      throw mgp::ValueException("First argument must be a node, node's id or a list of those.");
    }
    // this part of code is very similar so I could extract it in a function but then lots of type casting has to be
    // done so I'm not sure if it is worth it?

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
