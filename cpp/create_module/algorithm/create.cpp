#include "create.hpp"

void Create::SetProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
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

    const auto prop_key_list = arguments[1].ValueList();
    const auto prop_value_list = arguments[2].ValueList();

    if (prop_key_list.Size() != prop_value_list.Size()) {
      throw mgp::IndexException("Key and value lists must be the same size.");
    }

    for (auto node : graph.Nodes()) {
      if (nodeIds.contains(node.Id())) {
        for (size_t i = 0; i < prop_key_list.Size(); i++) {
          node.SetProperty((std::string)prop_key_list[i].ValueString(), prop_value_list[i]);
          auto record = record_factory.NewRecord();
          record.Insert(std::string(kResultProperties).c_str(), std::move(node));
        }
      }
    }
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
