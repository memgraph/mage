#include "refactor.hpp"

#include "mgp.hpp"

void Refactor::RenameLabel(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto old_label{arguments[0].ValueString()};
    const auto new_label{arguments[1].ValueString()};
    const auto nodes{arguments[2].ValueList()};

    auto change_label = [&old_label, &new_label](mgp::Node node) {
      if (node.HasLabel(old_label)) {
        node.RemoveLabel(old_label);
        node.AddLabel(new_label);
      }
    };

    if (!nodes.Empty()) {
      for (auto node : nodes) {
        change_label(node.ValueNode());
      }
      return;
    }

    mgp::Graph graph{memgraph_graph};
    for (auto node : graph.Nodes()) {
      change_label(node);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Refactor::RenameNodeProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto old_property_name{std::string(arguments[0].ValueString())};
    const auto new_property_name{std::string(arguments[1].ValueString())};
    const auto nodes{arguments[2].ValueList()};

    auto change_property = [&old_property_name, &new_property_name](mgp::Node node) {
      auto old_property = node.GetProperty(old_property_name);
      if (old_property.IsNull()) {
        return;
      }
      node.RemoveProperty(old_property_name);
      node.SetProperty(new_property_name, old_property);
    };

    if (!nodes.Empty()) {
      for (auto node : nodes) {
        change_property(node.ValueNode());
      }
      return;
    }

    mgp::Graph graph{memgraph_graph};
    for (auto node : graph.Nodes()) {
      change_property(node);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
