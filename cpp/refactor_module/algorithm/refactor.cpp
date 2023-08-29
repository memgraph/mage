#include "refactor.hpp"

#include <mg_utils.hpp>
#include <unordered_set>

void Refactor::InsertCloneNodesRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const int cycle_id,
                                      const int node_id) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertIntValueResult(record, std::string(kResultClonedNodeId).c_str(), cycle_id, memory);
  mg_utility::InsertNodeValueResult(graph, record, std::string(kResultNewNode).c_str(), node_id, memory);
}

void Refactor::CloneNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  auto graph = mgp::Graph(memgraph_graph);
  try {
    const auto nodes = arguments[0].ValueList();
    const auto clone_rels = arguments[1].ValueBool();
    const auto skip_props = arguments[2].ValueList();
    std::unordered_set<std::string_view> skip_props_searchable;
    for (auto element : skip_props) {
      skip_props_searchable.insert(element.ValueString());
    }

    for (auto node : nodes) {
      mgp::Node old_node = node.ValueNode();
      mgp::Node new_node = graph.CreateNode();

      for (auto label : old_node.Labels()) {
        new_node.AddLabel(label);
      }

      for (auto prop : old_node.Properties()) {
        if (skip_props.Empty() || !skip_props_searchable.contains(prop.first)) {
          new_node.SetProperty(prop.first, prop.second);
        }
      }

      if (clone_rels) {
        for (auto rel : old_node.InRelationships()) {
          graph.CreateRelationship(rel.From(), new_node, rel.Type());
        }
        for (auto rel : old_node.OutRelationships()) {
          graph.CreateRelationship(new_node, rel.To(), rel.Type());
        }
      }
      InsertCloneNodesRecord(memgraph_graph, result, memory, old_node.Id().AsInt(), new_node.Id().AsInt());
    }
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
