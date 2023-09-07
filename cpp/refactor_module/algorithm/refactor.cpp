#include "refactor.hpp"

#include <mg_utils.hpp>
#include <unordered_set>

mgp::Node getCategoryNode(mgp::Graph &graph, std::unordered_set<mgp::Node> &created_nodes,
                          std::string_view new_prop_name_key, mgp::Value &new_node_name, std::string_view new_label) {
  for (auto node : created_nodes) {
    if (node.GetProperty(std::string(new_prop_name_key)) == new_node_name) {
      return node;
    }
  }
  mgp::Node category_node = graph.CreateNode();
  category_node.AddLabel(new_label);
  category_node.SetProperty(std::string(new_prop_name_key), new_node_name);
  created_nodes.insert(category_node);
  return category_node;
}

void Refactor::Categorize(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto original_prop_key = arguments[0].ValueString();
    const auto rel_type = arguments[1].ValueString();
    const auto is_outgoing = arguments[2].ValueBool();
    const auto new_label = arguments[3].ValueString();
    const auto new_prop_name_key = arguments[4].ValueString();
    const auto copy_props_list = arguments[5].ValueList();

    std::unordered_set<mgp::Node> created_nodes;

    for (auto node : graph.Nodes()) {
      auto new_node_name = node.GetProperty(std::string(original_prop_key));
      if (new_node_name.IsNull()) {
        continue;
      }

      auto category_node = getCategoryNode(graph, created_nodes, new_prop_name_key, new_node_name, new_label);

      if (is_outgoing) {
        graph.CreateRelationship(node, category_node, rel_type);
      } else {
        graph.CreateRelationship(category_node, node, rel_type);
      }

      node.RemoveProperty(std::string(original_prop_key));
      for (auto key : copy_props_list) {
        auto prop_key = std::string(key.ValueString());
        auto prop_value = node.GetProperty(prop_key);
        if (prop_value.IsNull() || prop_key == new_prop_name_key) {
          continue;
        }
        category_node.SetProperty(prop_key, prop_value);
        node.RemoveProperty(prop_key);
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultCategorize).c_str(), "success");
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

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
    std::unordered_set<mgp::Value> skip_props_searchable{skip_props.begin(), skip_props.end()};

    for (auto node : nodes) {
      mgp::Node old_node = node.ValueNode();
      mgp::Node new_node = graph.CreateNode();

      for (auto label : old_node.Labels()) {
        new_node.AddLabel(label);
      }

      for (auto prop : old_node.Properties()) {
        if (skip_props.Empty() || !skip_props_searchable.contains(mgp::Value(prop.first))) {
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
