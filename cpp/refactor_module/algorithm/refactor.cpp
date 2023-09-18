#include "refactor.hpp"

#include <mg_utils.hpp>
#include <string>
#include <unordered_set>

#include "mgp.hpp"

namespace {
void ThrowInvalidTypeException(const mgp::Value &value) {
  std::ostringstream oss;
  oss << value.Type();
  throw mgp::ValueException("Unsuppported type for this operation, received type: " + oss.str());
}
}  // namespace

void Refactor::From(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Relationship relationship{arguments[0].ValueRelationship()};
    const mgp::Node new_from{arguments[1].ValueNode()};
    mgp::Graph graph{memgraph_graph};

    graph.SetFrom(relationship, new_from);
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kFromResult).c_str(), relationship);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Refactor::To(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Relationship relationship{arguments[0].ValueRelationship()};
    const mgp::Node new_to{arguments[1].ValueNode()};
    mgp::Graph graph{memgraph_graph};

    graph.SetTo(relationship, new_to);
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kToResult).c_str(), relationship);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Refactor::RenameLabel(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto old_label{arguments[0].ValueString()};
    const auto new_label{arguments[1].ValueString()};
    const auto nodes{arguments[2].ValueList()};

    int64_t nodes_changed{0};
    for (const auto &node_value : nodes) {
      auto node = node_value.ValueNode();
      if (!node.HasLabel(old_label)) {
        continue;
      }

      node.RemoveLabel(old_label);
      node.AddLabel(new_label);
      nodes_changed++;
    }
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kRenameLabelResult).c_str(), nodes_changed);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Refactor::RenameNodeProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto old_property_name{std::string(arguments[0].ValueString())};
    const auto new_property_name{std::string(arguments[1].ValueString())};
    const auto nodes{arguments[2].ValueList()};

    int64_t nodes_changed{0};
    for (const auto &node_value : nodes) {
      auto node = node_value.ValueNode();
      auto old_property = node.GetProperty(old_property_name);
      if (old_property.IsNull()) {
        continue;
      }

      node.RemoveProperty(old_property_name);
      node.SetProperty(new_property_name, old_property);
      nodes_changed++;
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kRenameLabelResult).c_str(), nodes_changed);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Refactor::InsertCloneNodesRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const int cycle_id,
                                      const int node_id) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertIntValueResult(record, std::string(kResultClonedNodeId).c_str(), cycle_id, memory);
  mg_utility::InsertNodeValueResult(graph, record, std::string(kResultNewNode).c_str(), node_id, memory);
}

mgp::Node GetStandinOrCopy(const mgp::List &standin_nodes, const mgp::Node node,
                           const std::map<mgp::Node, mgp::Node> &old_new_node_mirror) {
  for (auto pair : standin_nodes) {
    if (!pair.IsList() || !pair.ValueList()[0].IsNode() || !pair.ValueList()[1].IsNode()) {
      throw mgp::ValueException(
          "Configuration map must consist of specific keys and values described in documentation.");
    }
    if (node == pair.ValueList()[0].ValueNode()) {
      return pair.ValueList()[1].ValueNode();
    }
  }
  try {
    return old_new_node_mirror.at(node);
  } catch (const std::out_of_range &e) {
    throw mgp::ValueException("Can't clone relationship without cloning relationship's source and/or target nodes.");
  }
}

bool CheckIfStandin(const mgp::Node &node, const mgp::List &standin_nodes) {
  for (auto pair : standin_nodes) {
    if (!pair.IsList() || !pair.ValueList()[0].IsNode()) {
      throw mgp::ValueException(
          "Configuration map must consist of specific keys and values described in documentation.");
    }
    if (node == pair.ValueList()[0].ValueNode()) {
      return true;
    }
  }
  return false;
}

void CloneNodes(const std::vector<mgp::Node> &nodes, const mgp::List &standin_nodes, mgp::Graph &graph,
                const std::unordered_set<mgp::Value> &skip_props_searchable,
                std::map<mgp::Node, mgp::Node> &old_new_node_mirror, mgp_graph *memgraph_graph, mgp_result *result,
                mgp_memory *memory) {
  for (auto node : nodes) {
    if (CheckIfStandin(node, standin_nodes)) {
      continue;
    }
    mgp::Node new_node = graph.CreateNode();

    for (auto label : node.Labels()) {
      new_node.AddLabel(label);
    }

    for (auto prop : node.Properties()) {
      if (skip_props_searchable.empty() || !skip_props_searchable.contains(mgp::Value(prop.first))) {
        new_node.SetProperty(prop.first, prop.second);
      }
    }
    old_new_node_mirror.insert({node, new_node});
    Refactor::InsertCloneNodesRecord(memgraph_graph, result, memory, node.Id().AsInt(), new_node.Id().AsInt());
  }
}

void CloneRels(const std::vector<mgp::Relationship> &rels, const mgp::List &standin_nodes, mgp::Graph &graph,
               const std::unordered_set<mgp::Value> &skip_props_searchable,
               std::map<mgp::Node, mgp::Node> &old_new_node_mirror, mgp_graph *memgraph_graph, mgp_result *result,
               mgp_memory *memory) {
  for (auto rel : rels) {
    mgp::Relationship new_relationship =
        graph.CreateRelationship(GetStandinOrCopy(standin_nodes, rel.From(), old_new_node_mirror),
                                 GetStandinOrCopy(standin_nodes, rel.To(), old_new_node_mirror), rel.Type());
    for (auto prop : rel.Properties()) {
      if (skip_props_searchable.empty() || !skip_props_searchable.contains(mgp::Value(prop.first))) {
        new_relationship.SetProperty(prop.first, prop.second);
      }
    }
  }
}

void Refactor::CloneNodesAndRels(mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory,
                                 const std::vector<mgp::Node> &nodes, const std::vector<mgp::Relationship> &rels,
                                 const mgp::Map &config_map) {
  mgp::List standin_nodes;
  mgp::List skip_props;
  if ((!config_map.At("standinNodes").IsList() && !config_map.At("standinNodes").IsNull()) ||
      (!config_map.At("skipProperties").IsList() && !config_map.At("skipProperties").IsNull())) {
    throw mgp::ValueException("Configuration map must consist of specific keys and values described in documentation.");
  }
  if (!config_map.At("standinNodes").IsNull()) {
    standin_nodes = config_map.At("standinNodes").ValueList();
  }
  if (!config_map.At("skipProperties").IsNull()) {
    skip_props = config_map.At("skipProperties").ValueList();
  }
  std::unordered_set<mgp::Value> skip_props_searchable{skip_props.begin(), skip_props.end()};

  auto graph = mgp::Graph(memgraph_graph);

  std::map<mgp::Node, mgp::Node> old_new_node_mirror;
  CloneNodes(nodes, standin_nodes, graph, skip_props_searchable, old_new_node_mirror, memgraph_graph, result, memory);
  CloneRels(rels, standin_nodes, graph, skip_props_searchable, old_new_node_mirror, memgraph_graph, result, memory);
}

void Refactor::CloneSubgraphFromPaths(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result,
                                      mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  try {
    const auto paths = arguments[0].ValueList();
    const auto config_map = arguments[1].ValueMap();

    std::unordered_set<mgp::Node> distinct_nodes;
    std::unordered_set<mgp::Relationship> distinct_relationships;
    for (auto path_value : paths) {
      auto path = path_value.ValuePath();
      for (size_t index = 0; index < path.Length(); index++) {
        distinct_nodes.insert(path.GetNodeAt(index));
        distinct_relationships.insert(path.GetRelationshipAt(index));
      }
      distinct_nodes.insert(path.GetNodeAt(path.Length()));
    }
    std::vector<mgp::Node> nodes_vector{distinct_nodes.begin(), distinct_nodes.end()};
    std::vector<mgp::Relationship> rels_vector{distinct_relationships.begin(), distinct_relationships.end()};
    CloneNodesAndRels(memgraph_graph, result, memory, nodes_vector, rels_vector, config_map);

  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void Refactor::CloneSubgraph(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  try {
    const auto nodes = arguments[0].ValueList();
    const auto rels = arguments[1].ValueList();
    const auto config_map = arguments[2].ValueMap();

    std::unordered_set<mgp::Node> distinct_nodes;
    std::unordered_set<mgp::Relationship> distinct_rels;

    for (auto node : nodes) {
      distinct_nodes.insert(node.ValueNode());
    }
    for (auto rel : rels) {
      distinct_rels.insert(rel.ValueRelationship());
    }

    if (distinct_rels.size() == 0 && distinct_nodes.size() > 0) {
      for (auto node : distinct_nodes) {
        for (auto rel : node.OutRelationships()) {
          if (distinct_nodes.contains(rel.To())) {
            distinct_rels.insert(rel);
          }
        }
      }
    }

    std::vector<mgp::Node> nodes_vector{distinct_nodes.begin(), distinct_nodes.end()};
    std::vector<mgp::Relationship> rels_vector{distinct_rels.begin(), distinct_rels.end()};

    CloneNodesAndRels(memgraph_graph, result, memory, nodes_vector, rels_vector, config_map);

  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

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
  mgp::MemoryDispatcherGuard guard{memory};
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

void Refactor::CloneNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
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

void Refactor::InvertRel(mgp::Graph &graph, mgp::Relationship &rel) {
  const auto old_from = rel.From();
  const auto old_to = rel.To();
  graph.SetFrom(rel, old_to);
  graph.SetTo(rel, old_from);
}

void Refactor::Invert(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph = mgp::Graph(memgraph_graph);
    mgp::Relationship rel = arguments[0].ValueRelationship();

    InvertRel(graph, rel);
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kReturnIdInvert).c_str(), rel.Id().AsInt());
    record.Insert(std::string(kReturnRelationshipInvert).c_str(), rel);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Refactor::TransferProperties(const mgp::Node &node, mgp::Relationship &rel) {
  for (auto &[key, value] : node.Properties()) {
    rel.SetProperty(key, value);
  }
}

void Refactor::Collapse(mgp::Graph &graph, const mgp::Node &node, const std::string &type,
                        const mgp::RecordFactory &record_factory) {
  if (node.InDegree() != 1 || node.OutDegree() != 1) {
    throw mgp::ValueException("Out and in degree of the nodes both must be 1!");
  }

  const mgp::Node from_node = (*node.InRelationships().begin()).From();
  const mgp::Node to_node = (*node.OutRelationships().begin()).To();
  if (from_node == node && to_node == node) {
    throw mgp::ValueException("Nodes with self relationships are non collapsible!");
  }
  mgp::Relationship new_rel = graph.CreateRelationship(from_node, to_node, type);
  TransferProperties(node, new_rel);

  auto record = record_factory.NewRecord();
  record.Insert(std::string(kReturnIdCollapseNode).c_str(), node.Id().AsInt());
  record.Insert(std::string(kReturnRelationshipCollapseNode).c_str(), new_rel);
  graph.DetachDeleteNode(node);
}

void Refactor::CollapseNode(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph = mgp::Graph(memgraph_graph);
    const mgp::Value input = arguments[0];
    const std::string type{arguments[1].ValueString()};

    if (!input.IsNode() && !input.IsInt() && !input.IsList()) {
      record_factory.SetErrorMessage("Input can only be node, node ID, or list of nodes/IDs");
      return;
    }

    if (input.IsNode()) {
      const mgp::Node node = input.ValueNode();
      Collapse(graph, node, type, record_factory);
    } else if (input.IsInt()) {
      const mgp::Node node = graph.GetNodeById(mgp::Id::FromInt(input.ValueInt()));
      Collapse(graph, node, type, record_factory);
    } else if (input.IsList()) {
      for (auto elem : input.ValueList()) {
        if (elem.IsNode()) {
          const mgp::Node node = elem.ValueNode();
          Collapse(graph, node, type, record_factory);
        } else if (elem.IsInt()) {
          const mgp::Node node = graph.GetNodeById(mgp::Id::FromInt(elem.ValueInt()));
          Collapse(graph, node, type, record_factory);
        } else {
          record_factory.SetErrorMessage("Elements in the list can only be Node or ID");
          return;
        }
      }
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Refactor::ExtractNode(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph{memgraph_graph};
    mgp::Value rel_or_id{arguments[0]};
    auto labels{arguments[1].ValueList()};
    auto out_type{arguments[2].ValueString()};
    auto in_type{arguments[3].ValueString()};

    auto extract = [&](mgp::Relationship relationship) {
      auto new_node = graph.CreateNode();
      for (const auto &label : labels) {
        new_node.AddLabel(label.ValueString());
      }
      graph.CreateRelationship(relationship.From(), new_node, in_type);
      graph.CreateRelationship(new_node, relationship.To(), out_type);
      graph.DeleteRelationship(relationship);

      auto record = record_factory.NewRecord();
      record.Insert(std::string(kResultExtractNode1).c_str(), new_node.Id().AsInt());
      record.Insert(std::string(kResultExtractNode2).c_str(), new_node);
      record.Insert(std::string(kResultExtractNode3).c_str(), "");
    };

    std::unordered_set<int64_t> ids;
    auto parse = [&ids, &extract](const mgp::Value &rel_or_id) {
      if (rel_or_id.IsInt()) {
        ids.insert(rel_or_id.ValueInt());
      } else if (rel_or_id.IsRelationship()) {
        extract(rel_or_id.ValueRelationship());
      } else {
        ThrowInvalidTypeException(rel_or_id);
      }
    };

    if (!rel_or_id.IsList()) {
      parse(rel_or_id);
      return;
    }

    for (const auto &list_element : rel_or_id.ValueList()) {
      parse(list_element);
    }

    if (ids.empty()) {
      return;
    }

    for (auto relationship : graph.Relationships()) {
      if (ids.contains(relationship.Id().AsInt())) {
        extract(relationship);
      }
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
