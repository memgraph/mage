#include "refactor.hpp"

#include <mg_utils.hpp>
#include <unordered_set>

void Refactor::InsertCloneNodesRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const int cycle_id,
                                      const int node_id) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertIntValueResult(record, std::string(kResultClonedNodeId).c_str(), cycle_id, memory);
  mg_utility::InsertNodeValueResult(graph, record, std::string(kResultNewNode).c_str(), node_id, memory);
}

mgp::Node GetStandinOrCopy(mgp::List &standin_nodes, mgp::Node node,
                           std::map<mgp::Node, mgp::Node> &old_new_node_mirror) {
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
    return old_new_node_mirror.at(node);  // what if they send me wrong path
  } catch (const std::out_of_range &e) {
    throw mgp::ValueException("Can't clone relationship without cloning relationship's source and/or target nodes.");
  }
}

bool CheckIfStandin(mgp::Node node, mgp::List standin_nodes) {
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
  for (auto node : nodes) {
    if (CheckIfStandin(node, standin_nodes)) {
      continue;
    }
    mgp::Node new_node = graph.CreateNode();

    for (auto label : node.Labels()) {
      new_node.AddLabel(label);
    }

    for (auto prop : node.Properties()) {
      if (skip_props.Empty() || !skip_props_searchable.contains(mgp::Value(prop.first))) {
        new_node.SetProperty(prop.first, prop.second);
      }
    }
    old_new_node_mirror.insert({node, new_node});
    InsertCloneNodesRecord(memgraph_graph, result, memory, node.Id().AsInt(), new_node.Id().AsInt());
  }

  for (auto rel : rels) {
    mgp::Relationship new_relationship =
        graph.CreateRelationship(GetStandinOrCopy(standin_nodes, rel.From(), old_new_node_mirror),
                                 GetStandinOrCopy(standin_nodes, rel.To(), old_new_node_mirror), rel.Type());
    for (auto prop : rel.Properties()) {
      if (skip_props.Empty() || !skip_props_searchable.contains(mgp::Value(prop.first))) {
        new_relationship.SetProperty(prop.first, prop.second);
      }
    }
  }
}

void Refactor::CloneSubgraphFromPaths(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result,
                                      mgp_memory *memory) {
  mgp::memory = memory;
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
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  try {
    const auto nodes = arguments[0].ValueList();
    const auto rels = arguments[1].ValueList();
    const auto config_map = arguments[2].ValueMap();

    std::unordered_set<mgp::Node> distinct_nodes;
    std::unordered_set<mgp::Relationship> distinct_rels;

    for (auto node : nodes) {
      if (node.IsNode()) {
        distinct_nodes.insert(node.ValueNode());
      }
    }
    for (auto rel : rels) {
      if (rel.IsRelationship()) {
        distinct_rels.insert(rel.ValueRelationship());
      }
    }

    if (distinct_rels.size() == 0 && distinct_nodes.size() > 1) {
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
