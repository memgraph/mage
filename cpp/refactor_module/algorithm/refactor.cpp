#include "refactor.hpp"

#include <mg_utils.hpp>
#include <unordered_set>

void Refactor::InsertCloneNodesRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const int cycle_id,
                                      const int node_id) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertIntValueResult(record, std::string(kResultClonedNodeId).c_str(), cycle_id, memory);
  mg_utility::InsertNodeValueResult(graph, record, std::string(kResultNewNode).c_str(), node_id, memory);
}

mgp::Node GetStandinOrCopy(mgp::List &standinNodes, mgp::Node node,
                           std::map<mgp::Node, mgp::Node> &old_new_node_mirror) {
  for (size_t i = 0; i < standinNodes.Size(); i += 2) {  // what if its not pairs
    if (node == standinNodes[i].ValueNode()) {
      return standinNodes[i + 1].ValueNode();
    }
  }
  return old_new_node_mirror.at(node);  // what if they send me wrong path
}

bool CheckIfStandin(mgp::Node node, mgp::List standinNodes) {
  for (size_t i = 0; i < standinNodes.Size(); i += 2) {
    if (node == standinNodes[i].ValueNode()) {
      return true;
    }
  }
  return false;
}

void Refactor::CloneNodesAndRels(mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory,
                                 const std::vector<mgp::Node> &nodes, const std::vector<mgp::Relationship> &rels,
                                 const mgp::Map &config_map) {
  mgp::List standinNodes;
  mgp::List skip_props;
  //   if (!config_map.At("standinNodes").IsList() || !config_map.At("skipProperties").IsList()) {
  //     throw mgp::ValueException("Configuration map must consist of specific keys and values described in
  //     documentation.");
  //   }
  if (!config_map.At("standinNodes").IsNull()) {
    standinNodes = config_map.At("standinNodes").ValueList();
  }
  if (!config_map.At("skipProperties").IsNull()) {
    skip_props = config_map.At("skipProperties").ValueList();
  }
  std::unordered_set<mgp::Value> skip_props_searchable{skip_props.begin(), skip_props.end()};

  auto graph = mgp::Graph(memgraph_graph);

  std::map<mgp::Node, mgp::Node> old_new_node_mirror;
  for (auto node : nodes) {
    if (CheckIfStandin(node, standinNodes)) {
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
        graph.CreateRelationship(GetStandinOrCopy(standinNodes, rel.From(), old_new_node_mirror),
                                 GetStandinOrCopy(standinNodes, rel.To(), old_new_node_mirror), rel.Type());
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