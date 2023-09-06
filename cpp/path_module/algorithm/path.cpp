#include "path.hpp"

void GetStartNodes(const mgp::Value element, const mgp::Graph &graph, std::unordered_set<mgp::Node> &start_nodes) {
  if (!(element.IsNode() || element.IsInt())) {
    throw mgp::ValueException("The first argument needs to be a node, an integer ID, or a list thereof.");
  }
  if (element.IsNode()) {
    start_nodes.insert(element.ValueNode());
    return;
  }
  start_nodes.insert(graph.GetNodeById(mgp::Id::FromInt(element.ValueInt())));
}

bool RelFilterAllows(const mgp::Map &config, std::string_view type, bool ingoing) {
  mgp::List list_of_types = config.At("relationshipFilter").ValueList();
  if (list_of_types.Size() == 0) {
    return true;
  }
  for (const auto element : list_of_types) {
    auto string_rel_type = element.ValueString();
    if (string_rel_type.front() == '<' && ingoing) {
      string_rel_type.remove_prefix(1);
    }
    if (string_rel_type.back() == '>' && !ingoing) {
      string_rel_type.remove_suffix(1);
    }
    if (string_rel_type == type || string_rel_type.size() == 0) {
      return true;
    }
  }
  return false;
}

bool IsLabelListed(const mgp::Node node, std::unordered_set<std::string_view> &set) {
  for (const auto label : node.Labels()) {
    if (set.contains(label)) {
      return true;
    }
  }
  return false;
}

void Path::VisitNode(const mgp::Node node, std::map<mgp::Node, std::int64_t> &visited_nodes, bool is_start,
                     const mgp::Map &config, int64_t hop_count, Path::LabelSets &labelFilterSets,
                     mgp::List &to_be_returned_nodes) {
  if (config.At("maxLevel").ValueInt() != -1 && hop_count > config.At("maxLevel").ValueInt()) {
    return;
  }
  if (config.At("filterStartNode").ValueBool() || !is_start) {
    if ((IsLabelListed(node, labelFilterSets.blacklist)) ||
        (!labelFilterSets.whitelist.empty() && !IsLabelListed(node, labelFilterSets.whitelist) &&
         !IsLabelListed(node, labelFilterSets.end_list) && !IsLabelListed(node, labelFilterSets.termination_list))) {
      return;
    }
  }
  try {
    if (visited_nodes.at(node) <= hop_count) {
      return;
    }
  } catch (const std::out_of_range &e) {
    // it's okay, the node is not in visited nodes map
    if (!is_start || config.At("minLevel").ValueInt() != 1) {
      if ((labelFilterSets.end_list.empty() && labelFilterSets.termination_list.empty()) ||
          IsLabelListed(node, labelFilterSets.end_list) || IsLabelListed(node, labelFilterSets.termination_list)) {
        to_be_returned_nodes.AppendExtend(mgp::Value(node));
      }
    }
  }
  visited_nodes.insert({node, hop_count});
  if (IsLabelListed(node, labelFilterSets.termination_list)) {
    return;
  }
  for (const auto in_rel : node.InRelationships()) {
    if (RelFilterAllows(config, in_rel.Type(), true)) {
      VisitNode(in_rel.From(), visited_nodes, false, config, hop_count + 1, labelFilterSets, to_be_returned_nodes);
    }
  }
  for (const auto out_rel : node.OutRelationships()) {
    if (RelFilterAllows(config, out_rel.Type(), false)) {
      VisitNode(out_rel.To(), visited_nodes, false, config, hop_count + 1, labelFilterSets, to_be_returned_nodes);
    }
  }
}

void Path::ParseLabels(const mgp::List &list_of_labels, LabelSets &labelSets) {
  for (const auto label : list_of_labels) {
    std::string_view label_string = label.ValueString();
    const char first_elem = label_string.front();
    switch (first_elem) {
      case '-':
        label_string.remove_prefix(1);
        labelSets.blacklist.insert(label_string);
        break;
      case '>':
        label_string.remove_prefix(1);
        labelSets.end_list.insert(label_string);
        break;
      case '+':
        label_string.remove_prefix(1);
        labelSets.whitelist.insert(label_string);
        break;
      case '/':
        label_string.remove_prefix(1);
        labelSets.termination_list.insert(label_string);
        break;
      default:
        labelSets.whitelist.insert(label_string);
        break;
    }
  }
}

void SetConfig(mgp::Map &config) {
  auto value = config.At("maxLevel");
  if (value.IsNull()) {
    config.Insert("maxLevel", mgp::Value(int64_t(-1)));
  }
  value = config.At("relationshipFilter");
  if (value.IsNull()) {
    config.Insert("relationshipFilter", mgp::Value(mgp::List()));
  }
  value = config.At("labelFilter");
  if (value.IsNull()) {
    config.Insert("labelFilter", mgp::Value(mgp::List()));
  }
  value = config.At("filterStartNode");
  if (value.IsNull()) {
    config.Insert("filterStartNode", mgp::Value(false));
  }
  value = config.At("minLevel");
  if (value.IsNull()) {
    config.Insert("minLevel", mgp::Value(int64_t(0)));
  }
  if (!(config.At("minLevel").IsInt() && config.At("maxLevel").IsInt() && config.At("relationshipFilter").IsList() &&
        config.At("labelFilter").IsList() && config.At("filterStartNode").IsBool()) ||
      (config.At("minLevel").ValueInt() != 0 && config.At("minLevel").ValueInt() != 1)) {
    throw mgp::ValueException(
        "The config parameter needs to be a map with keys and values in line with the documentation.");
  }
}

void Path::SubgraphNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto config = arguments[1].ValueMap();
    SetConfig(config);

    std::unordered_set<mgp::Node> start_nodes;
    if (arguments[0].IsList()) {
      for (const auto element : arguments[0].ValueList()) {
        GetStartNodes(element, graph, start_nodes);
      }
    } else {
      GetStartNodes(arguments[0], graph, start_nodes);
    }

    LabelSets labelFilterSets;
    ParseLabels(config.At("labelFilter").ValueList(), labelFilterSets);

    std::map<mgp::Node, std::int64_t> visited_nodes;
    mgp::List to_be_returned_nodes;
    for (const auto node : start_nodes) {
      VisitNode(node, visited_nodes, true, config, 0, labelFilterSets, to_be_returned_nodes);
    }

    for (auto node : to_be_returned_nodes) {
      auto record = record_factory.NewRecord();
      record.Insert(std::string(kResultSubgraphNodes).c_str(), node);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Path::SubgraphAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto config = arguments[1].ValueMap();
    SetConfig(config);

    std::unordered_set<mgp::Node> start_nodes;
    if (arguments[0].IsList()) {
      for (const auto element : arguments[0].ValueList()) {
        GetStartNodes(element, graph, start_nodes);
      }
    } else {
      GetStartNodes(arguments[0], graph, start_nodes);
    }

    LabelSets labelFilterSets;
    ParseLabels(config.At("labelFilter").ValueList(), labelFilterSets);

    std::map<mgp::Node, std::int64_t> visited_nodes;
    mgp::List to_be_returned_nodes;
    for (const auto node : start_nodes) {
      VisitNode(node, visited_nodes, true, config, 0, labelFilterSets, to_be_returned_nodes);
    }

    std::unordered_set<mgp::Node> to_be_returned_nodes_searchable;
    for (auto node : to_be_returned_nodes) {
      to_be_returned_nodes_searchable.insert(node.ValueNode());
    }

    mgp::List to_be_returned_rels;
    for (auto node : to_be_returned_nodes) {
      for (auto rel : node.ValueNode().OutRelationships()) {
        if (to_be_returned_nodes_searchable.contains(rel.To())) {
          to_be_returned_rels.AppendExtend(mgp::Value(rel));
        }
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultNodesSubgraphAll).c_str(), to_be_returned_nodes);
    record.Insert(std::string(kResultRelsSubgraphAll).c_str(), to_be_returned_rels);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
