#include "path.hpp"

void GetStartNodes(const mgp::Value element, const mgp::Graph &graph, std::unordered_set<mgp::Node> &start_nodes) {
  if (!(element.IsNode() || element.IsInt())) {
    throw mgp::ValueException("First argument must be type node, id or a list of those.");
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

void Path::VisitNode(const mgp::Node node, std::unordered_map<mgp::Node, std::int64_t> &visited_nodes, bool is_start,
                     const mgp::Map &config, int64_t hop_count, Path::LabelSets &labelFilterSets,
                     const mgp::RecordFactory &record_factory) {
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
    if (!is_start || config.At("minLevel").ValueInt() != 1) {  // what if they send me not int or nothing at all
      if ((labelFilterSets.end_list.empty() && labelFilterSets.termination_list.empty()) ||
          IsLabelListed(node, labelFilterSets.end_list) || IsLabelListed(node, labelFilterSets.termination_list)) {
        auto record = record_factory.NewRecord();
        record.Insert(std::string(kResultSubgraphNodes).c_str(), node);
      }
    }
  }
  visited_nodes.insert({node, hop_count});
  if (IsLabelListed(node, labelFilterSets.termination_list)) {
    return;
  }
  for (const auto in_rel : node.InRelationships()) {
    if (RelFilterAllows(config, in_rel.Type(), true)) {
      VisitNode(in_rel.From(), visited_nodes, false, config, hop_count + 1, labelFilterSets, record_factory);
    }
  }
  for (const auto out_rel : node.OutRelationships()) {
    if (RelFilterAllows(config, out_rel.Type(), false)) {
      VisitNode(out_rel.To(), visited_nodes, false, config, hop_count + 1, labelFilterSets, record_factory);
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

void SetDefaultConfig(mgp::Map &config) {
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
        "Config parameter must be a map with specific keys and values described in documentation.");
  }
}

void Path::SubgraphNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto config = arguments[1].ValueMap();
    SetDefaultConfig(config);

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

    std::unordered_map<mgp::Node, std::int64_t> visited_nodes;
    for (const auto node : start_nodes) {
      VisitNode(node, visited_nodes, true, config, 0, labelFilterSets, record_factory);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
