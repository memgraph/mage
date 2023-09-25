#include "path.hpp"

#include <algorithm>
#include <unordered_set>

#include "mgp.hpp"

Path::PathHelper::PathHelper(const mgp::List &labels, const mgp::List &relationships, int64_t min_hops,
                             int64_t max_hops) {
  ParseLabels(labels);
  FilterLabelBoolStatus();
  ParseRelationships(relationships);
  config_.min_hops = min_hops;
  config_.max_hops = max_hops;
}

Path::RelDirection Path::PathHelper::GetDirection(std::string &rel_type) {
  auto it = config_.relationship_sets.find(rel_type);
  if (it == config_.relationship_sets.end()) {
    return RelDirection::kNone;
  }
  return it->second;
}

bool Path::PathHelper::PathSizeOk(const int64_t path_size) const {
  return (path_size <= config_.max_hops) && (path_size >= config_.min_hops);
}

bool Path::PathHelper::PathTooBig(const int64_t path_size) const { return path_size > config_.max_hops; }

bool Path::PathHelper::Whitelisted(const bool whitelisted) const {
  return (config_.label_bools_status.whitelist_empty || whitelisted);
}

bool Path::PathHelper::ShouldExpand(const LabelBools &label_bools) const {
  return !label_bools.blacklisted &&
         ((label_bools.end_node && config_.label_bools_status.end_node_activated) || label_bools.terminated ||
          (!config_.label_bools_status.termination_activated && !config_.label_bools_status.end_node_activated &&
           Whitelisted(label_bools.whitelisted)));
}

void Path::PathHelper::FilterLabelBoolStatus() {
  // end node is activated, which means only paths ending with it can be saved as
  // result, but they can be expanded further
  config_.label_bools_status.end_node_activated = !config_.label_sets.end_list.empty();

  // whitelist is empty, which means all nodes are whitelisted
  config_.label_bools_status.whitelist_empty = config_.label_sets.whitelist.empty();

  // there is a termination node, so only paths ending with it are allowed
  config_.label_bools_status.termination_activated = !config_.label_sets.termination_list.empty();
}

/*function to set appropriate parameters for filtering*/
void Path::PathHelper::FilterLabel(std::string_view label, LabelBools &label_bools) {
  if (config_.label_sets.blacklist.find(label) != config_.label_sets.blacklist.end()) {  // if label is blacklisted
    label_bools.blacklisted = true;
  }

  if (config_.label_sets.termination_list.find(label) !=
      config_.label_sets.termination_list.end()) {  // if label is termination label
    label_bools.terminated = true;
  }

  if (config_.label_sets.end_list.find(label) != config_.label_sets.end_list.end()) {  // if label is end label
    label_bools.end_node = true;
  }

  if (config_.label_sets.whitelist.find(label) != config_.label_sets.whitelist.end()) {  // if label is whitelisted
    label_bools.whitelisted = true;
  }
}

/*function that takes input list of labels, and sorts them into appropriate category
sets were used so when filtering is done, its done in O(1)*/
void Path::PathHelper::ParseLabels(const mgp::List &list_of_labels) {
  for (const auto label : list_of_labels) {
    std::string_view label_string = label.ValueString();
    const char first_elem = label_string.front();
    switch (first_elem) {
      case '-':
        label_string.remove_prefix(1);
        config_.label_sets.blacklist.insert(label_string);
        break;
      case '>':
        label_string.remove_prefix(1);
        config_.label_sets.end_list.insert(label_string);
        break;
      case '+':
        label_string.remove_prefix(1);
        config_.label_sets.whitelist.insert(label_string);
        break;
      case '/':
        label_string.remove_prefix(1);
        config_.label_sets.termination_list.insert(label_string);
        break;
      default:
        config_.label_sets.whitelist.insert(label_string);
        break;
    }
  }
}

/*function that takes input list of relationships, and sorts them into appropriate categories
sets were also used to reduce complexity*/
void Path::PathHelper::ParseRelationships(const mgp::List &list_of_relationships) {
  if (list_of_relationships.Size() ==
      0) {  // if no relationships were passed as arguments, all relationships are allowed
    config_.any_outgoing = true;
    config_.any_incoming = true;
    return;
  }

  for (const auto rel : list_of_relationships) {
    std::string rel_type{std::string(rel.ValueString())};
    bool starts_with = rel_type.starts_with('<');
    bool ends_with = rel_type.ends_with('>');

    if (rel_type.size() == 1) {
      if (starts_with) {
        config_.any_incoming = true;
      } else if (ends_with) {
        config_.any_outgoing = true;
      } else {
        config_.relationship_sets[rel_type] = RelDirection::kAny;
      }
      continue;
    }

    if (starts_with && ends_with) {
      config_.relationship_sets[rel_type.substr(1, rel_type.size() - 2)] = RelDirection::kBoth;
    } else if (starts_with) {
      config_.relationship_sets[rel_type.substr(1)] = RelDirection::kIncoming;
    } else if (ends_with) {
      config_.relationship_sets[rel_type.substr(0, rel_type.size() - 1)] = RelDirection::kOutgoing;
    } else {
      config_.relationship_sets[rel_type] = RelDirection::kAny;
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

void Path::Create(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto start_node{arguments[0].ValueNode()};
    auto relationships{arguments[1].ValueMap()};

    mgp::Path path{start_node};
    for (const auto &relationship : relationships["rel"].ValueList()) {
      if (relationship.IsNull()) {
        break;
      }
      if (!relationship.IsRelationship()) {
        std::ostringstream oss;
        oss << relationship.Type();
        throw mgp::ValueException("Expected relationship or null type, got " + oss.str());
      }

      const auto rel = relationship.ValueRelationship();
      auto last_node = path.GetNodeAt(path.Length());

      bool endpoint_is_from = false;

      if (last_node.Id() == rel.From().Id()) {
        endpoint_is_from = true;
      }

      auto contains = [](mgp::Relationships relationships, const mgp::Id id) {
        for (const auto relationship : relationships) {
          if (relationship.To().Id() == id) {
            return true;
          }
        }
        return false;
      };

      if ((endpoint_is_from && !contains(rel.From().OutRelationships(), rel.To().Id())) ||
          (!endpoint_is_from && !contains(rel.To().OutRelationships(), rel.From().Id()))) {
        break;
      }

      path.Expand(rel);
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultCreate).c_str(), path);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

/*function used for traversal and filtering*/
void Path::PathDFS(mgp::Path &path, const mgp::RecordFactory &record_factory, int64_t path_size,
                   PathHelper &path_helper, std::unordered_set<int64_t> &visited) {
  const mgp::Node node{path.GetNodeAt(path.Length())};

  LabelBools label_bools;
  for (auto label : node.Labels()) {
    path_helper.FilterLabel(label, label_bools);
  }

  if (path_helper.PathSizeOk(path_size) && path_helper.ShouldExpand(label_bools)) {
    auto record = record_factory.NewRecord();
    record.Insert(std::string(std::string(kResultExpand)).c_str(), path);
  }

  if (path_helper.PathTooBig(path_size + 1) || label_bools.terminated || label_bools.blacklisted ||
      !(label_bools.end_node || path_helper.Whitelisted(label_bools.whitelisted))) {
    return;
  }

  std::set<std::pair<std::string_view, int64_t>> seen;

  auto iterate = [&](mgp::Relationships relationships, bool outgoing) {
    for (const auto relationship : relationships) {
      auto type = std::string(relationship.Type());
      auto wanted_direction = path_helper.GetDirection(type);

      if ((wanted_direction == RelDirection::kNone && !path_helper.AnyDirected(outgoing)) ||
          visited.contains(relationship.Id().AsInt())) {
        continue;
      }

      RelDirection curr_direction = outgoing ? RelDirection::kOutgoing : RelDirection::kIncoming;

      auto expand = [&]() {
        path.Expand(relationship);
        visited.insert(relationship.Id().AsInt());
        PathDFS(path, record_factory, path_size + 1, path_helper, visited);
        visited.erase(relationship.Id().AsInt());
        path.Pop();
      };

      if (wanted_direction == RelDirection::kAny || curr_direction == wanted_direction ||
          path_helper.AnyDirected(outgoing)) {
        expand();
      } else if (wanted_direction == RelDirection::kBoth) {
        if (outgoing && seen.contains({type, relationship.To().Id().AsInt()})) {
          expand();
        } else {
          seen.insert({type, relationship.From().Id().AsInt()});
        }
      }
    }
  };

  iterate(node.InRelationships(), false);
  iterate(node.OutRelationships(), true);
}

void Path::StartFunction(const mgp::Node &node, const mgp::RecordFactory &record_factory, PathHelper &path_helper) {
  mgp::Path path = mgp::Path(node);
  std::unordered_set<int64_t> visited;
  PathDFS(path, record_factory, 0, path_helper, visited);
}

void Path::Expand(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto graph = mgp::Graph(memgraph_graph);
    const mgp::Value start_value = arguments[0];
    mgp::List relationships{arguments[1].ValueList()};
    const mgp::List labels{arguments[2].ValueList()};
    int64_t min_hops{arguments[3].ValueInt()};
    int64_t max_hops{arguments[4].ValueInt()};

    PathHelper path_helper = PathHelper(labels, relationships, min_hops, max_hops);

    auto parse = [&](const mgp::Value &value) {
      if (value.IsNode()) {
        StartFunction(value.ValueNode(), record_factory, path_helper);
      } else if (value.IsInt()) {
        StartFunction(graph.GetNodeById(mgp::Id::FromInt(value.ValueInt())), record_factory, path_helper);
      } else {
        throw mgp::ValueException("Invalid start type. Expected Node, Int, List[Node, Int]");
      }
    };

    if (!start_value.IsList()) {
      parse(start_value);
      return;
    }

    for (const auto &list_item : start_value.ValueList()) {
      parse(list_item);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

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
    if (string_rel_type == type || string_rel_type.empty()) {
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
    for (const auto &node : start_nodes) {
      VisitNode(node, visited_nodes, true, config, 0, labelFilterSets, to_be_returned_nodes);
    }

    for (const auto &node : to_be_returned_nodes) {
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
    for (const auto &node : start_nodes) {
      VisitNode(node, visited_nodes, true, config, 0, labelFilterSets, to_be_returned_nodes);
    }

    std::unordered_set<mgp::Node> to_be_returned_nodes_searchable;
    for (const auto &node : to_be_returned_nodes) {
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
