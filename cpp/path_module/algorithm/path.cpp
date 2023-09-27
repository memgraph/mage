#include "path.hpp"

#include "mgp.hpp"

Path::PathHelper::PathHelper(const mgp::List &labels, const mgp::List &relationships, int64_t min_hops,
                             int64_t max_hops) {
  ParseLabels(labels);
  FilterLabelBoolStatus();
  ParseRelationships(relationships);
  config_.min_hops = min_hops;
  config_.max_hops = max_hops;
}

Path::PathHelper::PathHelper(const mgp::Map &config) {
  if (!(config.At("minHops").IsNull() || config.At("minHops").IsInt()) ||
      !(config.At("maxHops").IsNull() || config.At("maxHops").IsInt()) ||
      !(config.At("relationshipFilter").IsNull() || config.At("relationshipFilter").IsList()) ||
      !(config.At("labelFilter").IsNull() || config.At("labelFilter").IsList()) ||
      !(config.At("filterStartNode").IsNull() || config.At("filterStartNode").IsBool()) ||
      !(config.At("beginSequenceAtStart").IsNull() || config.At("beginSequenceAtStart").IsBool()) ||
      !(config.At("bfs").IsNull() || config.At("bfs").IsBool())) {
    throw mgp::ValueException(
        "The config parameter needs to be a map with keys and values in line with the documentation.");
  }

  auto value = config.At("maxHops");
  if (!value.IsNull()) {
    config_.max_hops = value.ValueInt();
  }
  value = config.At("minHops");
  if (!value.IsNull()) {
    config_.min_hops = value.ValueInt();
  }

  value = config.At("relationshipFilter");
  if (!value.IsNull()) {
    ParseRelationships(value.ValueList());
  } else {
    ParseRelationships(mgp::List());
  }

  value = config.At("labelFilter");
  if (!value.IsNull()) {
    ParseLabels(value.ValueList());
  } else {
    ParseLabels(mgp::List());
  }
  FilterLabelBoolStatus();

  value = config.At("filterStartNode");
  config_.filter_start_node = value.IsNull() ? true : value.ValueBool();

  value = config.At("beginSequenceAtStart");
  config_.begin_sequence_at_start = value.IsNull() ? true : value.ValueBool();

  value = config.At("bfs");
  config_.bfs = value.IsNull() ? false : value.ValueBool();
}

Path::RelDirection Path::PathHelper::GetDirection(std::string &rel_type) {
  auto it = config_.relationship_sets.find(rel_type);
  if (it == config_.relationship_sets.end()) {
    return RelDirection::kNone;
  }
  return it->second;
}

Path::LabelBools Path::PathHelper::GetLabelBools(const mgp::Node &node) {
  LabelBools label_bools;
  for (auto label : node.Labels()) {
    FilterLabel(label, label_bools);
  }
  return label_bools;
}

bool Path::PathHelper::AreLabelsValid(const LabelBools &label_bools) const {
  return !label_bools.blacklisted &&
         ((label_bools.end_node && config_.label_bools_status.end_node_activated) || label_bools.terminated ||
          (!config_.label_bools_status.termination_activated && !config_.label_bools_status.end_node_activated &&
           Whitelisted(label_bools.whitelisted)));
}

bool Path::PathHelper::ContinueExpanding(const LabelBools &label_bools, size_t path_size) const {
  return (static_cast<int64_t>(path_size) <= config_.max_hops && !label_bools.blacklisted && !label_bools.terminated &&
          (label_bools.end_node || Whitelisted(label_bools.whitelisted)));
}

bool Path::PathHelper::PathSizeOk(const int64_t path_size) const {
  return (path_size <= config_.max_hops) && (path_size >= config_.min_hops);
}

bool Path::PathHelper::PathTooBig(const int64_t path_size) const { return path_size > config_.max_hops; }

bool Path::PathHelper::Whitelisted(const bool whitelisted) const {
  return (config_.label_bools_status.whitelist_empty || whitelisted);
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

void Path::Create(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
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

void Path::PathExpand::ExpandPath(mgp::Path &path, const mgp::Relationship &relationship, int64_t path_size) {
  path.Expand(relationship);
  visited_.insert(relationship.Id().AsInt());
  DFS(path, path_size + 1);
  visited_.erase(relationship.Id().AsInt());
  path.Pop();
}

void Path::PathExpand::Expand(mgp::Path &path, mgp::Relationships relationships, bool outgoing, int64_t path_size,
                              std::set<std::pair<std::string_view, int64_t>> &seen) {
  for (const auto relationship : relationships) {
    auto type = std::string(relationship.Type());
    auto wanted_direction = helper_.GetDirection(type);

    if ((wanted_direction == RelDirection::kNone && !helper_.AnyDirected(outgoing)) ||
        visited_.contains(relationship.Id().AsInt())) {
      continue;
    }

    RelDirection curr_direction = outgoing ? RelDirection::kOutgoing : RelDirection::kIncoming;

    if (wanted_direction == RelDirection::kAny || curr_direction == wanted_direction || helper_.AnyDirected(outgoing)) {
      ExpandPath(path, relationship, path_size);
    } else if (wanted_direction == RelDirection::kBoth) {
      if (outgoing && seen.contains({type, relationship.To().Id().AsInt()})) {
        ExpandPath(path, relationship, path_size);
      } else {
        seen.insert({type, relationship.From().Id().AsInt()});
      }
    }
  }
}

/*function used for traversal and filtering*/
void Path::PathExpand::DFS(mgp::Path &path, int64_t path_size) {
  const mgp::Node node{path.GetNodeAt(path_size)};

  LabelBools label_bools = helper_.GetLabelBools(node);
  if (helper_.PathSizeOk(path_size) && helper_.AreLabelsValid(label_bools)) {
    auto record = record_factory_.NewRecord();
    record.Insert(std::string(kResultExpand).c_str(), path);
  }

  if (!helper_.ContinueExpanding(label_bools, path_size + 1)) {
    return;
  }

  std::set<std::pair<std::string_view, int64_t>> seen;
  this->Expand(path, node.InRelationships(), false, path_size, seen);
  this->Expand(path, node.OutRelationships(), true, path_size, seen);
}

void Path::PathExpand::StartAlgorithm(const mgp::Node node) {
  mgp::Path path = mgp::Path(node);
  DFS(path, 0);
}

void Path::PathExpand::Parse(const mgp::Value &value) {
  if (value.IsNode()) {
    StartAlgorithm(value.ValueNode());
  } else if (value.IsInt()) {
    StartAlgorithm(graph_.GetNodeById(mgp::Id::FromInt(value.ValueInt())));
  } else {
    throw mgp::ValueException("Invalid start type. Expected Node, Int, List[Node, Int]");
  }
}

void Path::Expand(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto graph = mgp::Graph(memgraph_graph);
    const mgp::Value start_value = arguments[0];
    mgp::List relationships{arguments[1].ValueList()};
    const mgp::List labels{arguments[2].ValueList()};
    int64_t min_hops{arguments[3].ValueInt()};
    int64_t max_hops{arguments[4].ValueInt()};

    PathHelper path_helper{labels, relationships, min_hops, max_hops};
    PathExpand path_expand{path_helper, record_factory, graph};

    if (!start_value.IsList()) {
      path_expand.Parse(start_value);
      return;
    }

    for (const auto &list_item : start_value.ValueList()) {
      path_expand.Parse(list_item);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Path::PathSubgraph::Parse(const mgp::Value &value) {
  if (!(value.IsNode() || value.IsInt())) {
    throw mgp::ValueException("The first argument needs to be a node, an integer ID, or a list thereof.");
  }
  if (value.IsNode()) {
    start_nodes_.insert(value.ValueNode());
    return;
  }
  start_nodes_.insert(graph_.GetNodeById(mgp::Id::FromInt(value.ValueInt())));
}

void Path::PathSubgraph::Expand(Pair &pair, mgp::Relationships relationships, bool outgoing, std::queue<Pair> &queue,
                                std::set<std::pair<std::string_view, int64_t>> &seen) {
  for (const auto relationship : relationships) {
    auto next_node = outgoing ? relationship.To() : relationship.From();
    auto type = std::string(relationship.Type());
    auto wanted_direction = helper_.GetDirection(type);

    if (helper_.FilterRelationships(pair.hop_count == 0)) {
      if ((wanted_direction == RelDirection::kNone && !helper_.AnyDirected(outgoing)) ||
          visited_.contains(next_node.Id().AsInt())) {
        continue;
      }
    }

    RelDirection curr_direction = outgoing ? RelDirection::kOutgoing : RelDirection::kIncoming;

    if (wanted_direction == RelDirection::kAny || curr_direction == wanted_direction || helper_.AnyDirected(outgoing)) {
      visited_.insert(next_node.Id().AsInt());
      queue.push({next_node, pair.hop_count + 1});
    } else if (wanted_direction == RelDirection::kBoth) {
      if (outgoing && seen.contains({type, relationship.To().Id().AsInt()})) {
        visited_.insert(next_node.Id().AsInt());
        queue.push({next_node, pair.hop_count + 1});
        to_be_returned_nodes_.AppendExtend(mgp::Value{next_node});
      } else {
        seen.insert({type, relationship.From().Id().AsInt()});
      }
    }
  }
}

void Path::PathSubgraph::InsertNode(const mgp::Node &node, int64_t hop_count, LabelBools &label_bools) {
  if (helper_.FilterNodes(hop_count == 0)) {
    if (helper_.AreLabelsValid(label_bools)) {
      to_be_returned_nodes_.AppendExtend(mgp::Value(node));
    }
    return;
  }

  if (!visited_.contains(node.Id().AsInt())) {
    to_be_returned_nodes_.AppendExtend(mgp::Value(node));
  }
}

mgp::List Path::PathSubgraph::BFS() {
  std::queue<Pair> queue;
  std::unordered_set<int64_t> visited;

  for (const auto &node : start_nodes_) {
    queue.push({node, 0});
    visited.insert(node.Id().AsInt());
  }

  while (!queue.empty()) {
    auto pair = queue.front();
    queue.pop();

    if (helper_.PathTooBig(pair.hop_count)) {
      continue;
    }

    LabelBools label_bools = helper_.GetLabelBools(pair.node);
    InsertNode(pair.node, pair.hop_count, label_bools);
    if (!helper_.ContinueExpanding(label_bools, pair.hop_count + 1)) {
      continue;
    }

    std::set<std::pair<std::string_view, int64_t>> seen;
    this->Expand(pair, pair.node.InRelationships(), false, queue, seen);
    this->Expand(pair, pair.node.OutRelationships(), true, queue, seen);
  }

  return to_be_returned_nodes_;
}

void Path::SubgraphNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto config = arguments[1].ValueMap();
    PathHelper path_helper{config};
    PathSubgraph path_subgraph{path_helper, record_factory, graph};

    auto start_value = arguments[0];
    if (!start_value.IsList()) {
      path_subgraph.Parse(start_value);
    } else {
      for (const auto &list_item : start_value.ValueList()) {
        path_subgraph.Parse(list_item);
      }
    }

    auto to_be_returned_nodes = path_subgraph.BFS();

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
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto config = arguments[1].ValueMap();
    PathHelper path_helper{config};
    PathSubgraph path_subgraph{path_helper, record_factory, graph};

    auto start_value = arguments[0];
    if (!start_value.IsList()) {
      path_subgraph.Parse(start_value);
    } else {
      for (const auto &list_item : start_value.ValueList()) {
        path_subgraph.Parse(list_item);
      }
    }

    auto to_be_returned_nodes = path_subgraph.BFS();

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
