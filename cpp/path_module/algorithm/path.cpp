#include "path.hpp"

#include <algorithm>

#include "mgp.hpp"

void Path::Elements(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard(memory);
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);

  try {
    const auto path{arguments[0].ValuePath()};
    mgp::List split_path(path.Length() * 2 + 1);
    for (int i = 0; i < path.Length(); ++i) {
      split_path.Append(mgp::Value(path.GetNodeAt(i)));
      split_path.Append(mgp::Value(path.GetRelationshipAt(i)));
    }
    split_path.Append(mgp::Value(path.GetNodeAt(path.Length())));
    result.SetValue(std::move(split_path));

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
  }
}

void Path::Combine(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard(memory);
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);

  try {
    auto path1{arguments[0].ValuePath()};
    const auto path2{arguments[1].ValuePath()};

    for (int i = 0; i < path2.Length(); ++i) {
      path1.Expand(path2.GetRelationshipAt(i));
    }

    result.SetValue(std::move(path1));

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
  }
}

void Path::Slice(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard(memory);
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);

  try {
    const auto path{arguments[0].ValuePath()};
    const auto offset{arguments[1].ValueInt()};
    const auto length{arguments[2].ValueInt()};

    mgp::Path new_path{path.GetNodeAt(offset)};
    size_t max_iteration = std::min((length == -1 ? path.Length() : offset + length), path.Length());
    for (int i = static_cast<int>(offset); i < max_iteration; ++i) {
      new_path.Expand(path.GetRelationshipAt(i));
    }

    result.SetValue(std::move(new_path));

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
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

bool Path::PathSizeOk(const int64_t path_size, const int64_t &max_hops, const int64_t &min_hops) {
  return (path_size + 1 <= max_hops) && (path_size + 1 >= min_hops);
}

bool Path::Whitelisted(const bool &whitelisted, const bool &whitelist_empty) {
  return (whitelisted || whitelist_empty);
}
bool Path::ShouldExpand(const LabelBools &labelBools, const LabelBoolsStatus &labelStatus) {
  return !labelBools.blacklisted && ((labelBools.end_node && labelStatus.end_node_activated) || labelBools.terminated ||
                                     (!labelStatus.termination_activated && !labelStatus.end_node_activated &&
                                      Whitelisted(labelBools.whitelisted, labelStatus.whitelist_empty)));
}
void Path::FilterLabelBoolStatus(const LabelSets &labelSets, LabelBoolsStatus &labelStatus) {
  if (labelSets.end_list.size() != 0) {  // end node is activated, which means only paths ending with it can be saved as
                                         // result, but they can be expanded further
    labelStatus.end_node_activated = true;
  }
  if (labelSets.whitelist.size() == 0) {  // whitelist is empty, which means all nodes are whitelisted
    labelStatus.whitelist_empty = true;
  }

  if (labelSets.termination_list.size() != 0) {
    labelStatus.termination_activated = true;  // there is a termination node, so only paths ending with it are allowed
  }
}

bool Path::RelationshipAllowed(const std::string &rel_type, const RelationshipSets &relationshipSets,
                               bool &any_outgoing, bool &any_incoming, bool outgoing) {
  if (outgoing) {  // for outgoing rels
    if (!any_outgoing && (relationshipSets.outgoing_rel.find(rel_type) == relationshipSets.outgoing_rel.end()) &&
        (relationshipSets.any_rel.find(rel_type) ==
         relationshipSets.any_rel.end())) {  // check if relationship is allowed or all relationships are allowed
      return false;
    }

  } else {  // incoming rels
    if (!any_incoming && (relationshipSets.incoming_rel.find(rel_type) == relationshipSets.incoming_rel.end()) &&
        (relationshipSets.any_rel.find(rel_type) == relationshipSets.any_rel.end())) {  // check if rel allowed
      return false;
    }
  }
  return true;
}
/*function to set appropriate parameters for filtering*/
void Path::FilterLabel(const std::string_view label, const LabelSets &labelSets, LabelBools &labelBools) {
  if (labelSets.blacklist.find(label) != labelSets.blacklist.end()) {  // if label is blacklisted
    labelBools.blacklisted = true;
  }

  if (labelSets.termination_list.find(label) != labelSets.termination_list.end()) {  // if label is termination label
    labelBools.terminated = true;
  }

  if (labelSets.end_list.find(label) != labelSets.end_list.end()) {  // if label is end label
    labelBools.end_node = true;
  }

  if (labelSets.whitelist.find(label) != labelSets.whitelist.end()) {  // if label is whitelisted
    labelBools.whitelisted = true;
  }
}

/*function that takes input list of labels, and sorts them into appropriate category
sets were used so when filtering is done, its done in O(1)*/
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

/*function that takes input list of relationships, and sorts them into appropriate categories
sets were also used to reduce complexity*/
void Path::ParseRelationships(const mgp::List &list_of_relationships, RelationshipSets &relationshipSets,
                              bool &any_outgoing, bool &any_incoming) {
  if (list_of_relationships.Size() ==
      0) {  // if no relationships were passed as arguments, all relationships are allowed
    any_outgoing = true;
    any_incoming = true;
    return;
  }
  for (const auto rel : list_of_relationships) {
    std::string rel_type = std::string(rel.ValueString());
    const size_t size = rel_type.size();
    const char first_elem = rel_type[0];
    const char last_elem = rel_type[size - 1];
    if (first_elem == '<' && last_elem == '>') {
      throw mgp::ValueException("Wrong relationship format => <relationship> is not allowed!");
    } else if (first_elem == '<' && size == 1) {
      any_incoming = true;  // all incoming relatiomships are allowed
    } else if (first_elem == '<' && size != 1) {
      relationshipSets.incoming_rel.insert(rel_type.erase(0, 1));  // only specified incoming relationships are allowed
    } else if (last_elem == '>' && size == 1) {
      any_outgoing = true;  // all outgoing relationships are allowed

    } else if (last_elem == '>' && size != 1) {
      rel_type.pop_back();
      relationshipSets.outgoing_rel.insert(rel_type);  // only specifed outgoing relationships are allowed
    } else {                                           // if not specified, a relationship goes both ways
      relationshipSets.any_rel.insert(rel_type);
    }
  }
}

void Path::DfsByDirection(mgp::Path &path, std::unordered_set<mgp::Relationship> &relationships_set,
                          const mgp::RecordFactory &record_factory, int64_t path_size, const int64_t min_hops,
                          const int64_t max_hops, const LabelSets &labelSets, const LabelBoolsStatus &labelStatus,
                          const RelationshipSets &relationshipSets, bool &any_outgoing, bool &any_incoming,
                          bool outgoing) {
  const mgp::Node &node = path.GetNodeAt(path_size);  // get latest node in path, and expand on it

  mgp::Relationships rels = outgoing ? node.OutRelationships() : node.InRelationships();
  for (auto rel : rels) {
    // go through every relationship of the node and expand to the other node of the relationship
    if (relationships_set.find(rel) !=
        relationships_set.end()) {  // relationships_set contains all relationships already visited in this path, and
                                    // the usage of this if loop is to evade cycles
      continue;
    }
    mgp::Path cpy = mgp::Path(path);
    const std::string rel_type = std::string(rel.Type());
    bool rel_allowed = outgoing ? RelationshipAllowed(rel_type, relationshipSets, any_outgoing, any_incoming, true)
                                : RelationshipAllowed(rel_type, relationshipSets, any_outgoing, any_incoming, false);
    if (!rel_allowed) {  // if relationship not allowed, go to next one
      continue;
    }
    cpy.Expand(rel);  // expand the path with this relationships
    std::unordered_set<mgp::Relationship> relationships_set_cpy = relationships_set;
    relationships_set_cpy.insert(rel);  // insert the relationship into visited relationships

    /*this part is for label filtering*/
    LabelBools labelBools;
    mgp::Labels labels = outgoing ? rel.To().Labels() : rel.From().Labels();
    for (auto label : labels) {  // set booleans to their value for the label of the finish node
      FilterLabel(label, labelSets, labelBools);
    }

    if (PathSizeOk(path_size, max_hops, min_hops) && ShouldExpand(labelBools, labelStatus)) {
      auto record = record_factory.NewRecord();
      record.Insert(std::string(std::string(kResultExpand).c_str()).c_str(), cpy);
    }
    if (path_size + 1 < max_hops && !labelBools.blacklisted && !labelBools.terminated &&
        (labelBools.end_node || Whitelisted(labelBools.whitelisted, labelStatus.whitelist_empty))) {
      PathDFS(cpy, relationships_set_cpy, record_factory, path_size + 1, min_hops, max_hops, labelSets, labelStatus,
              relationshipSets, any_outgoing, any_incoming);
    }
  }
}

/*function used for traversal and filtering*/
void Path::PathDFS(mgp::Path path, std::unordered_set<mgp::Relationship> &relationships_set,
                   const mgp::RecordFactory &record_factory, int64_t path_size, const int64_t min_hops,
                   const int64_t max_hops, const LabelSets &labelSets, const LabelBoolsStatus &labelStatus,
                   const RelationshipSets &relationshipSets, bool &any_outgoing, bool &any_incoming) {
  DfsByDirection(path, relationships_set, record_factory, path_size, min_hops, max_hops, labelSets, labelStatus,
                 relationshipSets, any_outgoing, any_incoming, true);
  DfsByDirection(path, relationships_set, record_factory, path_size, min_hops, max_hops, labelSets, labelStatus,
                 relationshipSets, any_outgoing, any_incoming, false);
}

void Path::StartFunction(const mgp::Node &node, const mgp::RecordFactory &record_factory, int64_t path_size,
                         const int64_t min_hops, const int64_t max_hops, const LabelSets &labelSets,
                         const LabelBoolsStatus &labelStatus, const RelationshipSets &relationshipSets,
                         bool &any_outgoing, bool &any_incoming) {
  mgp::Path path = mgp::Path(node);
  std::unordered_set<mgp::Relationship> relationships_set;
  PathDFS(path, relationships_set, record_factory, 0, min_hops, max_hops, labelSets, labelStatus, relationshipSets,
          any_outgoing, any_incoming);
}

void Path::Expand(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph = mgp::Graph(memgraph_graph);
    const mgp::Value start_value = arguments[0];
    mgp::List relationships = arguments[1].ValueList();
    const mgp::List labels = arguments[2].ValueList();
    int64_t min_hops = arguments[3].ValueInt();
    int64_t max_hops = arguments[4].ValueInt();

    /*filter label part*/
    LabelSets labelSets;
    LabelBoolsStatus labelStatus;
    ParseLabels(labels, labelSets);
    FilterLabelBoolStatus(labelSets, labelStatus);
    /*end filter label part*/

    /*filter relationships part*/
    RelationshipSets relationshipSets;
    bool any_outgoing = false;
    bool any_incoming = false;
    ParseRelationships(relationships, relationshipSets, any_outgoing, any_incoming);
    /*end filter relationships part*/

    if (start_value.IsNode()) {
      StartFunction(start_value.ValueNode(), record_factory, 0, min_hops, max_hops, labelSets, labelStatus,
                    relationshipSets, any_outgoing, any_incoming);
    } else if (start_value.IsInt()) {
      StartFunction(graph.GetNodeById(mgp::Id::FromInt(start_value.ValueInt())), record_factory, 0, min_hops, max_hops,
                    labelSets, labelStatus, relationshipSets, any_outgoing, any_incoming);
    } else if (start_value.IsList()) {
      for (const auto value : start_value.ValueList()) {
        if (value.IsNode()) {
          StartFunction(value.ValueNode(), record_factory, 0, min_hops, max_hops, labelSets, labelStatus,
                        relationshipSets, any_outgoing, any_incoming);

        } else if (value.IsInt()) {
          StartFunction(graph.GetNodeById(mgp::Id::FromInt(value.ValueInt())), record_factory, 0, min_hops, max_hops,
                        labelSets, labelStatus, relationshipSets, any_outgoing, any_incoming);
        } else {
          throw mgp::ValueException("Invalid start type. Expected Node, Int, List[Node, Int]");
        }
      }
    } else {
      throw mgp::ValueException("Invalid start type. Expected Node, Int, List[Node, Int]");
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
