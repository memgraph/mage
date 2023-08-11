#include "path.hpp"

bool Path::PathSizeOk(const int64_t path_size, const int64_t &max_hops, const int64_t &min_hops) {
  return (path_size + 1 <= max_hops) && (path_size + 1 >= min_hops);
}

bool Path::Whitelisted(const bool &whitelisted, const bool &whitelist_empty) {
  return (whitelisted || whitelist_empty);
}

void Path::FilterLabelBoolStatus(const LabelSets &labelSets, LabelBoolsStatus &labelStatus) {
  if (labelSets.end_list.size() != 0) {  // end node is activated, which means only paths ending with it can be saved as
                                         // result, but they can be expanded further
    labelStatus.end_node_activated = true;
  }
  if (labelSets.whitelist.size() == 0) {  // whitelist is emptym which means, everything is a whitelisted node
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
  const std::string label_string = std::string(label);
  if (labelSets.blacklist.find(label_string) != labelSets.blacklist.end()) {  // if label is blacklisted
    labelBools.blacklisted = true;
  }

  if (labelSets.termination_list.find(label_string) !=
      labelSets.termination_list.end()) {  // if label is termination label
    labelBools.terminated = true;
  }

  if (labelSets.end_list.find(label_string) != labelSets.end_list.end()) {  // if label is end label
    labelBools.end_node = true;
  }

  if (labelSets.whitelist.find(label_string) != labelSets.whitelist.end()) {  // if label is whitelisted
    labelBools.whitelisted = true;
  }
}

/*function that takes input list of labels, and sorts them into appropriate category
sets were used so when filtering is done, its done in O(1)*/
void Path::ParseLabels(const mgp::List &list_of_labels, LabelSets &labelSets) {
  for (const auto label : list_of_labels) {
    std::string label_string = std::string(label.ValueString());
    const char first_elem = label_string[0];
    switch (first_elem) {
      case '-':
        labelSets.blacklist.insert(label_string.erase(0, 1));
        break;
      case '>':
        labelSets.end_list.insert(label_string.erase(0, 1));
        break;
      case '+':
        labelSets.whitelist.insert(label_string.erase(0, 1));
        break;
      case '/':
        labelSets.termination_list.insert(label_string.erase(0, 1));
        break;
      default:  // default is that everything goes to whitelist, unless specified as above
        labelSets.whitelist.insert(label_string);
        break;
    }
  }
}

/*function that takes input list of relationships, and sorts them into appropriate categories
sets were also used to reduce complexity*/
void Path::Parse_Relationships(const mgp::List &list_of_relationships, RelationshipSets &relationshipSets,
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

    if (PathSizeOk(path_size, max_hops, min_hops) && !labelBools.blacklisted &&
        ((labelBools.end_node && labelStatus.end_node_activated) || labelBools.terminated ||
         (!labelStatus.termination_activated && !labelStatus.end_node_activated &&
          Whitelisted(labelBools.whitelisted, labelStatus.whitelist_empty)))) {
      auto record = record_factory.NewRecord();
      record.Insert(std::string(std::string(kResultExpand).c_str()).c_str(), cpy);
    }
    if (path_size + 1 < max_hops && !labelBools.blacklisted && !labelBools.terminated &&
        (labelBools.end_node || Whitelisted(labelBools.whitelisted, labelStatus.whitelist_empty))) {
      Path_DFS(cpy, relationships_set_cpy, record_factory, path_size + 1, min_hops, max_hops, labelSets, labelStatus,
               relationshipSets, any_outgoing, any_incoming);
    }
  }
}

/*function used for traversal and filtering*/
void Path::Path_DFS(mgp::Path path, std::unordered_set<mgp::Relationship> relationships_set,
                    const mgp::RecordFactory &record_factory, int64_t path_size, const int64_t min_hops,
                    const int64_t max_hops, const LabelSets &labelSets, const LabelBoolsStatus &labelStatus,
                    const RelationshipSets &relationshipSets, bool &any_outgoing, bool &any_incoming) {
  DfsByDirection(path, relationships_set, record_factory, path_size, min_hops, max_hops, labelSets, labelStatus,
                 relationshipSets, any_outgoing, any_incoming, true);
  DfsByDirection(path, relationships_set, record_factory, path_size, min_hops, max_hops, labelSets, labelStatus,
                 relationshipSets, any_outgoing, any_incoming, false);
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
    Parse_Relationships(relationships, relationshipSets, any_outgoing, any_incoming);
    /*end filter relationships part*/

    if (start_value.IsNode()) {
      mgp::Path path = mgp::Path(start_value.ValueNode());
      std::unordered_set<mgp::Relationship> relationships_set;
      Path_DFS(path, relationships_set, record_factory, 0, min_hops, max_hops, labelSets, labelStatus, relationshipSets,
               any_outgoing, any_incoming);
    } else if (start_value.IsInt()) {
      mgp::Path path = mgp::Path(graph.GetNodeById(mgp::Id::FromInt(start_value.ValueInt())));
      std::unordered_set<mgp::Relationship> relationships_set;
      Path_DFS(path, relationships_set, record_factory, 0, min_hops, max_hops, labelSets, labelStatus, relationshipSets,
               any_outgoing, any_incoming);
    } else if (start_value.IsList()) {
      for (const auto value : start_value.ValueList()) {
        if (value.IsNode()) {
          mgp::Path path = mgp::Path(value.ValueNode());
          std::unordered_set<mgp::Relationship> relationships_set;
          Path_DFS(path, relationships_set, record_factory, 0, min_hops, max_hops, labelSets, labelStatus,
                   relationshipSets, any_outgoing, any_incoming);

        } else if (value.IsInt()) {
          mgp::Path path = mgp::Path(graph.GetNodeById(mgp::Id::FromInt(value.ValueInt())));
          std::unordered_set<mgp::Relationship> relationships_set;
          Path_DFS(path, relationships_set, record_factory, 0, min_hops, max_hops, labelSets, labelStatus,
                   relationshipSets, any_outgoing, any_incoming);
        } else {
          throw mgp::ValueException("Invalid start type, expected Node, Int,or List of Node or Int");
        }
      }
    } else {
      throw mgp::ValueException("Invalid start type, expected Node, Int,or List of Node or Int");
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
