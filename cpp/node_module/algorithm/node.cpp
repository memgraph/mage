#include "node.hpp"

bool Node::FindRelationship(std::unordered_set<std::string_view> types, mgp::Relationships relationships) {
  if (types.contains("") && relationships.cbegin() != relationships.cend()) {
    return true;
  }
  for (auto relationship : relationships) {
    if (types.contains(relationship.Type())) {
      return true;
    }
  }
  return false;
}

void Node::RelExists(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto node = arguments[0].ValueNode();
    auto patterns = arguments[1].ValueList();

    if (patterns.Empty()) {
      patterns.AppendExtend(mgp::Value(""));
    }

    std::unordered_set<std::string_view> in_rels;
    std::unordered_set<std::string_view> out_rels;

    for (auto pattern_value : patterns) {
      auto pattern = pattern_value.ValueString();
      if (pattern[0] == '<' && pattern[pattern.size() - 1] == '>') {
        throw mgp::ValueException("Invalid relationship specification!");
      }
      if (pattern[0] == '<') {
        in_rels.insert(pattern.substr(1, pattern.size()));
        continue;
      }
      if (pattern[pattern.size() - 1] == '>') {
        out_rels.insert(pattern.substr(0, pattern.size() - 1));
        continue;
      }
      in_rels.insert(std::move(pattern));
      out_rels.insert(std::move(pattern));
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRelExists).c_str(), FindRelationship(in_rels, node.InRelationships()) ||
                                                             FindRelationship(out_rels, node.OutRelationships()));
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
