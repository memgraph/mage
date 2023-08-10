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
    const auto input_pattern = arguments[1].ValueString();

    std::unordered_set<std::string_view> patterns;
    auto i{0};
    auto j{input_pattern.find("|")};
    while (j < input_pattern.size()) {
      patterns.insert(input_pattern.substr(i, j - i));
      i = j + 1;
      j = input_pattern.find("|", i);
    }
    patterns.insert(input_pattern.substr(i));

    std::unordered_set<std::string_view> in_rels;
    std::unordered_set<std::string_view> out_rels;

    for (auto pattern : patterns) {
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
      in_rels.insert(pattern);
      out_rels.insert(pattern);
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRelExists).c_str(), FindRelationship(in_rels, node.InRelationships()) ||
                                                             FindRelationship(out_rels, node.OutRelationships()));
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
