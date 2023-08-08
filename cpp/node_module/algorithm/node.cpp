#include "node.hpp"

bool Node::FindRelationship(std::string_view type, mgp::Relationships relationships) {
  if (type.size() == 0 && relationships.cbegin() != relationships.cend()) {
    return true;
  }
  for (auto relationship : relationships) {
    if (relationship.Type() == type) {
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
    const auto pattern = arguments[1].ValueString();
    bool exists = false;

    if (pattern[0] == '<' && pattern[pattern.size() - 1] == '>') {
      throw mgp::ValueException("Invalid relationship specification!");
    }
    if (pattern[0] == '<') {
      exists = FindRelationship(pattern.substr(1, pattern.size()), node.InRelationships());
    } else if (pattern[pattern.size() - 1] == '>') {
      exists = FindRelationship(pattern.substr(0, pattern.size() - 1), node.OutRelationships());
    } else {
      exists =
          (FindRelationship(pattern, node.InRelationships()) || FindRelationship(pattern, node.OutRelationships()));
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRelExists).c_str(), std::move(exists));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
