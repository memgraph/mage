#include "node.hpp"

bool Node::RelationshipExist(const mgp::Node &node, std::string &rel_type) {
  char direction{' '};
  if (rel_type[rel_type.size() - 1] == '<' || rel_type[rel_type.size() - 1] == '>') {
    direction = rel_type[rel_type.size() - 1];
    rel_type.pop_back();
  }
  for (auto rel : node.OutRelationships()) {
    if (std::string(rel.Type()) == rel_type && direction != '<') {
      return true;
    }
  }
  for (auto rel : node.InRelationships()) {
    if (std::string(rel.Type()) == rel_type && direction != '>') {
      return true;
    }
  }
  return false;
}

void Node::RelationshipsExist(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const mgp::Node node = arguments[0].ValueNode();
    const mgp::List relationships = arguments[1].ValueList();
    if (relationships.Size() == 0) {
      throw mgp::ValueException("Input relationships list must not be empty!");
    }
    mgp::Map relationship_map = mgp::Map();
    for (auto rel : relationships) {
      std::string rel_type{rel.ValueString()};
      if (RelationshipExist(node, rel_type)) {
        relationship_map.Insert(rel.ValueString(), mgp::Value(true));
      } else {
        relationship_map.Insert(rel.ValueString(), mgp::Value(false));
      }
    }
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kReturnRelationshipsExist).c_str(), std::move(relationship_map));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
