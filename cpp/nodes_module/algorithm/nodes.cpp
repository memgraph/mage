#include "nodes.hpp"

bool Nodes::RelationshipExist(const mgp::Node &node, std::string &rel_type) {
  char direction{' '};
  if (rel_type[rel_type.size() - 1] == '<' || rel_type[rel_type.size() - 1] == '>') {
    direction = rel_type[rel_type.size() - 1];
    rel_type.pop_back();
  }
  for (auto rel : node.OutRelationships()) {
    if (std::string(rel.Type()) == rel_type && (direction == '>' || direction != '<')) {
      return true;
    }
  }
  for (auto rel : node.InRelationships()) {
    if (std::string(rel.Type()) == rel_type && (direction == '<' || direction != '>')) {
      return true;
    }
  }
  return false;
}

void Nodes::RelationshipsExist(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const mgp::List nodes = arguments[0].ValueList();
    const mgp::List relationships = arguments[1].ValueList();
    if (nodes.Size() == 0 || relationships.Size() == 0) {
      throw mgp::ValueException("Input lists must not be empty!");
    }

    for (auto node : nodes) {
      mgp::Map return_map = mgp::Map();
      mgp::Map relationship_map = mgp::Map();
      for (auto rel : relationships) {
        std::string rel_type{rel.ValueString()};
        if (RelationshipExist(node.ValueNode(), rel_type)) {
          relationship_map.Insert(rel.ValueString(), mgp::Value(true));
        } else {
          relationship_map.Insert(rel.ValueString(), mgp::Value(false));
        }
      }
      return_map.Insert("Node", node);
      return_map.Insert("Relationships_exist_status", mgp::Value(std::move(relationship_map)));
      auto record = record_factory.NewRecord();
      record.Insert(std::string(kReturnRelationshipsExist).c_str(), std::move(return_map));
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
