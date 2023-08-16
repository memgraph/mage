#include "nodes.hpp"

bool Nodes::RelationshipExist(const mgp::Node &node, std::string &rel_type) {
  char direction{' '};
  if (rel_type[0] == '<' && rel_type[rel_type.size() - 1] == '>') {
    throw mgp::ValueException("Invalid relationship specification!");
  } else if (rel_type[rel_type.size() - 1] == '>') {
    direction = rel_type[rel_type.size() - 1];
    rel_type.pop_back();
  } else if (rel_type[0] == '<') {
    direction = rel_type[0];
    rel_type.erase(0, 1);
  }
  for (const auto rel : node.OutRelationships()) {
    if (std::string(rel.Type()) == rel_type && direction != '<') {
      return true;
    }
  }
  for (const auto rel : node.InRelationships()) {
    if (std::string(rel.Type()) == rel_type && direction != '>') {
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
    const mgp::Graph graph = mgp::Graph(memgraph_graph);
    const mgp::List nodes = arguments[0].ValueList();
    const mgp::List relationships = arguments[1].ValueList();
    if (nodes.Size() == 0 || relationships.Size() == 0) {
      throw mgp::ValueException("Input lists must not be empty!");
    }

    for (auto element: nodes) {
      if(!element.IsNode() && !element.IsInt()){
        throw mgp::ValueException("Input arguments must be nodes or their ID's");
      }
      mgp::Node node = element.IsNode() ? element.ValueNode() : graph.GetNodeById(mgp::Id::FromInt(element.ValueInt()));
      mgp::Map return_map = mgp::Map();
      mgp::Map relationship_map = mgp::Map();
      for (auto rel : relationships) {
        std::string rel_type{rel.ValueString()};
        if (RelationshipExist(node, rel_type)) {
          relationship_map.Insert(rel.ValueString(), mgp::Value(true));
        } else {
          relationship_map.Insert(rel.ValueString(), mgp::Value(false));
        }
      }
      return_map.Insert(std::string(kNodeRelationshipsExist).c_str(), mgp::Value(node));
      return_map.Insert(std::string(kRelationshipsExistStatus).c_str(), mgp::Value(std::move(relationship_map)));
      auto record = record_factory.NewRecord();
      record.Insert(std::string(kReturnRelationshipsExist).c_str(), std::move(return_map));
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
