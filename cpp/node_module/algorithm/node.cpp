#include "node.hpp"

#include <cstdint>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include "mgp.hpp"

namespace {

std::unordered_map<std::string_view, uint8_t> get_type_direction(const mgp::Value &types) {
  std::unordered_map<std::string_view, uint8_t> result;
  for (const auto &type_value : types.ValueList()) {
    auto type = type_value.ValueString();
    if (type.starts_with('<')) {
      result[type.substr(1, type.size() - 1)] |= 1;
    } else if (type.ends_with('>')) {
      result[type.substr(0, type.size() - 1)] |= 2;
    } else {
      result[type] |= 3;
    }
  }
  return result;
}

mgp::List get_relationship_types(const mgp::Value &node_value, const mgp::Value &types_value) {
  auto type_direction = get_type_direction(types_value);

  std::unordered_set<std::string_view> types;
  const auto node = node_value.ValueNode();
  if (type_direction.empty()) {
    for (const auto relationship : node.InRelationships()) {
      types.insert(relationship.Type());
    }
    for (const auto relationship : node.OutRelationships()) {
      types.insert(relationship.Type());
    }
  } else {
    for (const auto relationship : node.InRelationships()) {
      if (type_direction[relationship.Type()] & 1) {
        types.insert(relationship.Type());
      }
    }
    for (const auto relationship : node.OutRelationships()) {
      if (type_direction[relationship.Type()] & 2) {
        types.insert(relationship.Type());
      }
    }
  }

  mgp::List result{types.size()};
  for (const auto &type : types) {
    auto value = mgp::Value(type);
    result.Append(value);
  }
  return result;
}
}  // namespace

void Node::RelationshipTypes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRelationshipTypes).c_str(), get_relationship_types(arguments[0], arguments[1]));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
