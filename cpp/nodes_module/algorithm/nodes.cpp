#include "nodes.hpp"
#include <sys/types.h>

#include <cstdint>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include "mgp.hpp"

namespace {
void throw_exception(const mgp::Value &value) {
  std::ostringstream oss;
  oss << value.Type();
  throw mgp::ValueException("Unsuppported type for this operation, received type: " + oss.str());
};

mgp::Value insert_node_relationship_types(const auto &node,
                                          std::unordered_map<std::string_view, uint8_t> &type_direction) {
  mgp::Map result{};
  result.Insert("node", mgp::Value(node));

  std::unordered_set<std::string_view> types;
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

  mgp::List types_list{types.size()};
  for (const auto &type : types) {
    auto value = mgp::Value(type);
    types_list.Append(value);
  }
  result.Insert("types", mgp::Value(std::move(types_list)));

  return mgp::Value(std::move(result));
}

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

mgp::List get_relationship_types(mgp_graph *memgraph_graph, const mgp::Value &argument, const mgp::Value &types) {
  mgp::List result{};
  mgp::Graph graph{memgraph_graph};
  auto type_direction = get_type_direction(types);

  if (argument.IsNode()) {
    result.AppendExtend(insert_node_relationship_types(argument.ValueNode(), type_direction));
  } else if (argument.IsInt()) {
    result.AppendExtend(
        insert_node_relationship_types(graph.GetNodeById(mgp::Id::FromInt(argument.ValueInt())), type_direction));
  } else if (argument.IsList()) {
    for (const auto &list_item : argument.ValueList()) {
      if (list_item.IsNode()) {
        result.AppendExtend(insert_node_relationship_types(list_item.ValueNode(), type_direction));
      } else if (list_item.IsInt()) {
        result.AppendExtend(
            insert_node_relationship_types(graph.GetNodeById(mgp::Id::FromInt(list_item.ValueInt())), type_direction));
      } else {
        throw_exception(list_item);
      }
    }
  } else {
    throw_exception(argument);
  }
  return result;
}
}  // namespace

void Nodes::RelationshipTypes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRelationshipTypes).c_str(),
                  get_relationship_types(memgraph_graph, arguments[0], arguments[1]));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
