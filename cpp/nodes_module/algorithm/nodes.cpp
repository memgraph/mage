#include "nodes.hpp"

#include <unordered_set>
#include "mgp.hpp"

namespace {
void ThrowException(const mgp::Value &value) {
  std::ostringstream oss;
  oss << value.Type();
  throw mgp::ValueException("Unsuppported type for this operation, received type: " + oss.str());
};

mgp::Value InsertNodeRelationshipTypes(const mgp::Node &node,
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

std::unordered_map<std::string_view, uint8_t> GetTypeDirection(const mgp::Value &types) {
  std::unordered_map<std::string_view, uint8_t> result;
  for (const auto &type_value : types.ValueList()) {
    auto type = type_value.ValueString();
    if (type.starts_with('<')) {
      if (type.ends_with('>')) {
        throw mgp::ValueException("<type> format not allowed. Use type instead.");
      }
      result[type.substr(1, type.size() - 1)] |= 1;
    } else if (type.ends_with('>')) {
      result[type.substr(0, type.size() - 1)] |= 2;
    } else {
      result[type] |= 3;
    }
  }
  return result;
}

mgp::List GetRelationshipTypes(mgp_graph *memgraph_graph, const mgp::Value &argument, const mgp::Value &types) {
  mgp::List result{};
  mgp::Graph graph{memgraph_graph};
  auto type_direction = GetTypeDirection(types);

  auto ParseNode = [&](const mgp::Value &value) {
    if (value.IsNode()) {
      result.AppendExtend(InsertNodeRelationshipTypes(value.ValueNode(), type_direction));
    } else if (value.IsInt()) {
      result.AppendExtend(
          InsertNodeRelationshipTypes(graph.GetNodeById(mgp::Id::FromInt(value.ValueInt())), type_direction));
    } else {
      ThrowException(value);
    }
  };

  if (!argument.IsList()) {
    ParseNode(argument);
    return result;
  }

  for (const auto &list_item : argument.ValueList()) {
    ParseNode(list_item);
  }

  return result;
}

void DetachDeleteNode(const mgp::Value &node, mgp::Graph &graph) {
  if (node.IsInt()) {
    graph.DetachDeleteNode(graph.GetNodeById(mgp::Id::FromInt(node.ValueInt())));
  } else if (node.IsNode()) {
    graph.DetachDeleteNode(node.ValueNode());
  } else {
    ThrowException(node);
  }
}

}  // namespace

void Nodes::RelationshipTypes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRelationshipTypes).c_str(),
                  GetRelationshipTypes(memgraph_graph, arguments[0], arguments[1]));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Nodes::Delete(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph{memgraph_graph};
    auto nodes{arguments[0]};

    if (!nodes.IsList()) {
      DetachDeleteNode(nodes, graph);
      return;
    }

    for (const auto &list_item : nodes.ValueList()) {
      DetachDeleteNode(list_item, graph);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
