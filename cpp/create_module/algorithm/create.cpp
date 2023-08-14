#include "create.hpp"

#include <unordered_set>
#include "mgp.hpp"

namespace {
void ThrowException(const mgp::Value &value) {
  std::ostringstream oss;
  oss << value.Type();
  throw mgp::ValueException("Unsupported type for this operation, received type: " + oss.str());
}

void DeleteAndOutput(mgp::Relationship &relationship, const mgp::List &keys, const mgp::RecordFactory &record_factory) {
  for (const auto &key : keys) {
    relationship.RemoveProperty(std::string(key.ValueString()));
  }

  auto record = record_factory.NewRecord();
  record.Insert(std::string(Create::kResultRemoveRelProperties).c_str(), relationship);
}

}  // namespace

void Create::RemoveRelProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    mgp::Graph graph{memgraph_graph};
    const auto keys{arguments[1].ValueList()};

    std::unordered_set<int64_t> ids;

    auto ParseValue = [&](const mgp::Value &value) {
      if (value.IsRelationship()) {
        auto relationship_copy = value.ValueRelationship();
        DeleteAndOutput(relationship_copy, keys, record_factory);
      } else if (value.IsInt()) {
        ids.insert(value.ValueInt());
      } else {
        ThrowException(value);
      }
    };

    if (!arguments[0].IsList()) {
      ParseValue(arguments[0]);
    } else {
      for (const auto &list_item : arguments[0].ValueList()) {
        ParseValue(list_item);
      }
    }

    if (ids.empty()) {
      return;
    }

    for (auto relationship : graph.Relationships()) {
      if (ids.contains(relationship.Id().AsInt())) {
        DeleteAndOutput(relationship, keys, record_factory);
      }
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Create::Relationship(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    auto node_from{arguments[0].ValueNode()};
    const auto relationship_type{arguments[1].ValueString()};
    const auto properties{arguments[2].ValueMap()};
    auto node_to{arguments[3].ValueNode()};

    mgp::Graph graph{memgraph_graph};
    auto relationship = graph.CreateRelationship(node_from, node_to, relationship_type);

    for (const auto &map_item : properties) {
      relationship.SetProperty(std::string(map_item.key), map_item.value);
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRelationship).c_str(), relationship);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
