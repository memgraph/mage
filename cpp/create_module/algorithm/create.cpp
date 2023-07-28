#include "create.hpp"
#include <iterator>
#include <sstream>
#include <unordered_set>
#include "mgp.hpp"

namespace {
const std::unordered_set<int64_t> get_relationship_ids(const mgp::Value &relationships) {
  std::unordered_set<int64_t> ids;
  if (relationships.IsRelationship()) {
    ids.insert(relationships.ValueRelationship().Id().AsInt());
  } else if (relationships.IsInt()) {
    ids.insert(relationships.ValueInt());
  } else if (relationships.IsList()) {
    const auto list{relationships.ValueList()};
    for (const auto &list_item : list) {
      if (list_item.IsRelationship()) {
        ids.insert(list_item.ValueRelationship().Id().AsInt());
      } else if (list_item.IsInt()) {
        ids.insert(list_item.ValueInt());
      } else {
        std::ostringstream oss;
        oss << list_item.Type();
        throw mgp::ValueException("Unsupported type for this operation, received type: " + oss.str());
      }
    }
  }
  return ids;
}
}  // namespace

void Create::SetRelProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    mgp::Graph graph{memgraph_graph};
    const auto keys{arguments[1].ValueList()};
    const auto values{arguments[2].ValueList()};

    auto all_relationships = graph.Relationships();
    const auto ids = get_relationship_ids(arguments[0]);

    if (keys.Size() != values.Size()) {
      throw mgp::IndexException("Keys and values are not the same size");
    }

    for (auto relationship : all_relationships) {
      if (ids.contains(relationship.Id().AsInt())) {
        auto it1 = keys.begin();
        auto it2 = values.begin();

        while (it1 != keys.end() && it2 != values.end()) {
          relationship.SetProperty(std::string((*it1).ValueString()), *it2);
          ++it1;
          ++it2;
        }
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultSetRelProperties).c_str(), false);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
