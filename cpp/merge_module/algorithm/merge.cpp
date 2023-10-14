#include "merge.hpp"

#include <algorithm>
#include <optional>
#include <string_view>
#include <unordered_map>
#include "mgp.hpp"

namespace {

template <typename T>
concept GraphObject = std::is_same<T, mgp::Node>::value || std::is_same<T, mgp::Relationship>::value;

template <GraphObject NodeOrRel>
bool SameProps(const NodeOrRel &node_or_rel, const mgp::Map &props) {
  for (const auto &[k, v] : props) {
    if (node_or_rel.GetProperty(std::string(k)) != v) {
      return false;
    }
  }
  return true;
}

std::vector<mgp::Relationship> MatchRelationship(const mgp::Node &from, const mgp::Node &to, std::string_view type,
                                                 const mgp::Map &props) {
  std::vector<mgp::Relationship> rels;
  for (const auto rel : from.OutRelationships()) {
    if (rel.To() != to || rel.Type() != type || !SameProps(rel, props)) {
      continue;
    }
    rels.push_back(rel);
  }
  return rels;
}

}  // namespace

void Merge::Relationship(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto graph{mgp::Graph(memgraph_graph)};

    const auto start_node{arguments[0].ValueNode()};
    const auto relationship_type{arguments[1].ValueString()};
    const auto props{arguments[2].ValueMap()};
    const auto create_props{arguments[3].ValueMap()};
    const auto end_node{arguments[4].ValueNode()};
    const auto match_props{arguments[5].ValueMap()};

    if (relationship_type.empty()) {
      throw mgp::ValueException("Relationship type can't be an empty string.");
    }

    auto convert_to_map = [](const mgp::Map &properties) {
      std::unordered_map<std::string_view, mgp::Value> map;
      for (const auto &[k, v] : properties) {
        map.emplace(k, v);
      }
      return map;
    };

    auto props_map = convert_to_map(props);
    auto create_props_map = convert_to_map(create_props);
    auto match_props_map = convert_to_map(match_props);

    auto rels = MatchRelationship(start_node, end_node, relationship_type, props);
    if (!rels.empty()) {
      for (auto &rel : rels) {
        rel.SetProperties(match_props_map);
        auto record = record_factory.NewRecord();
        record.Insert(std::string(kRelationshipResult).c_str(), rel);
      }
      return;
    }

    auto new_rel = graph.CreateRelationship(start_node, end_node, relationship_type);
    new_rel.SetProperties(props_map);
    new_rel.SetProperties(create_props_map);
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kRelationshipResult).c_str(), new_rel);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
