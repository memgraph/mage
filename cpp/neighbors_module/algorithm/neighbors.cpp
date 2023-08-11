#include "neighbors.hpp"

#include <fmt/format.h>
#include <unordered_set>

bool Known(const mgp::Node &node, std::vector<std::unordered_set<mgp::Node>> &list) {
  for (auto element : list) {
    if (element.contains(node)) {
      return true;
    }
  }
  return false;
}

void DetermineDirection(mgp::List &rel_types, std::unordered_set<std::string_view> &in_rels,
                        std::unordered_set<std::string_view> &out_rels) {
  if (rel_types.Empty()) {
    rel_types.AppendExtend(mgp::Value(""));
  }

  for (auto rel_type_value : rel_types) {
    auto rel_type = rel_type_value.ValueString();
    if (rel_type[0] == '<' && rel_type[rel_type.size() - 1] == '>') {
      throw mgp::ValueException("Invalid relationship specification!");
    }
    if (rel_type[0] == '<') {
      in_rels.insert(rel_type.substr(1, rel_type.size()));
      continue;
    }
    if (rel_type[rel_type.size() - 1] == '>') {
      out_rels.insert(rel_type.substr(0, rel_type.size() - 1));
      continue;
    }
    in_rels.insert(std::move(rel_type));
    out_rels.insert(std::move(rel_type));
  }
}

void Neighbors::AtHop(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto node = arguments[0].ValueNode();
    auto rel_types = arguments[1].ValueList();
    const auto distance = arguments[2].ValueInt();

    std::unordered_set<std::string_view> in_rels;
    std::unordered_set<std::string_view> out_rels;

    DetermineDirection(rel_types, in_rels, out_rels);

    std::vector<std::unordered_set<mgp::Node>> list;
    std::unordered_set<mgp::Node> set;
    set.insert(node);
    list.push_back(set);

    while (list.size() <= distance) {
      std::unordered_set<mgp::Node> set;
      for (auto node : list.back()) {
        if (!in_rels.empty()) {
          for (auto relationship : node.InRelationships()) {
            if ((in_rels.contains("") || in_rels.contains(relationship.Type())) && !Known(relationship.From(), list)) {
              set.insert(relationship.From());
            }
          }
        }
        if (!out_rels.empty()) {
          for (auto relationship : node.OutRelationships()) {
            if ((out_rels.contains("") || out_rels.contains(relationship.Type())) && !Known(relationship.To(), list)) {
              set.insert(relationship.To());
            }
          }
        }
      }
      if (set.empty()) {
        throw mgp::ValueException(fmt::format("There are no nodes at hop {}.", distance));
      }
      list.push_back(set);
    }

    for (auto node : list.back()) {
      auto record = record_factory.NewRecord();
      record.Insert(std::string(kResultAtHop).c_str(), std::move(node));
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
