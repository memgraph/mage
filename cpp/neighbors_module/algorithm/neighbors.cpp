#include "neighbors.hpp"

#include <fmt/format.h>
#include <list>
#include <unordered_set>

bool Known(const mgp::Node &node, std::list<std::unordered_set<mgp::Node>> &list) {
  for (auto element : list) {
    if (element.contains(node)) {
      return true;
    }
  }
  return false;
}

void Neighbors::AtHop(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto node = arguments[0].ValueNode();
    auto rel_type = arguments[1].ValueString();
    const auto distance = arguments[2].ValueInt();

    if (rel_type[0] == '<' && rel_type[rel_type.size() - 1] == '>') {
      throw mgp::ValueException("Invalid relationship specification!");
    }
    bool in = (rel_type[0] == '<');
    bool out = (rel_type[rel_type.size() - 1] == '>');
    if (in) {
      rel_type = rel_type.substr(1, rel_type.size());
    }
    if (out) {
      rel_type = rel_type.substr(0, rel_type.size() - 1);
    }
    if (!in && !out) {
      in = out = 1;
    }

    std::list<std::unordered_set<mgp::Node>> list;
    std::unordered_set<mgp::Node> set;
    set.insert(node);
    list.push_back(set);

    while (list.size() <= distance) {
      std::unordered_set<mgp::Node> set;
      for (auto node : list.back()) {
        if (in) {
          for (auto relationship : node.InRelationships()) {
            if ((rel_type == "" || relationship.Type() == rel_type) && !Known(relationship.From(), list)) {
              set.insert(relationship.From());
            }
          }
        }
        if (out) {
          for (auto relationship : node.OutRelationships()) {
            if ((rel_type == "" || relationship.Type() == rel_type) && !Known(relationship.To(), list)) {
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
