#include "neighbors.hpp"

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

void Neighbors::ByHop(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto node = arguments[0].ValueNode();
    auto input_rel_type = arguments[1].ValueString();
    const auto distance = arguments[2].ValueInt();

    std::unordered_set<std::string_view> rel_types;
    auto i{0};
    auto j{input_rel_type.find("|")};
    while (j < input_rel_type.size()) {
      rel_types.insert(input_rel_type.substr(i, j - i));
      i = j + 1;
      j = input_rel_type.find("|", i);
    }
    rel_types.insert(input_rel_type.substr(i));

    std::unordered_set<std::string_view> in_rels;
    std::unordered_set<std::string_view> out_rels;

    for (auto rel_type : rel_types) {
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

    std::list<std::unordered_set<mgp::Node>> list;
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
      list.push_back(set);
    }

    list.pop_front();
    for (auto set_element : list) {
      mgp::List return_list;
      for (auto node_element : set_element) {
        return_list.AppendExtend(mgp::Value(std::move(node_element)));
      }
      auto record = record_factory.NewRecord();
      record.Insert(std::string(kResultByHop).c_str(), return_list);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
