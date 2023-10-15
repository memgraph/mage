#include "neighbors.hpp"

#include <fmt/format.h>
#include <list>
#include <string_view>
#include <unordered_set>

Neighbors::Config::Config(const mgp::List &list_of_relationships) {
  if (list_of_relationships.Size() ==
      0) {  // if no relationships were passed as arguments, all relationships are allowed
    any_outgoing = true;
    any_incoming = true;
    return;
  }

  for (const auto &rel : list_of_relationships) {
    std::string rel_type{std::string(rel.ValueString())};
    bool starts_with = rel_type.starts_with('<');
    bool ends_with = rel_type.ends_with('>');

    if (rel_type.size() == 1) {
      if (starts_with) {
        any_incoming = true;
      } else if (ends_with) {
        any_outgoing = true;
      } else {
        rel_direction[rel_type] = RelDirection::kAny;
      }
      continue;
    }

    if (starts_with && ends_with) {  // <type>
      rel_direction[rel_type.substr(1, rel_type.size() - 2)] = RelDirection::kBoth;
    } else if (starts_with) {  // <type
      rel_direction[rel_type.substr(1)] = RelDirection::kIncoming;
    } else if (ends_with) {  // type>
      rel_direction[rel_type.substr(0, rel_type.size() - 1)] = RelDirection::kOutgoing;
    } else {  // type
      rel_direction[rel_type] = RelDirection::kAny;
    }
  }
}

Neighbors::RelDirection Neighbors::Config::GetDirection(std::string_view rel_type) {
  auto it = rel_direction.find(rel_type);
  if (it == rel_direction.end()) {
    return RelDirection::kNone;
  }
  return it->second;
}

bool Known(const mgp::Node &node, std::list<std::unordered_set<mgp::Node>> &list) {
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
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto node = arguments[0].ValueNode();
    auto rel_types = arguments[1].ValueList();
    const auto distance = arguments[2].ValueInt();

    std::unordered_set<std::string_view> in_rels;
    std::unordered_set<std::string_view> out_rels;

    DetermineDirection(rel_types, in_rels, out_rels);

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

void Neighbors::ByHop(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto node = arguments[0].ValueNode();
    auto rel_types = arguments[1].ValueList();
    const auto distance = arguments[2].ValueInt();

    std::unordered_set<std::string_view> in_rels;
    std::unordered_set<std::string_view> out_rels;

    DetermineDirection(rel_types, in_rels, out_rels);

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

void Neighbors::ToHop(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto node{arguments[0].ValueNode()};
    const auto rel_types{arguments[1].ValueList()};
    const auto distance{arguments[2].ValueInt()};

    Config config{rel_types};

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}