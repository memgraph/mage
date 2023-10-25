#include "algo.hpp"

#include "mgp.hpp"

Algo::PathFinder::PathFinder(const mgp::Node &start_node, const mgp::Node &end_node, int64_t max_length)
    : start_node_(start_node), end_node_id_(end_node.Id()), max_length_(max_length) {}

void Algo::PathFinder::UpdateRelationshipDirection(const mgp::List &relationship_types) {
  rel_direction_.clear();
  for (const auto relationship : relationship_types) {
    auto value = relationship.ValueString();
    auto incoming = static_cast<uint8_t>(value.starts_with('<'));
    auto outgoing = static_cast<uint8_t>(value.ends_with('>'));
    auto substr = value.substr(incoming, value.size() - incoming - outgoing);
    rel_direction_[substr] |= (incoming | (outgoing << 1U));
  }
}

void Algo::PathFinder::DFS(const mgp::Node &curr_node, mgp::Path &curr_path, std::unordered_set<int64_t> &visited) {
  if (curr_node.Id() == end_node_id_) {
    paths_.emplace_back(curr_path);
    return;
  }

  if (static_cast<int64_t>(curr_path.Length()) == max_length_) {
    return;
  }

  visited.insert(curr_node.Id().AsInt());
  std::unordered_set<int64_t> seen;

  auto iterate = [&visited, &seen, &curr_path, this](mgp::Relationships relationships, bool outgoing) {
    uint16_t bitmask = (outgoing ? 2 : 1);

    for (const auto relationship : relationships) {
      auto next_node_id = outgoing ? relationship.To().Id().AsInt() : relationship.From().Id().AsInt();

      if (visited.contains(next_node_id)) {
        continue;
      }

      auto label = std::string(relationship.Type());
      int label_bitmask = 0;

      if (!rel_direction_.empty()) {
        auto it = rel_direction_.find(label);
        if (it == rel_direction_.end()) {
          continue;
        }
        label_bitmask = it->second;
      }

      if (label_bitmask == 0 || label_bitmask == bitmask) {
        curr_path.Expand(relationship);
        DFS(outgoing ? relationship.To() : relationship.From(), curr_path, visited);
        curr_path.Pop();
      } else if (label_bitmask == 3) {
        if (outgoing && seen.contains(relationship.To().Id().AsInt())) {
          curr_path.Expand(relationship);
          DFS(relationship.To(), curr_path, visited);
          curr_path.Pop();
        } else if (!outgoing) {
          seen.insert(relationship.From().Id().AsInt());
        }
      }
    }
  };

  iterate(curr_node.InRelationships(), false);
  iterate(curr_node.OutRelationships(), true);
  visited.erase(curr_node.Id().AsInt());
}

std::vector<mgp::Path> Algo::PathFinder::FindAllPaths() {
  paths_.clear();
  mgp::Path path{start_node_};
  std::unordered_set<int64_t> visited;
  DFS(start_node_, path, visited);
  return paths_;
}

void Algo::AllSimplePaths(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto start_node{arguments[0].ValueNode()};
    const auto end_node{arguments[1].ValueNode()};
    const auto rel_types{arguments[2].ValueList()};
    const auto max_nodes{arguments[3].ValueInt()};

    PathFinder pathfinder{start_node, end_node, max_nodes};
    pathfinder.UpdateRelationshipDirection(rel_types);

    std::vector<mgp::Path> paths = pathfinder.FindAllPaths();

    for (const auto &path : paths) {
      auto record = record_factory.NewRecord();
      record.Insert(std::string(kResultAllSimplePaths).c_str(), path);
    }
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Algo::Cover(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto list_nodes = arguments[0].ValueList();
    std::unordered_set<mgp::Node> nodes;
    for (const auto &elem : list_nodes) {
      auto node = elem.ValueNode();
      nodes.insert(node);
    }

    for (const auto &node : nodes) {
      for (const auto rel : node.OutRelationships()) {
        if (nodes.find(rel.To()) != nodes.end()) {
          auto record = record_factory.NewRecord();
          record.Insert(std::string(kCoverRet1).c_str(), rel);
        }
      }
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
