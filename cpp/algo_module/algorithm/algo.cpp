#include "algo.hpp"

#include "mgp.hpp"

Algo::PathFinder::PathFinder(const mgp::Node &start_node, const mgp::Node &end_node, int64_t max_length)
    : _start_node(start_node), _end_node_id(end_node.Id()), _max_length(max_length) {}

void Algo::PathFinder::UpdateRelationshipDirection(const mgp::List &relationship_types) {
  _rel_direction.clear();
  for (const auto relationship : relationship_types) {
    auto value = relationship.ValueString();
    auto incoming = static_cast<uint8_t>(value.starts_with('<'));
    auto outgoing = static_cast<uint8_t>(value.ends_with('>'));
    auto substr = value.substr(incoming, value.size() - incoming - outgoing);
    _rel_direction[substr] |= (incoming | (outgoing << 1U));
  }
}

void Algo::PathFinder::DFS(const mgp::Node &curr_node, mgp::Path &curr_path, std::unordered_set<int64_t> &visited) {
  if (curr_node.Id() == _end_node_id) {
    _paths.emplace_back(curr_path);
    return;
  }

  if (static_cast<int64_t>(curr_path.Length()) == _max_length) {
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

      if (!_rel_direction.empty()) {
        auto it = _rel_direction.find(label);
        if (it == _rel_direction.end()) {
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
  _paths.clear();
  mgp::Path path{_start_node};
  std::unordered_set<int64_t> visited;
  DFS(_start_node, path, visited);
  return _paths;
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