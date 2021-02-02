#include "data_structures/graph.hpp"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <map>
#include <set>
#include <vector>

namespace graphdata {

Graph::Graph(bool log_change) { log_change_ = log_change; }

Graph::~Graph() {}

void Graph::Clear() {
  adj_list_.clear();
  nodes_.clear();
  edges_.clear();
  nodes_to_edge_.clear();
  neighbours_.clear();
}

const std::vector<Node> &Graph::Nodes() const { return nodes_; }

const std::vector<Edge> &Graph::Edges() const { return edges_; }

std::vector<Edge> Graph::ExistingEdges() const {
  std::vector<Edge> output;
  output.reserve(edges_.size());
  for (const auto &edge : edges_) {
    if (edge.id == Graph::kDeletedEdgeId) continue;
    output.push_back(edge);
  }
  return output;
}

bool Graph::IsEdgeValid(uint32_t edge_id) const {
  if (edge_id < 0 || edge_id >= edges_.size()) return false;
  if (edges_[edge_id].id == kDeletedEdgeId) return false;
  return true;
}

std::vector<uint32_t> Graph::GetEdgesBetweenNodes(uint32_t first,
                                                  uint32_t second) const {
  typedef std::multimap<std::pair<uint32_t, uint32_t>, uint32_t>::const_iterator
      multimap_itr;
  typedef std::pair<multimap_itr, multimap_itr> multimap_pair;
  std::vector<uint32_t> ret;
  multimap_pair range = nodes_to_edge_.equal_range(
      {std::min(first, second), std::max(first, second)});
  ret.reserve(std::distance(range.first, range.second));
  for (auto it = range.first; it != range.second; ++it) {
    if (!IsEdgeValid(it->second)) continue;
    ret.push_back(it->second);
  }
  return ret;
}

const std::vector<uint32_t> &Graph::IncidentEdges(uint32_t node_id) const {
  assert(node_id >= 0 && node_id < nodes_.size() && "Invalid node id");
  return adj_list_[node_id];
}

const std::vector<Neighbour> &Graph::Neighbours(uint32_t node_id) const {
  assert(node_id >= 0 && node_id < nodes_.size() && "Invalid node id");
  return neighbours_[node_id];
}

const Node &Graph::GetNode(uint32_t node_id) const {
  assert(node_id >= 0 && node_id < nodes_.size() && "Invalid node id");
  return nodes_[node_id];
}

const Edge &Graph::GetEdge(uint32_t edge_id) const {
  assert(IsEdgeValid(edge_id));
  return edges_[edge_id];
}

void Graph::EraseEdge(uint32_t u, uint32_t v) {
  assert(u >= 0 && u < nodes_.size());
  assert(v >= 0 && v < nodes_.size());

  auto it = nodes_to_edge_.find({std::min(u, v), std::max(u, v)});
  if (it == nodes_to_edge_.end()) return;
  uint32_t edge_id = it->second;

  for (auto it = adj_list_[u].begin(); it != adj_list_[u].end(); ++it) {
    if (edges_[*it].to != v && edges_[*it].from != v) continue;
    edges_[*it].id = Graph::kDeletedEdgeId;
    adj_list_[u].erase(it);
    break;
  }
  for (auto it = adj_list_[v].begin(); it != adj_list_[v].end(); ++it) {
    if (edges_[*it].to != u && edges_[*it].from != u) continue;
    edges_[*it].id = Graph::kDeletedEdgeId;
    adj_list_[v].erase(it);
    break;
  }

  for (auto it = neighbours_[u].begin(); it != neighbours_[u].end(); ++it) {
    if (it->edge_id != edge_id) continue;
    neighbours_[u].erase(it);
    break;
  }
  for (auto it = neighbours_[v].begin(); it != neighbours_[v].end(); ++it) {
    if (it->edge_id != edge_id) continue;
    neighbours_[v].erase(it);
    break;
  }
}

uint32_t Graph::CreateNode() {
  uint32_t id = nodes_.size();
  nodes_.push_back({id});
  adj_list_.push_back({});
  neighbours_.push_back({});
  return id;
}

uint32_t Graph::CreateEdge(uint32_t from, uint32_t to) {
  assert(from >= 0 && to >= 0 && from < nodes_.size() && to < nodes_.size() &&
         "Invalid node id");
  uint32_t id = edges_.size();
  edges_.push_back({id, from, to});
  adj_list_[from].push_back(id);
  adj_list_[to].push_back(id);
  neighbours_[from].emplace_back(to, id);
  neighbours_[to].emplace_back(from, id);
  nodes_to_edge_.insert({{std::min(from, to), std::max(from, to)}, id});
  return id;
}
}  // namespace obsdata
