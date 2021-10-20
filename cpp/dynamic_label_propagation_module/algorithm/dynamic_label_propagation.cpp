#include <cmath>
#include <mg_graph.hpp>

#include "dynamic_label_propagation.hpp"

namespace LabelRankT {
#pragma region helper_methods
///@brief Checks if vector is subset of another vector.
///
///@tparam T -- set element data type
///
///@param v1 -- subset candidate
///@param v2 -- vector against which the check is performed
///
///@return -- whether v1 is subset of v2
template <class T>
bool is_subset(std::vector<T> const& v1, std::vector<T> const& v2) {
  for (const auto& element : v1) {
    if (std::find(v2.begin(), v2.end(), element) == v2.end()) return false;
  }

  return true;
}

std::unordered_set<std::uint64_t> LabelRankT::in_neighbors_memgraph_ids(
    std::uint64_t node_id) {
  std::unordered_set<std::uint64_t> neighbors;

  auto neighbor_set = this->is_directed
                          ? graph->InNeighbours(graph->GetInnerNodeId(node_id))
                          : graph->Neighbours(graph->GetInnerNodeId(node_id));
  for (const auto node_i : neighbor_set) {
    neighbors.insert(graph->GetMemgraphNodeId(node_i.node_id));
  }

  return neighbors;
}

std::vector<std::uint64_t> LabelRankT::nodes_memgraph_ids() {
  std::vector<std::uint64_t> nodes;

  for (const auto node : graph->Nodes()) {
    nodes.push_back(graph->GetMemgraphNodeId(node.id));
  }

  return nodes;
}

void LabelRankT::set_structures(std::uint64_t node_id) {
  times_updated[node_id] = 0;
  std::unordered_map<std::uint64_t, double> node_label_Ps;

  double sum_w_i = w_selfloop;
  for (const auto node_j_id : in_neighbors_memgraph_ids(node_id)) {
    sum_w_i += get_total_weight_between(node_j_id, node_id);
  }

  // add self-loop
  node_label_Ps[node_id] = w_selfloop / sum_w_i;

  // add other edges
  for (const auto node_j_id : in_neighbors_memgraph_ids(node_id)) {
    node_label_Ps[node_j_id] +=
        get_total_weight_between(node_j_id, node_id) / sum_w_i;
  }

  label_Ps[node_id] = node_label_Ps;
  sum_w[node_id] = sum_w_i;
}

void LabelRankT::remove_deleted_nodes(
    std::unordered_set<std::uint64_t>& deleted_nodes_ids) {
  for (const auto node_id : deleted_nodes_ids) {
    label_Ps.erase(node_id);
    sum_w.erase(node_id);
    times_updated.erase(node_id);
  }
}

double LabelRankT::get_total_weight_between(std::uint64_t from_node_id,
                                            std::uint64_t to_node_id) {
  double total_weight = 0;
  for (auto edge_id : graph->GetEdgesBetweenNodes(from_node_id, to_node_id)) {
    auto edge = graph->GetEdge(edge_id);

    // edge direction check
    if (this->is_directed &&
        (edge.from != from_node_id || edge.to != to_node_id))
      continue;

    total_weight += get_weight(edge_id);
  }

  return total_weight;
}

double LabelRankT::get_weight(std::uint64_t edge_id) {
  if (!this->is_weighted) return this->DEFAULT_WEIGHT;

  return graph->GetWeight(edge_id);
}

void LabelRankT::reset_times_updated() {
  for (auto [node_id, _] : times_updated) {
    times_updated[node_id] = 0;
  }
}

std::int64_t LabelRankT::node_label(std::uint64_t node_id) {
  std::int64_t node_label = -1;
  double max_P = 0;

  for (const auto [label, P] : label_Ps[node_id]) {
    if (P > max_P || (P == max_P && (std::int64_t)label < node_label)) {
      max_P = P;
      node_label = label;
    }
  }

  return node_label;
}

std::unordered_map<std::uint64_t, std::int64_t> LabelRankT::graph_labels() {
  std::unordered_map<std::uint64_t, std::int64_t> labels;

  std::set<std::int64_t> labels_ordered;
  std::map<std::int64_t, std::int64_t> lookup;

  for (const auto [node, _] : label_Ps) {
    labels_ordered.insert(node_label(node));
  }

  int64_t label_i = 1;
  for (const auto label : labels_ordered) {
    if (label == -1) {  // no label found
      lookup[label] = label;
    } else {
      lookup[label] = label_i;
      label_i++;
    }
  }

  for (const auto [node, _] : label_Ps) {
    labels.insert({node, lookup[node_label(node)]});
  }

  this->graph.reset();

  return labels;
}

std::vector<std::uint64_t> LabelRankT::most_probable_labels(
    std::uint64_t node_id) {
  double max_P = 0;

  for (const auto [label, P] : label_Ps[node_id]) {
    if (P > max_P) max_P = P;
  }

  std::vector<std::uint64_t> most_probable_labels = {};

  for (const auto [label, P] : label_Ps[node_id]) {
    if (P == max_P) most_probable_labels.push_back(label);
  }

  return most_probable_labels;
}
#pragma endregion helper_methods

#pragma region label_propagation_steps
bool LabelRankT::distinct_enough(std::uint64_t node_i_id,
                                 double similarity_threshold) {
  std::vector<std::uint64_t> labels_i = most_probable_labels(node_i_id);
  std::uint64_t label_similarity = 0;

  auto in_neighbors_ids = in_neighbors_memgraph_ids(node_i_id);
  for (const auto node_j_id : in_neighbors_ids) {
    std::vector<std::uint64_t> labels_j = most_probable_labels(node_j_id);

    if (is_subset(labels_i, labels_j)) label_similarity++;
  }

  int node_i_in_degree = in_neighbors_ids.size();

  return label_similarity <= node_i_in_degree * similarity_threshold;
}

std::unordered_map<std::uint64_t, double> LabelRankT::propagate(
    std::uint64_t node_i_id) {
  std::unordered_map<std::uint64_t, double> new_label_Ps_i;

  // propagate own probabilities (handle self-loops)
  for (const auto [label, P] : label_Ps[node_i_id]) {
    new_label_Ps_i.insert({label, w_selfloop / sum_w[node_i_id] * P});
  }

  // propagate neighbors’ probabilities
  for (const auto node_j_id : in_neighbors_memgraph_ids(node_i_id)) {
    double contribution =
        get_total_weight_between(node_j_id, node_i_id) / sum_w[node_i_id];
    for (const auto [label, P] : label_Ps[node_j_id]) {
      new_label_Ps_i[label] += contribution * P;
    }
  }

  return new_label_Ps_i;
}

void LabelRankT::inflate(
    std::unordered_map<std::uint64_t, double>& node_label_Ps, double exponent) {
  double sum_Ps = 0;

  for (const auto [label, _] : node_label_Ps) {
    double inflated_node_label_Ps = pow(node_label_Ps[label], exponent);
    node_label_Ps[label] = inflated_node_label_Ps;
    sum_Ps += inflated_node_label_Ps;
  }

  for (const auto [label, _] : node_label_Ps) {
    node_label_Ps[label] /= sum_Ps;
  }
}

void LabelRankT::cutoff(
    std::unordered_map<std::uint64_t, double>& node_label_Ps,
    double min_value) {
  std::vector<std::uint64_t> to_be_removed;

  for (const auto [label, P] : node_label_Ps) {
    if (P < min_value) to_be_removed.push_back(label);
  }

  for (const auto label : to_be_removed) {
    node_label_Ps.erase(label);
  }
}

std::pair<bool, std::uint64_t> LabelRankT::iteration(
    bool incremental, std::unordered_set<std::uint64_t> changed_nodes,
    std::unordered_set<std::uint64_t> to_delete) {
  std::unordered_map<std::uint64_t, std::unordered_map<std::uint64_t, double>>
      updated_label_Ps;

  bool none_updated = true;
  std::uint64_t most_updates = 0;

  for (const auto node_id : nodes_memgraph_ids()) {
    if (incremental) {
      bool was_updated = changed_nodes.count(node_id) > 0 ? true : false;
      bool was_deleted = to_delete.count(node_id) > 0 ? true : false;

      if (!was_updated || was_deleted) continue;
    }

    // node selection (a.k.a. conditional update)
    if (!distinct_enough(node_id, similarity_threshold)) continue;
    none_updated = false;

    // label propagation
    std::unordered_map<std::uint64_t, double> updated_node_label_Ps =
        propagate(node_id);

    // inflation
    inflate(updated_node_label_Ps, exponent);

    // cutoff
    cutoff(updated_node_label_Ps, min_value);

    updated_label_Ps.insert({node_id, updated_node_label_Ps});
  }

  for (const auto [node, updated_node_label_Ps] : updated_label_Ps) {
    label_Ps[node] = updated_node_label_Ps;
    times_updated[node]++;

    if (times_updated[node] > most_updates) most_updates = times_updated[node];
  }

  return {none_updated, most_updates};
}
#pragma endregion label_propagation_steps

std::unordered_map<std::uint64_t, std::int64_t> LabelRankT::calculate_labels(
    std::unique_ptr<mg_graph::Graph<>>& graph,
    std::unordered_set<std::uint64_t> changed_nodes,
    std::unordered_set<std::uint64_t> to_delete) {
  this->graph = std::move(graph);

  bool incremental = changed_nodes.size() >= 1 ? true : false;
  if (incremental) {
    remove_deleted_nodes(to_delete);
    for (const auto node_id : changed_nodes) {
      set_structures(node_id);
    }
  } else {
    label_Ps.clear();
    sum_w.clear();
    times_updated.clear();
    for (const auto node_id : nodes_memgraph_ids()) {
      set_structures(node_id);
    }
  }

  for (std::uint64_t i = 0; i < this->max_iterations; i++) {
    auto [none_updated, most_updates] =
        iteration(incremental, changed_nodes, to_delete);
    if (none_updated || most_updates > this->max_updates) break;
  }

  calculated = true;
  reset_times_updated();

  return graph_labels();
}

std::unordered_map<std::uint64_t, std::int64_t> LabelRankT::get_labels(
    std::unique_ptr<mg_graph::Graph<>>& graph) {
  std::unordered_map<std::uint64_t, std::int64_t> labels;
  if (!calculated) {
    return calculate_labels(graph);
  }

  return graph_labels();
}

std::unordered_map<std::uint64_t, std::int64_t> LabelRankT::set_labels(
    std::unique_ptr<mg_graph::Graph<>>& graph, bool is_directed,
    bool is_weighted, double similarity_threshold, double exponent,
    double min_value, std::string weight_property, double w_selfloop,
    std::uint64_t max_iterations, std::uint64_t max_updates) {
  this->is_directed = is_directed;
  this->is_weighted = is_weighted;
  this->similarity_threshold = similarity_threshold;
  this->exponent = exponent;
  this->min_value = min_value;
  this->weight_property = weight_property;
  this->w_selfloop = w_selfloop;
  this->max_iterations = max_iterations;
  this->max_updates = max_updates;

  return calculate_labels(graph);
}

std::unordered_map<std::uint64_t, std::int64_t> LabelRankT::update_labels(
    std::unique_ptr<mg_graph::Graph<>>& graph,
    std::vector<std::uint64_t> modified_nodes,
    std::vector<std::pair<std::uint64_t, std::uint64_t>> modified_edges,
    std::vector<std::uint64_t> deleted_nodes,
    std::vector<std::pair<std::uint64_t, std::uint64_t>> deleted_edges) {
  if (!calculated) return calculate_labels(graph);

  std::unordered_set<std::uint64_t> changed_nodes(modified_nodes.begin(),
                                                  modified_nodes.end());
  std::unordered_set<std::uint64_t> to_delete(deleted_nodes.begin(),
                                              deleted_nodes.end());

  for (const auto edge : modified_edges) {
    changed_nodes.insert(edge.first);
    changed_nodes.insert(edge.second);
  }

  for (const auto edge : deleted_edges) {
    if (to_delete.count(edge.first) == 0) changed_nodes.insert(edge.first);
    if (to_delete.count(edge.second) == 0) changed_nodes.insert(edge.second);
  }

  return calculate_labels(graph, changed_nodes, to_delete);
}
}  // namespace LabelRankT
