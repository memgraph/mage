#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mgp.hpp>

namespace LabelRankT {
std::unordered_set<std::uint64_t> DEFAULT_EMPTY_SET = {};

class LabelRankT {
 private:
  /* #region parameters */
  /// default relationship weight
  static constexpr double DEFAULT_WEIGHT = 1;

  /// graph directedness
  bool is_directed = false;
  /// whether to use weights
  bool is_weighted = false;

  /// similarity threshold used in the node selection step, values within [0, 1]
  double similarity_threshold = 0.7;
  /// exponent used in the inflation step
  double exponent = 4;
  /// lowest probability at the cutoff step
  double min_value = 0.1;

  /// weight property name
  std::string weight_property = "weight";
  /// default self-loop weight
  double w_selfloop = 1.0;

  /// maximum number of iterations
  std::uint64_t max_iterations = 100;
  /// maximum number of updates for any node
  std::uint64_t max_updates = 5;
  /* #endregion */

  /* #region structures */
  /// map containing each node’s community label probabilities
  std::unordered_map<std::uint64_t, std::unordered_map<std::uint64_t, double>> label_Ps;
  /// sum of weights of each node’s in-relationships
  std::unordered_map<std::uint64_t, double> sum_w;
  /// how many times each node has been updated
  std::unordered_map<std::uint64_t, std::uint64_t> times_updated;

  /// flag whether community labels have already been calculated
  bool calculated = false;
  /* #endregion */

  /* #region helper_methods */
  ///@brief Checks if unordered_set is subset of another unordered_set.
  ///
  ///@tparam T -- set element data type
  ///
  ///@param a -- subset candidate
  ///@param b -- set against which the check is performed
  ///
  ///@return -- whether a is subset of b
  template <class T>
  bool IsSubset(const std::unordered_set<T> &a, const std::unordered_set<T> &b) {
    return std::all_of(a.begin(), a.end(), [&](const T &value) { return b.find(value) != b.end(); });
  }

  ///@brief Returns in-neighbors’ Memgraph IDs for a given node’s Memgraph ID.
  /// If the graph is undirected, all neighbors are included.
  ///
  ///@param node_id -- given node’s Memgraph ID
  ///
  ///@return -- vector of neighbors’ Memgraph IDs
  std::unordered_set<std::uint64_t> InNeighborsIDs(const mgp::Graph &graph, std::uint64_t node_id) const {
    std::unordered_set<std::uint64_t> neighbors;

    for (const auto &in_relationship : graph.GetNodeById(mgp::Id::FromUint(node_id)).InRelationships()) {
      const auto neighbor_id = in_relationship.From().Id().AsUint();
      neighbors.insert(neighbor_id);
    }
    if (is_directed) {
      return neighbors;
    }

    for (const auto &out_relationship : graph.GetNodeById(mgp::Id::FromUint(node_id)).OutRelationships()) {
      const auto neighbor_id = out_relationship.To().Id().AsUint();
      neighbors.insert(neighbor_id);
    }
    return neighbors;
  }

  ///@brief Initializes the internal data structures.
  void InitializeStructures(const mgp::Graph &graph, std::uint64_t node_id) {
    times_updated[node_id] = 0;
    std::unordered_map<std::uint64_t, double> node_label_Ps;

    const auto in_neighbors = InNeighborsIDs(graph, node_id);

    double sum_w_i = w_selfloop;
    for (const auto node_j_id : in_neighbors) {
      sum_w_i += GetTotalWeightBetween(graph, node_j_id, node_id);
    }

    // add self-loop
    node_label_Ps[node_id] = w_selfloop / sum_w_i;

    // add other relationships
    for (const auto node_j_id : in_neighbors) {
      node_label_Ps[node_j_id] += GetTotalWeightBetween(graph, node_j_id, node_id) / sum_w_i;
    }

    label_Ps[node_id] = std::move(node_label_Ps);
    sum_w[node_id] = sum_w_i;
  }

  ///@brief Removes deleted nodes’ entries from the internal data structures.
  ///
  ///@param deleted_nodes_ids -- set of deleted nodes’ IDs
  void RemoveDeletedNodes(const std::unordered_set<std::uint64_t> &deleted_nodes_ids) {
    for (const auto node_id : deleted_nodes_ids) {
      label_Ps.erase(node_id);
      sum_w.erase(node_id);
      times_updated.erase(node_id);
    }
  }

  ///@brief Retrieves the total weight of relationships between two nodes.
  /// With undirected graphs it considers all relationships between the given node pair.
  /// On a directed graph, it checks if the source and target nodes match.
  ///
  ///@param from_node_id -- source node’s Memgraph ID
  ///@param to_node_id -- target node’s Memgraph ID
  ///
  ///@return -- total weight of relationships between from_node_id and to_node_id
  double GetTotalWeightBetween(const mgp::Graph &graph, std::uint64_t from_node_id, std::uint64_t to_node_id) const {
    double total_weight = 0;

    for (const auto &out_relationship : graph.GetNodeById(mgp::Id::FromUint(from_node_id)).OutRelationships()) {
      if (out_relationship.To().Id().AsUint() == to_node_id) {
        total_weight += GetWeight(graph, out_relationship);
      }
    }
    if (is_directed) {
      return total_weight;
    }

    for (const auto &in_relationship : graph.GetNodeById(mgp::Id::FromUint(from_node_id)).InRelationships()) {
      if (in_relationship.From().Id().AsUint() == to_node_id) {
        total_weight += GetWeight(graph, in_relationship);
      }
    }

    return total_weight;
  }

  ///@brief Retrieves given relationship’s weight.
  double GetWeight(const mgp::Graph &graph, const mgp::Relationship &relationship) const {
    if (!is_weighted) return DEFAULT_WEIGHT;

    try {
      const auto maybe_weight = relationship[weight_property];
      return maybe_weight.IsNumeric() ? maybe_weight.ValueNumeric() : DEFAULT_WEIGHT;
    } catch (const std::exception &e) {  // default if no such property exists
      return DEFAULT_WEIGHT;
    }
  }

  ///@brief Resets each node’s number of updates.
  void ResetTimesUpdated() {
    for (auto &[_, count] : times_updated) {
      count = 0;
    }
  }

  /// Returns given node’s current community label.
  /// If no label was found (e.g. with the min_value set too high),
  /// the assigned label number is -1.
  ///
  /// @return -- given node’s current community label
  std::int64_t NodeLabel(std::uint64_t node_id) {
    std::int64_t node_label = -1;
    double max_P = 0;

    for (const auto [label, P] : label_Ps[node_id]) {
      if (P > max_P || (P == max_P && static_cast<std::int64_t>(label) < node_label)) {
        max_P = P;
        node_label = label;
      }
    }

    return node_label;
  }

  /// Returns all nodes’ current community labels.
  //
  /// Community label numbers are initially derived from Memgraph’s node IDs. As
  /// those grow larger with graph updates, this method renumbers them so that
  /// they begin with 1.
  std::unordered_map<std::uint64_t, std::int64_t> AllLabels() {
    std::unordered_map<std::uint64_t, std::int64_t> labels;

    std::set<std::int64_t> labels_ordered;
    std::unordered_map<std::int64_t, std::int64_t> lookup;

    for (const auto &[node, _] : label_Ps) {
      labels_ordered.insert(NodeLabel(node));
    }

    int64_t label_i = 1;
    labels_ordered.erase(-1);
    for (const auto label : labels_ordered) {
      lookup[label] = label_i;
      label_i++;
    }
    lookup[-1] = -1;

    for (const auto &[node, _] : label_Ps) {
      labels.insert({node, lookup[NodeLabel(node)]});
    }

    return labels;
  }

  ///@return -- given node’s most probable community labels
  std::unordered_set<std::uint64_t> MostProbableLabels(std::uint64_t node_id) {
    double max_P = 0;

    for (const auto [label, P] : label_Ps[node_id]) {
      if (P > max_P) max_P = P;
    }

    std::unordered_set<std::uint64_t> most_probable_labels;

    for (const auto [label, P] : label_Ps[node_id]) {
      if (P == max_P) most_probable_labels.insert(label);
    }

    return most_probable_labels;
  }
  /* #endregion */

  /* #region label_propagation_steps */
  ///@brief Checks if given node’s label probabilities are sufficiently distinct
  /// from its neighbors’.
  /// For a label probability vector to be considered sufficiently distinct, it
  /// has to be a subset of less than k% label probabilty vectors of its
  /// neighbors.
  ///
  ///@param node_i -- given node
  ///@param similarity_threshold -- the k% threshold above
  ///
  ///@return -- whether the criterion is met
  bool DistinctEnough(const mgp::Graph &graph, std::uint64_t node_i_id, double similarity_threshold) {
    const auto labels_i = MostProbableLabels(node_i_id);
    std::uint64_t label_similarity = 0;

    const auto in_neighbors_ids = InNeighborsIDs(graph, node_i_id);
    for (const auto node_j_id : in_neighbors_ids) {
      const auto labels_j = MostProbableLabels(node_j_id);
      if (IsSubset(labels_i, labels_j)) label_similarity++;
    }

    int node_i_in_degree = in_neighbors_ids.size();

    return label_similarity <= node_i_in_degree * similarity_threshold;
  }

  ///@brief Performs label propagation on given node’s label probability vector.
  /// Label propagation works by calculating a weighted sum of the label
  /// probability vectors of the node’s neighbors.
  ///
  ///@param node_i -- given node
  ///
  ///@return -- updated label probabilities for given node
  std::unordered_map<std::uint64_t, double> Propagate(const mgp::Graph &graph, std::uint64_t node_i_id) {
    std::unordered_map<std::uint64_t, double> new_label_Ps_i;

    // propagate own probabilities (handle self-loops)
    for (const auto [label, P] : label_Ps[node_i_id]) {
      new_label_Ps_i.insert({label, w_selfloop / sum_w[node_i_id] * P});
    }

    // propagate neighbors’ probabilities
    for (const auto node_j_id : InNeighborsIDs(graph, node_i_id)) {
      const double contribution = GetTotalWeightBetween(graph, node_j_id, node_i_id) / sum_w[node_i_id];
      for (const auto [label, P] : label_Ps[node_j_id]) {
        new_label_Ps_i[label] += contribution * P;
      }
    }

    return new_label_Ps_i;
  }

  ///@brief Raises given node’s label probabilities to specified power and
  /// normalizes the result.
  ///
  ///@param node_label_Ps -- given node’s label probabilities
  ///@param exponent -- smallest acceptable value
  ///
  ///@return -- updated label probabilities for given node
  void Inflate(std::unordered_map<std::uint64_t, double> &node_label_Ps, double exponent) const {
    double sum_Ps = 0;

    for (const auto [label, _] : node_label_Ps) {
      const double inflated_node_label_Ps = pow(node_label_Ps[label], exponent);
      node_label_Ps[label] = inflated_node_label_Ps;
      sum_Ps += inflated_node_label_Ps;
    }

    for (const auto [label, _] : node_label_Ps) {
      node_label_Ps[label] /= sum_Ps;
    }
  }

  ///@brief Removes values under a set threshold from given node’s label
  /// probability vector.
  ///
  ///@param node_label_Ps -- given node’s label probabilities
  ///@param min_value -- smallest acceptable value
  ///
  ///@return -- updated label probabilities for given node
  void Cutoff(std::unordered_map<std::uint64_t, double> &node_label_Ps, double min_value) const {
    for (auto node_label_P = node_label_Ps.begin(); node_label_P != node_label_Ps.end();) {
      auto P = node_label_P->second;
      if (P < min_value)
        node_label_P = node_label_Ps.erase(node_label_P);
      else
        node_label_P++;
    }
  }

  ///@brief Performs an iteration of the LabelRankT algorithm.
  /// Each iteration has four steps: node selection, label propagation,
  /// inflation and cutoff.
  ///
  ///@param incremental -- whether the algorithm is ran incrementally
  ///@param affected_nodes -- the set of affected nodes (for incremental update)
  ///
  ///@return -- {whether no nodes have been updated, maximum number of updates
  /// of any node so far} pair
  std::pair<bool, std::uint64_t> Iteration(const mgp::Graph &graph, bool incremental = false,
                                           const std::unordered_set<std::uint64_t> &affected_nodes = {},
                                           const std::unordered_set<std::uint64_t> &deleted_nodes = {}) {
    std::unordered_map<std::uint64_t, std::unordered_map<std::uint64_t, double>> updated_label_Ps;

    bool none_updated = true;
    std::uint64_t most_updates = 0;

    for (const auto &[node_id, _] : this->sum_w) {
      if (incremental) {
        const bool was_updated = affected_nodes.count(node_id) > 0;
        const bool was_deleted = deleted_nodes.count(node_id) > 0;

        if (!was_updated || was_deleted) continue;
      }

      // node selection (a.k.a. conditional update)
      if (!DistinctEnough(graph, node_id, similarity_threshold)) continue;
      none_updated = false;

      // label propagation
      std::unordered_map<std::uint64_t, double> updated_node_label_Ps = Propagate(graph, node_id);

      // inflation
      Inflate(updated_node_label_Ps, exponent);

      // cutoff
      Cutoff(updated_node_label_Ps, min_value);

      updated_label_Ps.insert({node_id, std::move(updated_node_label_Ps)});
    }

    for (const auto &[node, updated_node_label_Ps] : updated_label_Ps) {
      label_Ps[node] = updated_node_label_Ps;
      times_updated[node]++;

      if (times_updated[node] > most_updates) most_updates = times_updated[node];
    }

    return {none_updated, most_updates};
  }
  /* #endregion */

  ///@brief Handles calculation of community labels and associated data
  /// structures for both dynamic and non-dynamic uses of the algorithm.
  ///
  ///@param graph -- current graph
  ///@param affected_nodes -- list of affected nodes (for incremental update)
  ///@param deleted_nodes -- list of deleted nodes (for incremental update)
  ///@param persist -- whether to store results
  ///
  ///@return -- {node id, community label} pairs
  std::unordered_map<std::uint64_t, std::int64_t> CalculateLabels(
      const mgp::Graph &graph, const std::unordered_set<std::uint64_t> &affected_nodes = {},
      const std::unordered_set<std::uint64_t> &deleted_nodes = {}, bool persist = true) {
    const bool incremental = affected_nodes.size() >= 1;
    if (incremental) {
      RemoveDeletedNodes(deleted_nodes);
      for (const auto node_id : affected_nodes) {
        InitializeStructures(graph, node_id);
      }
    } else {
      label_Ps.clear();
      sum_w.clear();
      times_updated.clear();
      for (const auto node : graph.Nodes()) {
        InitializeStructures(graph, node.Id().AsUint());
      }
    }

    for (std::uint64_t i = 0; i < max_iterations; i++) {
      const auto [none_updated, most_updates] = Iteration(graph, incremental, affected_nodes, deleted_nodes);
      if (none_updated || most_updates > max_updates) break;
    }

    if (persist) calculated = true;
    ResetTimesUpdated();

    return AllLabels();
  }

 public:
  ///@brief Creates an instance of the LabelRankT algorithm.
  LabelRankT() = default;

  ///@brief Returns previously calculated community labels.
  /// If no calculation has been done previously, calculates community labels
  /// with default parameter values.
  ///
  ///@param graph -- current graph
  ///
  ///@return -- {node id, community label} pairs
  std::unordered_map<std::uint64_t, std::int64_t> GetLabels(const mgp::Graph &graph) {
    if (!calculated) return CalculateLabels(graph, {}, {}, false);

    return AllLabels();
  }

  ///@brief Calculates and returns community labels using LabelRankT. The labels
  /// and the parameters for their calculation are reused in online
  /// community detection with UpdateLabels().
  ///
  ///@param graph -- current graph
  ///@param is_weighted -- whether graph is directed
  ///@param is_weighted -- whether graph is weighted
  ///@param similarity_threshold -- similarity threshold used in the node
  /// selection step, values in [0, 1]
  ///@param exponent -- exponent used in the inflation step
  ///@param min_value -- smallest acceptable probability in the cutoff step
  ///@param weight_property -- weight-containing relationship property’s name
  ///@param w_selfloop -- default weight of self-loops
  ///@param max_iterations -- maximum number of iterations
  ///@param max_updates -- maximum number of updates for any node
  std::unordered_map<std::uint64_t, std::int64_t> SetLabels(const mgp::Graph &graph, bool is_directed = false,
                                                            bool is_weighted = false, double similarity_threshold = 0.7,
                                                            double exponent = 4.0, double min_value = 0.1,
                                                            std::string weight_property = "weight",
                                                            double w_selfloop = 1.0, std::uint64_t max_iterations = 100,
                                                            std::uint64_t max_updates = 5) {
    this->is_directed = is_directed;
    this->is_weighted = is_weighted;
    this->similarity_threshold = similarity_threshold;
    this->exponent = exponent;
    this->min_value = min_value;
    this->weight_property = std::move(weight_property);
    this->w_selfloop = w_selfloop;
    this->max_iterations = max_iterations;
    this->max_updates = max_updates;

    return CalculateLabels(graph);
  }

  ///@brief Updates affected nodes’ community labels with LabelRankT.
  /// The maximum numbers of iterations and updates are reused from previous
  /// CalculateLabels() calls. If no calculation has been done previously,
  /// calculates community labels with default parameter values.
  ///
  ///@param graph -- current graph
  ///@param modified_nodes -- list of modified (created, updated) nodes
  ///@param modified_relationships -- list of modified (created, updated) relationships
  ///@param deleted_nodes -- list of deleted nodes
  ///@param deleted_relationships -- list of deleted relationships
  ///
  ///@return -- {node id, community label} pairs
  std::unordered_map<std::uint64_t, std::int64_t> UpdateLabels(
      const mgp::Graph &graph, std::unordered_set<std::uint64_t> &modified_nodes = DEFAULT_EMPTY_SET,
      const std::vector<std::pair<std::uint64_t, std::uint64_t>> &modified_relationships = {},
      const std::unordered_set<std::uint64_t> &deleted_nodes = {},
      const std::vector<std::pair<std::uint64_t, std::uint64_t>> &deleted_relationships = {}) {
    if (!calculated) return CalculateLabels(graph, {}, {}, false);

    for (const auto &[from, to] : modified_relationships) {
      modified_nodes.insert(from);
      modified_nodes.insert(to);
    }

    for (const auto &[from, to] : deleted_relationships) {
      if (deleted_nodes.count(from) == 0) modified_nodes.insert(from);
      if (deleted_nodes.count(to) == 0) modified_nodes.insert(to);
    }

    return CalculateLabels(graph, modified_nodes, deleted_nodes);
  }
};
}  // namespace LabelRankT
