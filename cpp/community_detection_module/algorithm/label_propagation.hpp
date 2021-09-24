#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace LabelRankT {

class LabelRankT {
 private:
  /// default edge weight
  double DEFAULT_WEIGHT = 1;
  /// weight property name
  std::string weight_property;

  /// default self-loop weight
  double w_selfloop;
  /// similarity threshold used in the node selection step, values in [0, 1]
  double similarity_threshold;
  /// exponent used in the inflation step
  double exponent;
  /// smallest acceptable probability in the cutoff step
  double min_value;

  /// maximum number of iterations
  std::uint64_t max_iterations = 100;
  /// maximum number of updates for any node
  std::uint64_t max_updates = 5;

  /// reference to current graph
  std::unique_ptr<mg_graph::Graph<>>& graph;

  /// map containing each node’s community label probabilities
  std::unordered_map<std::uint64_t, std::unordered_map<std::uint64_t, double>>
      label_Ps;
  /// sum of weights of each node’s in-edges
  std::unordered_map<std::uint64_t, double> sum_w;
  /// how many times each node has been updated
  std::unordered_map<std::uint64_t, std::uint64_t> times_updated;

  /// flag whether community labels have already been calculated
  bool calculated = false;

  ///@brief Initializes internal data structures.
  void set_structures(std::uint64_t node_id);

  double get_weight(std::uint64_t node_i_id, std::uint64_t node_j_id);

  ///@brief Removes deleted nodes’ entries from internal data structures.
  ///
  ///@param deleted_nodes_ids -- set of deleted nodes’ IDs
  void remove_deleted_nodes(
      std::unordered_set<std::uint64_t>& deleted_nodes_ids);

  ///@brief Resets each node’s number of updates.
  void reset_times_updated();

  ///@return -- given node’s current community label
  std::uint64_t get_label(std::uint64_t node_id);

  ///@return -- given node’s most probable community labels
  std::vector<std::uint64_t> most_probable_labels(std::uint64_t node_id);

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
  bool distinct_enough(mg_graph::Node<uint64_t> node_i,
                       double similarity_threshold);

  ///@brief Performs label propagation on given node’s label probability vector.
  /// Label propagation works by calculating a weighted sum of the label
  /// probability vectors of the node’s neighbors.
  ///
  ///@param node_i -- given node
  ///
  ///@return -- updated label probabilities for given node
  std::unordered_map<std::uint64_t, double> propagate(
      mg_graph::Node<uint64_t> node_i);

  ///@brief Raises given node’s label probabilities to specified power and
  /// normalizes the result.
  ///
  ///@param node_label_Ps -- given node’s label probabilities
  ///@param exponent -- smallest acceptable value
  ///
  ///@return -- updated label probabilities for given node
  void inflate(std::unordered_map<std::uint64_t, double>& node_label_Ps,
               double exponent);

  ///@brief Removes values under a set threshold from given node’s label
  /// probability vector.
  ///
  ///@param node_label_Ps -- given node’s label probabilities
  ///@param min_value -- smallest acceptable value
  ///
  ///@return -- updated label probabilities for given node
  void cutoff(std::unordered_map<std::uint64_t, double>& node_label_Ps,
              double min_value);

  ///@brief Performs an iteration of the LabelRankT algorithm.
  /// Each iteration has four steps: node selection, label propagation,
  /// inflation and cutoff.
  ///
  ///@param incremental -- whether the algorithm is ran incrementally
  ///@param changed_nodes -- the set of changed nodes (for incremental update)
  ///
  ///@return -- {whether no nodes have been updated, maximum number of updates
  /// of any node so far} pair
  std::pair<bool, std::uint64_t> iteration(
      bool incremental = false,
      std::unordered_set<std::uint64_t> changed_nodes = {});

 public:
  ///@brief Creates an instance of the LabelRankT algorithm with associated data
  /// structures.
  ///
  ///@param graph -- reference to current graph
  ///@param w_selfloop -- default weight of self-loops
  ///@param similarity_threshold -- similarity threshold used in the node
  /// selection step, values in [0, 1]
  ///@param exponent -- exponent used in the inflation step
  ///@param min_value -- smallest acceptable probability in the cutoff step
  LabelRankT(std::unique_ptr<mg_graph::Graph<>>& graph,
             std::string weight_property = "weight", double w_selfloop = 1,
             double similarity_threshold = 0.7, double exponent = 4,
             double min_value = 0.1)
      : weight_property(weight_property),
        w_selfloop(w_selfloop),
        similarity_threshold(similarity_threshold),
        exponent(exponent),
        min_value(min_value),
        graph(graph){};

  ///@brief Returns previously calculated community labels.
  /// If no calculation has been done previously, calculates community labels
  /// with default parameter values.
  ///
  ///@return -- {node id, community label} pairs
  std::unordered_map<std::uint64_t, std::uint64_t> get_labels();

  ///@brief Calculates community labels with LabelRankT.
  ///
  ///@param max_iterations -- maximum number of iterations
  ///@param max_updates -- maximum number of updates for any node
  ///@param changed_nodes -- list of changed nodes (for incremental update)
  ///@param to_delete -- list of deleted nodes (for incremental update)
  ///
  ///@return -- {node id, community label} pairs
  std::unordered_map<std::uint64_t, std::uint64_t> calculate_labels(
      std::uint64_t max_iterations = 100, std::uint64_t max_updates = 5,
      std::unordered_set<std::uint64_t> changed_nodes = {},
      std::unordered_set<std::uint64_t> to_delete = {});

  ///@brief Updates changed nodes’ community labels with LabelRankT.
  /// The maximum numbers of iterations and updates are reused from previous
  /// calculate_labels() calls. If no calculation has been done previously,
  /// calculates community labels with default parameter values.
  ///
  ///@param updated_nodes -- list of updated (added, modified) nodes
  ///@param updated_edges -- list of updated edges
  ///@param deleted_nodes -- list of deleted nodes
  ///@param deleted_edges -- list of deleted edges
  ///
  ///@return -- {node id, community label} pairs
  std::unordered_map<std::uint64_t, std::uint64_t> update_labels(
      std::vector<std::uint64_t> modified_nodes,
      std::vector<std::pair<std::uint64_t, std::uint64_t>> modified_edges,
      std::vector<std::uint64_t> deleted_nodes,
      std::vector<std::pair<std::uint64_t, std::uint64_t>> deleted_edges);
};
}  // namespace LabelRankT