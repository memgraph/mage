#pragma once

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace LabelRankT {

class LabelRankT {
 private:
#pragma region parameters
  /// default edge weight
  double DEFAULT_WEIGHT = 1;
  /// weight property name
  std::string weight_property = "weight";

  /// default self-loop weight
  double w_selfloop = 1;
  /// similarity threshold used in the node selection step, values within [0, 1]
  double similarity_threshold = 0.7;
  /// exponent used in the inflation step
  double exponent = 4;
  /// smallest acceptable probability in the cutoff step
  double min_value = 0.1;

  /// maximum number of iterations
  std::uint64_t max_iterations = 100;
  /// maximum number of updates for any node
  std::uint64_t max_updates = 5;
#pragma endregion parameters

#pragma region structures
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
#pragma endregion structures

#pragma region helper_methods
  ///@brief Returns a vector of all graph’s nodes’ Memgraph IDs.
  ///
  ///@return -- vector of all graph’s nodes’ Memgraph IDs
  std::vector<std::uint64_t> nodes_memgraph_ids();

  ///@brief Returns neighbors’ Memgraph IDs for a given node’s Memgraph ID.
  ///
  ///@param node_id -- given node’s Memgraph ID
  ///
  ///@return -- vector of neighbors’ Memgraph IDs
  std::vector<std::uint64_t> neighbors_memgraph_ids(std::uint64_t node_id);

  ///@brief Initializes internal data structures.
  void set_structures(std::uint64_t node_id);

  ///@brief Removes deleted nodes’ entries from internal data structures.
  ///
  ///@param deleted_nodes_ids -- set of deleted nodes’ IDs
  void remove_deleted_nodes(
      std::unordered_set<std::uint64_t>& deleted_nodes_ids);

  /// TBD
  double get_weight(std::uint64_t node_i_id, std::uint64_t node_j_id);

  ///@brief Resets each node’s number of updates.
  void reset_times_updated();

  /// Returns given node’s current community label.
  /// If no label was found (e.g. with the min_value set too high),
  /// the assigned label number is -1.
  ///
  /// @return -- given node’s current community label
  std::int64_t get_label(std::uint64_t node_id);

  ///@return -- given node’s most probable community labels
  std::vector<std::uint64_t> most_probable_labels(std::uint64_t node_id);
#pragma endregion helper_methods

#pragma region label_propagation_steps
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
  bool distinct_enough(std::uint64_t node_i_id, double similarity_threshold);

  ///@brief Performs label propagation on given node’s label probability vector.
  /// Label propagation works by calculating a weighted sum of the label
  /// probability vectors of the node’s neighbors.
  ///
  ///@param node_i -- given node
  ///
  ///@return -- updated label probabilities for given node
  std::unordered_map<std::uint64_t, double> propagate(std::uint64_t node_i_id);

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
      std::unordered_set<std::uint64_t> changed_nodes = {},
      std::unordered_set<std::uint64_t> to_delete = {});
#pragma endregion label_propagation_steps

 public:
  ///@brief Creates an instance of the LabelRankT algorithm.
  ///
  ///@param graph -- reference to current graph
  LabelRankT(std::unique_ptr<mg_graph::Graph<>>& graph) : graph(graph){};

  ///@brief Sets parameters given to the query module’s set() method.
  ///
  ///@param weight_property -- weight-containing edge property’s name
  ///@param w_selfloop -- default weight of self-loops
  ///@param similarity_threshold -- similarity threshold used in the node
  /// selection step, values in [0, 1]
  ///@param exponent -- exponent used in the inflation step
  ///@param min_value -- smallest acceptable probability in the cutoff step
  ///@param max_iterations -- maximum number of iterations
  ///@param max_updates -- maximum number of updates for any node
  void set_parameters(std::string weight_property, double w_selfloop,
                      double similarity_threshold, double exponent,
                      double min_value);

  ///@brief Returns previously calculated community labels.
  /// If no calculation has been done previously, calculates community labels
  /// with default parameter values.
  ///
  /// Label numbers are initially derived from Memgraph’s node IDs. As those
  /// grow larger with graph updates, this function renumbers them so that they
  /// begin with 1.
  ///
  ///@return -- {node id, community label} pairs
  std::unordered_map<std::uint64_t, std::int64_t> get_labels();

  ///@brief Calculates community labels with LabelRankT.
  ///
  ///@param max_iterations -- maximum number of iterations
  ///@param max_updates -- maximum number of updates for any node
  ///@param changed_nodes -- list of changed nodes (for incremental update)
  ///@param to_delete -- list of deleted nodes (for incremental update)
  ///
  ///@return -- {node id, community label} pairs
  std::unordered_map<std::uint64_t, std::int64_t> calculate_labels(
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
  std::unordered_map<std::uint64_t, std::int64_t> update_labels(
      std::vector<std::uint64_t> modified_nodes,
      std::vector<std::pair<std::uint64_t, std::uint64_t>> modified_edges,
      std::vector<std::uint64_t> deleted_nodes,
      std::vector<std::pair<std::uint64_t, std::uint64_t>> deleted_edges);
};
}  // namespace LabelRankT
