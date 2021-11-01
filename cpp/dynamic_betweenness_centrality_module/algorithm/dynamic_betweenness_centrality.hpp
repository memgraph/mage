#pragma once

#include <stack>
#include <vector>

#include <mg_graph.hpp>
#include <unordered_set>

namespace dynamic_bc_algorithm {

enum class Operation { INSERTION, DELETION };

class BetweennessCentralityData {
 public:
  /*--- Constructor ---*/
  BetweennessCentralityData() = default;

  void init(uint64_t number_of_nodes) {
    if (!BC.empty()) {
      BC.clear();
    }

    BC.reserve(number_of_nodes);
  };

  bool is_BC_empty() { return BC.empty(); }

  /*--- Members ---*/
  std::unordered_map<uint64_t, uint64_t> BC;
  std::unordered_set<uint64_t> articulation_points;

  // TODO: Use this variable for biconnected components
  std::vector<std::unordered_set<std::uint64_t>> biconnected_components;
};

// https://www.techiedelight.com/use-std-pair-key-std-unordered_map-cpp/
struct edge_hash : std::unary_function<mg_graph::Edge<>, std::size_t> {
  constexpr std::size_t operator()(const mg_graph::Edge<> &edge) const {
    return edge.id;  // TODO: Check if this is OK
  }
};

struct edge_equal : std::binary_function<mg_graph::Edge<>, mg_graph::Edge<>, bool> {
  constexpr bool operator()(const mg_graph::Edge<> &lhs, const mg_graph::Edge<> &rhs) const {
    return lhs.from == rhs.from && lhs.id == rhs.id && lhs.to == rhs.to;
  }
};

typedef struct iter_info_t {
  /** Node ID -> Set of predecessors of that node */
  std::unordered_map<uint64_t, std::unordered_set<uint64_t>> predecessors;

  std::unordered_map<uint64_t, uint64_t> sigma;
  std::unordered_map<uint64_t, uint64_t> distance;
  std::unordered_map<uint64_t, double> delta;
  std::unordered_map<uint64_t, double> delta_external;
  std::vector<uint64_t> visited_nodes_in_order;

  std::unordered_map<std::uint64_t, uint64_t> sigma_increment;

  // TODO: Not sure what S is
  std::vector<std::uint64_t> S;

} iter_info_t;

typedef struct affected_bcc_t {
  /** Nodes inside BCC*/
  std::vector<uint64_t> nodes{};

  /** Node ID -> List of neighbors map. */
  std::unordered_map<std::uint64_t, std::unordered_set<std::uint64_t>> node_id_to_neighbor_id_map{};

  /** Set of node ID -> Node ID pairs representing the edges in the affected biconnected component.  */
  std::unordered_set<mg_graph::Edge<>, edge_hash, edge_equal> edges{};

} affected_bcc_t;

/** Data used only during the algorithm */
typedef struct running_bc_update_data_t {
  /** Data of the affected biconnected component. */
  affected_bcc_t affected_bcc{};

  /** Articulation Point Id -> Number of outside nodes map. */
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>>
      articulation_point_id_to_external_subgraph_cardinality_map{};
} running_bc_update_data_t;

extern BetweennessCentralityData context;

/**
 * @brief The main algorithm
 *
 * @param graph
 * @param first_node - first node of the edge to be inserted or deleted
 * @param second_node - second node of the edge to be inserted or deleted
 * @param operation
 */
void iCentral(const mg_graph::GraphView<> &graph, const uint64_t &first_node, const uint64_t &second_node,
              const Operation &operation);

/**
 * @brief A function for retrieving the original node ID
 *
 * @param[in] graph
 *
 * @return An unordered map that maps node IDs to original node IDs from the database
 */
std::unordered_map<uint64_t, uint64_t> get_original_node_ID_mapping(const mg_graph::GraphView<> &graph);

// TODO: Use this at some point
/**
 * @brief A function that updates biconnected components and articulation points of a graph
 *
 * @param graph
 * @param biconnectedComponents A vector of biconnected components
 * @param articulationPoints An unordered set of articulation points
 */
void update_biconnected_components_and_articulation_points(
    const mg_graph::GraphView<> &graph, std::vector<std::unordered_set<std::uint64_t>> &biconnected_components,
    std::unordered_set<uint64_t> &articulation_points);

/**
 * @brief A function for initializing the BC map
 *
 * @param[in] graph
 */
void initialize_betweenness_centrality(const mg_graph::GraphView<> &graph);

void construct_BCC_data(const mg_graph::GraphView<> &edges_in_bcc, const uint64_t &first_node,
                        const uint64_t &second_node, running_bc_update_data_t &data);

/**
 * @brief Single source shortest path form given source node to other nodes inside affected BCC
 *
 * @param data
 * @param sourceNode
 * @param distances
 */
void SSSP(running_bc_update_data_t &data, const uint64_t &source_node, std::unordered_map<uint64_t, int> &distances);

/**
 * @brief An iteration of the iCentral algorithm
 *
 * @param delta_BC - how much the BC will change
 * @param node - source node
 * @param dd - difference between distances (source node <-> any node inside BCC) and (any node inside BCC <-> target
 *              node)
 * @param first_node - first node of the edge to be inserted or deleted
 * @param second_node - second node of the edge to be inserted or deleted
 * @param data - data used only during the algorithm
 * @param operation
 */
void iCentral_iteration(std::unordered_map<uint64_t, double> &delta_BC, const uint64_t &node, int dd,
                        const uint64_t &first_node, const uint64_t &second_node, running_bc_update_data_t &data,
                        const Operation &operation);

void BFS(const uint64_t &node, running_bc_update_data_t &data, iter_info_t &iter_info);
void RBFS(std::unordered_map<uint64_t, double> &delta_BC, const uint64_t &node, running_bc_update_data_t &data,
          iter_info_t &iter_info, const Operation &operation);
void partial_BFS_add(const uint64_t &node, running_bc_update_data_t &data, const uint64_t &first_node,
                     const uint64_t &second_node, iter_info_t &iter_info);
void partial_BFS_del(const uint64_t &node, running_bc_update_data_t &data, const uint64_t &first_node,
                     const uint64_t &second_node, iter_info_t &iter_info);

/**
 * @brief A function for getting the betweenness centrality of nodes
 *
 * @return An unordered map containing betweenness centrality of nodes
 */
std::unordered_map<uint64_t, uint64_t> get_betweenness_centrality(const mg_graph::GraphView<> &graph);
}  // namespace dynamic_bc_algorithm
