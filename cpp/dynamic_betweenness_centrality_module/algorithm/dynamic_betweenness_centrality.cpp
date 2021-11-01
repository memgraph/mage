#include "dynamic_betweenness_centrality.hpp"

#include <mg_generate.hpp>
#include "bcc_utility.hpp"

dynamic_bc_algorithm::BetweennessCentralityData dynamic_bc_algorithm::context;

std::unordered_map<uint64_t, uint64_t> dynamic_bc_algorithm::get_original_node_ID_mapping(
    const mg_graph::GraphView<> &graph) {
  std::unordered_map<uint64_t, uint64_t> nodeMap;

  nodeMap.reserve(graph.Nodes().size());

  for (const auto &node : graph.Nodes()) {
    uint64_t nodeID = node.id;
    nodeMap[nodeID] = graph.GetInnerNodeId(nodeID);
  }

  return nodeMap;
}

void dynamic_bc_algorithm::initialize_betweenness_centrality(const mg_graph::GraphView<> &graph) {
  context.init(graph.Nodes().size());
}

std::unordered_map<uint64_t, uint64_t> dynamic_bc_algorithm::get_betweenness_centrality(
    const mg_graph::GraphView<> &graph) {
  if (!context.is_BC_empty()) {
    context.init(graph.Nodes().size());
  }

  return context.BC;
}

void dynamic_bc_algorithm::construct_BCC_data(const mg_graph::GraphView<> &edges_in_bcc, const uint64_t &first_node,
                                              const uint64_t &second_node, running_bc_update_data_t &data) {
  std::unordered_set<uint64_t> articulationPoints;
  std::vector<std::vector<mg_graph::Edge<>>> list_of_edges_grouped_by_bcc;

  // TODO: Mapping
  std::vector<std::unordered_set<std::uint64_t>> list_of_nodes_grouped_by_bcc =
      bcc_algorithm::GetBiconnectedComponents(edges_in_bcc, articulationPoints, list_of_edges_grouped_by_bcc);

  const auto bcc_nodes = std::find_if(
      std::begin(list_of_nodes_grouped_by_bcc), std::end(list_of_nodes_grouped_by_bcc), [&](const auto &nodes_in_bcc) {
        return std::find(std::begin(nodes_in_bcc), std::end(nodes_in_bcc), first_node) != nodes_in_bcc.end() &&
               std::find(std::begin(nodes_in_bcc), std::end(nodes_in_bcc), second_node) != nodes_in_bcc.end();
      });

  //
  if (bcc_nodes != std::end(list_of_nodes_grouped_by_bcc)) {
    const auto edges = list_of_edges_grouped_by_bcc[bcc_nodes - std::begin(list_of_nodes_grouped_by_bcc)];

    std::copy(std::begin(edges), std::end(edges),
              std::inserter(data.affected_bcc.edges, std::end(data.affected_bcc.edges)));

    for (const auto &node : *bcc_nodes) {
      data.affected_bcc.nodes.push_back(node);
      const auto &neighbours = edges_in_bcc.Neighbours(node);
      std::transform(std::begin(neighbours), std::end(neighbours),
                     std::inserter(data.affected_bcc.node_id_to_neighbor_id_map[node],
                                   std::end(data.affected_bcc.node_id_to_neighbor_id_map[node])),
                     [](const auto &neighbour) { return neighbour.node_id; });

      // TODO: try to write this more optimal - maybe DFS?
      if (articulationPoints.find(node) != articulationPoints.end()) {
        std::unordered_set<uint64_t> visited;
        std::queue<uint64_t> queue;

        queue.push(node);
        int count = 0;

        while (!queue.empty()) {
          std::uint64_t current = queue.front();
          queue.pop();

          if (current != node && visited.find(current) == visited.end()) {
            count++;
          }

          visited.insert(current);

          for (auto &node_neighbour : edges_in_bcc.Neighbours(current)) {
            const auto neighbour_node_id = node_neighbour.node_id;

            if (visited.find(neighbour_node_id) == visited.end() &&
                bcc_nodes->find(neighbour_node_id) == bcc_nodes->end()) {
              queue.push(neighbour_node_id);
            }
          }
        }
        data.articulation_point_id_to_external_subgraph_cardinality_map[node].push_back(count);
      }
    }
  }
}

void dynamic_bc_algorithm::SSSP(dynamic_bc_algorithm::running_bc_update_data_t &data, const uint64_t &source_node,
                                std::unordered_map<uint64_t, int> &distances) {
  distances.clear();

  for (const auto node_id : data.affected_bcc.nodes) {
    distances[node_id] = -1;
  }

  std::queue<uint64_t> queue;
  queue.push(source_node);

  distances[source_node] = 0;

  while (!queue.empty()) {
    uint64_t currentNode = queue.front();
    queue.pop();

    for (auto &neighbour : data.affected_bcc.node_id_to_neighbor_id_map[currentNode]) {
      if (distances.find(neighbour) == distances.end()) {
        continue;
      }

      if (distances[neighbour] == -1) {
        distances[neighbour] = distances[currentNode] + 1;
        queue.push(neighbour);
      }
    }
  }
}

void dynamic_bc_algorithm::BFS(const uint64_t &node, running_bc_update_data_t &data, iter_info_t &iter_info) {
  std::queue<uint64_t> queue;

  iter_info.sigma[node] = 1;
  iter_info.distance[node] = 0;

  queue.push(node);

  while (!queue.empty()) {
    uint64_t current_node = queue.front();
    queue.pop();

    iter_info.visited_nodes_in_order.emplace_back(current_node);

    for (auto &neighbour : data.affected_bcc.node_id_to_neighbor_id_map[current_node]) {
      if (iter_info.distance.find(neighbour) == iter_info.distance.end()) {
        queue.push(neighbour);
        iter_info.distance[neighbour] = iter_info.distance[current_node] + 1;

        iter_info.sigma[neighbour] = iter_info.sigma[neighbour] + iter_info.sigma[current_node];
        iter_info.predecessors[neighbour].insert(current_node);
      }
    }
  }
}

void dynamic_bc_algorithm::RBFS(std::unordered_map<uint64_t, double> &delta_BC, const uint64_t &node,
                                running_bc_update_data_t &data, iter_info_t &iter_info, const Operation &operation) {
  // itereting reverse BFS order
  // TODO: Check if visited nodes is OK
  for (auto iter = iter_info.visited_nodes_in_order.rbegin(); iter != iter_info.visited_nodes_in_order.rend(); ++iter) {
    if (context.articulation_points.find(node) != context.articulation_points.end() &&
        context.articulation_points.find(*iter) != context.articulation_points.end()) {
      std::uint64_t VG_source, VG_current;
      // TODO: Check if this works
      VG_source = std::accumulate(data.articulation_point_id_to_external_subgraph_cardinality_map[node].begin(),
                                  data.articulation_point_id_to_external_subgraph_cardinality_map[node].end(), 0);
      VG_current = std::accumulate(data.articulation_point_id_to_external_subgraph_cardinality_map[*iter].begin(),
                                   data.articulation_point_id_to_external_subgraph_cardinality_map[*iter].end(), 0);

      auto c_t = VG_current * VG_source;

      iter_info.delta_external[*iter] = iter_info.delta_external[*iter] + (double)c_t;
    }

    for (auto predecessor : iter_info.predecessors[*iter]) {
      double sp_sn = ((double)iter_info.sigma[predecessor] / (double)iter_info.sigma[*iter]);
      iter_info.delta[predecessor] = iter_info.delta[predecessor] + sp_sn * (1 + iter_info.delta[*iter]);
      if (context.articulation_points.find(node) != context.articulation_points.end()) {
        iter_info.delta_external[predecessor] = iter_info.delta_external[predecessor] + iter_info.delta[*iter] * sp_sn;
      }
    }

    if (node != *iter) {
      if (operation == dynamic_bc_algorithm::Operation::INSERTION) {
        delta_BC[*iter] += iter_info.delta[*iter] / 2.0;
      } else {
        delta_BC[*iter] -= iter_info.delta[*iter] / 2.0;
      }
    }

    if (context.articulation_points.find(node) != context.articulation_points.end()) {
      std::uint64_t VG_source =
          std::accumulate(data.articulation_point_id_to_external_subgraph_cardinality_map[node].begin(),
                          data.articulation_point_id_to_external_subgraph_cardinality_map[node].end(), 0);

      if (operation == dynamic_bc_algorithm::Operation::INSERTION) {
        delta_BC[*iter] += iter_info.delta[*iter] * VG_source;
        delta_BC[*iter] += iter_info.delta_external[*iter] / 2.0;
      } else {
        delta_BC[*iter] -= iter_info.delta[*iter] * VG_source;
        delta_BC[*iter] -= iter_info.delta_external[*iter] / 2.0;
      }
    }
  }
}

void dynamic_bc_algorithm::partial_BFS_add(const uint64_t &node, running_bc_update_data_t &data,
                                           const uint64_t &first_node, const uint64_t &second_node,
                                           iter_info_t &iter_info) {
  std::queue<uint64_t> queue;
  std::uint64_t first, second;

  if (iter_info.distance[first_node] > iter_info.distance[second_node]) {
    first = second_node;
    second = first_node;
  } else {
    first = first_node;
    second = second_node;
  }

  if (iter_info.distance[second] != iter_info.distance[first] + 1) {
    iter_info.predecessors[second].clear();
    iter_info.predecessors[second].insert(first);
    iter_info.sigma[second] = iter_info.sigma[first];
  } else {
    iter_info.predecessors[second].insert(first);
    iter_info.sigma[second] += iter_info.sigma[first];
  }

  queue.push(second);
  iter_info.distance[second] = iter_info.distance[first] + 1;
  iter_info.sigma_increment[second] = iter_info.sigma[first];
  iter_info.visited_nodes_in_order.emplace_back(second);

  while (!queue.empty()) {
    std::uint64_t current_node = queue.front();
    queue.pop();
    std::unordered_set<std::uint64_t> neighbours = data.affected_bcc.node_id_to_neighbor_id_map[current_node];

    for (auto neighbour_of_current_node : neighbours) {
      if (iter_info.distance[neighbour_of_current_node] > iter_info.distance[current_node] + 1) {
        iter_info.distance[neighbour_of_current_node] = iter_info.distance[current_node] + 1;
        iter_info.predecessors[neighbour_of_current_node].clear();
        iter_info.predecessors[neighbour_of_current_node].insert(current_node);
        iter_info.sigma[neighbour_of_current_node] = 0;
        iter_info.sigma_increment[neighbour_of_current_node] = iter_info.sigma_increment[current_node];
        iter_info.sigma[neighbour_of_current_node] += iter_info.sigma_increment[neighbour_of_current_node];

        if (std::find(iter_info.visited_nodes_in_order.begin(), iter_info.visited_nodes_in_order.end(),
                      neighbour_of_current_node) != iter_info.visited_nodes_in_order.end()) {
          iter_info.visited_nodes_in_order.push_back(neighbour_of_current_node);
          queue.push(neighbour_of_current_node);
        } else if (iter_info.distance[neighbour_of_current_node] == iter_info.distance[current_node] + 1) {
          iter_info.sigma_increment[neighbour_of_current_node] += iter_info.sigma_increment[current_node];
          iter_info.sigma[neighbour_of_current_node] += iter_info.sigma_increment[current_node];

          if (std::find(iter_info.predecessors[neighbour_of_current_node].begin(),
                        iter_info.predecessors[neighbour_of_current_node].end(),
                        current_node) == iter_info.predecessors[neighbour_of_current_node].end()) {
            iter_info.predecessors[neighbour_of_current_node].insert(current_node);
          }
          if (std::find(iter_info.visited_nodes_in_order.begin(), iter_info.visited_nodes_in_order.end(),
                        neighbour_of_current_node) == iter_info.visited_nodes_in_order.end()) {
            iter_info.visited_nodes_in_order.push_back(neighbour_of_current_node);
            queue.push(neighbour_of_current_node);
          }
        }
      }
    }
  }

  // TODO: Figure out what this part does

  //  for (int i = 1; i < iter_info.S.size(); ++i) {
  //    if (iter_info.dist_vec[iter_info.S[i - 1]] > iter_info.dist_vec[iter_info.S[i]]) {
  //      int j = i;
  //      while (iter_info.dist_vec[iter_info.S[j - 1]] > iter_info.dist_vec[iter_info.S[j]]) {
  //        node_id_t tmp = iter_info.S[j - 1];
  //        iter_info.S[j - 1] = iter_info.S[j];
  //        iter_info.S[j] = tmp;
  //        --j;
  //      }
  //    }
  //  }
}

void dynamic_bc_algorithm::partial_BFS_del(const uint64_t &node, running_bc_update_data_t &data,
                                           const uint64_t &first_node, const uint64_t &second_node,
                                           iter_info_t &iter_info) {
  std::queue<std::uint64_t> queue;

  iter_info.sigma[node] = 1;
  iter_info.distance[node] = 0;

  queue.push(node);

  while (!queue.empty()) {
    std::uint64_t current_node = queue.front();
    queue.pop();

    iter_info.S.push_back(current_node);

    std::unordered_set<std::uint64_t> neighbours = data.affected_bcc.node_id_to_neighbor_id_map[current_node];

    for (auto neighbour : neighbours) {
      if ((node == first_node && neighbour == second_node) || (node == second_node && neighbour == first_node)) {
        continue;
      }
      if (iter_info.distance[neighbour] < 0) {
        queue.push(neighbour);
        iter_info.distance[neighbour] = iter_info.distance[current_node] + 1;
      }
      if (iter_info.distance[neighbour] == iter_info.distance[node] + 1) {
        iter_info.sigma[neighbour] = iter_info.sigma[neighbour] + iter_info.sigma[current_node];
        iter_info.predecessors[neighbour].insert(current_node);
      }
    }
  }
}

void dynamic_bc_algorithm::iCentral_iteration(std::unordered_map<uint64_t, double> &delta_BC, const uint64_t &node,
                                              int dd, const uint64_t &first_node, const uint64_t &second_node,
                                              running_bc_update_data_t &data, const Operation &operation) {
  iter_info_t info;

  uint64_t first, second;

  dd > 0 ? first = second_node, second = first : first = first_node, second = second_node;

  dynamic_bc_algorithm::BFS(node, data, info);
  dynamic_bc_algorithm::RBFS(delta_BC, node, data, info, operation);

  // TODO: Uncomment this and see if it works
  // operation == Operation::INSERTION ? dynamic_bc_algorithm::partial_BFS_add() :
  // dynamic_bc_algorithm::partial_BFS_del();

  dynamic_bc_algorithm::RBFS(delta_BC, node, data, info, operation);
}

void dynamic_bc_algorithm::iCentral(const mg_graph::GraphView<> &graph, const uint64_t &first_node,
                                    const uint64_t &second_node, const Operation &operation) {
  if (context.is_BC_empty()) {
    initialize_betweenness_centrality(graph);
  }

  // TODO: Call Brandes here and map the results to IDs from the database

  std::unordered_map<uint64_t, uint64_t> initialBC =
      const_cast<const std::unordered_map<uint64_t, uint64_t> &>(context.BC);

  running_bc_update_data_t data;
  // Start BC Update

  construct_BCC_data(graph, first_node, second_node, data);

  std::unordered_map<uint64_t, int> distancesFromSource;
  std::unordered_map<uint64_t, int> distancesFromDestination;

  SSSP(data, first_node, distancesFromSource);
  SSSP(data, second_node, distancesFromDestination);

  iter_info_t info;

  // TODO: Initialize info?

  std::unordered_map<uint64_t, double> delta_BC;

  // TODO: Initialize delta_BC to zero?
  for (const auto &[key, value1] : distancesFromSource) {
    int value2 = distancesFromDestination[key];
    if (value1 != value2) {
      int dd = value1 - value2;
      iCentral_iteration(delta_BC, key, dd, first_node, second_node, data, operation);
    }
  }

  // End BC Update

  auto debug_breakpoint = 0;
}

// TODO: Finish this
void dynamic_bc_algorithm::update_biconnected_components_and_articulation_points(
    const mg_graph::GraphView<> &graph, std::vector<std::unordered_set<std::uint64_t>> &biconnected_components,
    std::unordered_set<uint64_t> &articulation_points) {
  // biconnected_components = bcc_algorithm::GetBiconnectedComponents(graph, articulation_points, );
}

int main(int argc, char const *argv[]) {
  auto graph = mg_generate::BuildGraph(
      20, {{8, 9},   {8, 10},  {9, 0},   {10, 0},  {12, 11}, {12, 13}, {11, 5}, {13, 5}, {17, 15},
           {17, 19}, {15, 16}, {19, 16}, {16, 14}, {16, 18}, {14, 4},  {18, 4}, {0, 1},  {1, 2},
           {0, 3},   {2, 3},   {2, 4},   {3, 5},   {5, 6},   {6, 7},   {7, 4}},
      mg_graph::GraphType::kUndirectedGraph);

  // graph->CreateEdge(3, 7);

  uint64_t from = 3;
  uint64_t to = 7;

  dynamic_bc_algorithm::iCentral(*graph, from, to, dynamic_bc_algorithm::Operation::INSERTION);
}