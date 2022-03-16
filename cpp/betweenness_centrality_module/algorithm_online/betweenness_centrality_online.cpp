#include <mg_graph.hpp>

#include "../../biconnected_components_module/algorithm/biconnected_components.hpp"
#include "../algorithm/betweenness_centrality.hpp"
#include "betweenness_centrality_online.hpp"

namespace online_bc {
bool OnlineBC::Inconsistent(const mg_graph::GraphView<> &graph) {
  if (graph.Nodes().size() != this->node_bc_scores.size()) return true;

  for (const auto [node_inner_id] : graph.Nodes()) {
    if (!this->node_bc_scores.count(graph.GetMemgraphNodeId(node_inner_id))) return true;
  }

  return false;
}

std::unordered_map<std::uint64_t, double> OnlineBC::NormalizeBC(
    const std::unordered_map<std::uint64_t, double> &node_bc_scores, const std::uint64_t graph_order) {
  const double normalization_factor =
      this->directed ? 1.0 / ((graph_order - 1) * (graph_order - 2)) : 2.0 / ((graph_order - 1) * (graph_order - 2));
  std::unordered_map<std::uint64_t, double> normalized_bc_scores;
  for (const auto [node_id, bc_score] : node_bc_scores) {
    normalized_bc_scores[node_id] = bc_score * normalization_factor;
  }

  return normalized_bc_scores;
}

void OnlineBC::BrandesWrapper(const mg_graph::GraphView<> &graph, const std::uint64_t threads) {
  this->node_bc_scores.clear();

  const auto bc_scores = betweenness_centrality_alg::BetweennessCentrality(graph, this->directed, false, threads);
  for (std::uint64_t node_id = 0; node_id < graph.Nodes().size(); ++node_id) {
    this->node_bc_scores[graph.GetMemgraphNodeId(node_id)] = bc_scores[node_id];
  }
}

bcc_data OnlineBC::IsolateAffectedBCC(const mg_graph::GraphView<> &graph,
                                      const std::pair<std::uint64_t, std::uint64_t> updated_edge) const {
  std::unordered_set<std::uint64_t> articulation_points_by_bcc;
  std::vector<std::unordered_set<std::uint64_t>> nodes_by_bcc;
  const auto edges_by_bcc = bcc_algorithm::GetBiconnectedComponents(graph, articulation_points_by_bcc, nodes_by_bcc);

  std::unordered_set<std::uint64_t> affected_bcc_nodes;
  std::set<std::pair<std::uint64_t, std::uint64_t>>
      affected_bcc_edges;  // std::pair cannot be an std::unordered_set element
  std::unordered_set<std::uint64_t> affected_bcc_articulation_points;

  for (std::size_t i = 0; i < edges_by_bcc.size(); i++) {
    if (std::any_of(edges_by_bcc[i].begin(), edges_by_bcc[i].end(), [updated_edge](auto &edge) {
          return edge.from == updated_edge.first && edge.to == updated_edge.second;  // if the edge is in the BCC
        })) {
      for (const auto node : nodes_by_bcc[i]) {
        affected_bcc_nodes.insert(node);
      }
      for (const auto &edge : edges_by_bcc[i]) {
        affected_bcc_edges.insert({edge.from, edge.to});
      }
      for (const auto node : articulation_points_by_bcc) {
        if (affected_bcc_nodes.count(node)) affected_bcc_articulation_points.insert(node);
      }
    }
  }

  return {affected_bcc_nodes, affected_bcc_edges, affected_bcc_articulation_points};
}

std::unordered_map<std::uint64_t, int> OnlineBC::SSSPLengths(
    const mg_graph::GraphView<> &graph, const std::uint64_t source_node_id,
    const std::unordered_set<std::uint64_t> &affected_bcc_nodes) const {
  std::unordered_map<std::uint64_t, int> distances;
  distances[source_node_id] = 0;

  std::queue<std::uint64_t> queue({source_node_id});
  while (!queue.empty()) {
    const auto current_id = queue.front();
    queue.pop();

    for (const auto neighbor_id : graph.GetNeighboursMemgraphNodeIds(graph.GetInnerNodeId(current_id))) {
      if (!affected_bcc_nodes.count(neighbor_id)) continue;

      if (!distances.count(neighbor_id)) {  // if unvisited
        queue.push(neighbor_id);
        distances[neighbor_id] = distances[current_id] + 1;
      }
    }
  }

  return distances;
}

std::unordered_map<std::uint64_t, int> OnlineBC::PeripheralSubgraphsOrder(
    const mg_graph::GraphView<> &graph, std::unordered_set<std::uint64_t> affected_bcc_articulation_points,
    std::unordered_set<std::uint64_t> affected_bcc_nodes) const {
  std::unordered_map<std::uint64_t, int> peripheral_subgraphs_order;
  for (const auto articulation_point_id : affected_bcc_articulation_points) {
    std::unordered_set<std::uint64_t> visited({articulation_point_id});

    std::queue<std::uint64_t> queue({articulation_point_id});
    while (!queue.empty()) {
      const auto current_id = queue.front();
      queue.pop();

      for (const auto neighbor_id : graph.GetNeighboursMemgraphNodeIds(graph.GetInnerNodeId(current_id))) {
        if (affected_bcc_nodes.count(neighbor_id)) continue;

        if (!visited.count(neighbor_id)) {
          queue.push(neighbor_id);
          visited.insert(neighbor_id);
        }
      }
    }

    visited.erase(articulation_point_id);
    peripheral_subgraphs_order[articulation_point_id] = visited.size();
  }

  return peripheral_subgraphs_order;
}

brandesian_bfs_data OnlineBC::BrandesianBFS(const mg_graph::GraphView<> &graph, const std::uint64_t source_node_id,
                                            const std::unordered_set<std::uint64_t> &affected_bcc_nodes) const {
  std::unordered_map<std::uint64_t, int> distances;
  distances[source_node_id] = 0;
  std::unordered_map<std::uint64_t, int> n_shortest_paths({{source_node_id, 1}});
  std::unordered_map<std::uint64_t, std::set<std::uint64_t>> predecessors;
  std::vector<std::uint64_t> bfs_order({source_node_id});

  std::queue<std::uint64_t> queue({source_node_id});
  while (!queue.empty()) {
    const auto current_id = queue.front();
    queue.pop();

    for (const auto neighbor_id : graph.GetNeighboursMemgraphNodeIds(graph.GetInnerNodeId(current_id))) {
      if (!affected_bcc_nodes.count(neighbor_id)) continue;

      if (!distances.count(neighbor_id)) {  // if unvisited
        queue.push(neighbor_id);
        bfs_order.push_back(neighbor_id);
        distances[neighbor_id] = distances[current_id] + 1;
      }

      if (distances[neighbor_id] == distances[current_id] + 1) {
        n_shortest_paths[neighbor_id] += n_shortest_paths[current_id];
        predecessors[neighbor_id].insert(current_id);
      }
    }
  }

  n_shortest_paths[source_node_id] = 0;
  std::set<std::uint64_t> empty;
  predecessors[source_node_id] = empty;
  std::reverse(bfs_order.begin(), bfs_order.end());

  return {n_shortest_paths, predecessors, bfs_order};
}

void OnlineBC::Iteration(const mg_graph::GraphView<> &prior_graph, const mg_graph::GraphView<> &current_graph,
                         const std::uint64_t s_id, const std::unordered_set<std::uint64_t> &affected_bcc_nodes,
                         const std::unordered_set<std::uint64_t> &affected_bcc_articulation_points,
                         const std::unordered_map<std::uint64_t, int> &peripheral_subgraphs_order) {
  // Avoid counting edges twice if the graph is undirected
  const double quotient = this->directed ? 1.0 : NO_DOUBLE_COUNT;

  // Step 1: removes s_id’s contribution to betweenness centrality scores in the prior graph

  const auto [n_shortest_paths_prior, predecessors_prior, reverse_bfs_order_prior] =
      BrandesianBFS(prior_graph, s_id, affected_bcc_nodes);

  std::unordered_map<std::uint64_t, double> dependency_s_on, ext_dependency_s_on;
  for (const auto node_id : affected_bcc_nodes) {
    dependency_s_on[node_id] = 0;
    ext_dependency_s_on[node_id] = 0;
  }

  for (const auto w_id : reverse_bfs_order_prior) {
    if (affected_bcc_articulation_points.count(s_id) && affected_bcc_articulation_points.count(w_id)) {
      ext_dependency_s_on.at(w_id) = peripheral_subgraphs_order.at(s_id) * peripheral_subgraphs_order.at(w_id);
    }

    for (const auto p_id : predecessors_prior.at(w_id)) {
      auto ratio = static_cast<double>(n_shortest_paths_prior.at(p_id)) / n_shortest_paths_prior.at(w_id);
      dependency_s_on.at(p_id) += (ratio * (1 + dependency_s_on.at(w_id)));
      if (affected_bcc_articulation_points.count(s_id)) {
        ext_dependency_s_on.at(p_id) += (ext_dependency_s_on.at(w_id) * ratio);
      }
    }

    if (s_id != w_id) {
#pragma omp atomic update
      this->node_bc_scores[w_id] -= dependency_s_on.at(w_id) / quotient;
    }

    if (affected_bcc_articulation_points.count(s_id)) {
#pragma omp atomic update
      this->node_bc_scores[w_id] -= dependency_s_on.at(w_id) * peripheral_subgraphs_order.at(s_id);
#pragma omp atomic update
      this->node_bc_scores[w_id] -= ext_dependency_s_on.at(w_id) / quotient;
    }
  }

  // Step 2: adds s_id’s contribution to betweenness centrality scores in the current graph

  const auto [n_shortest_paths_current, predecessors_current, reverse_bfs_order_current] =
      BrandesianBFS(current_graph, s_id, affected_bcc_nodes);

  for (const auto node_id : affected_bcc_nodes) {
    dependency_s_on[node_id] = 0;
    ext_dependency_s_on[node_id] = 0;
  }

  for (const auto w_id : reverse_bfs_order_current) {
    if (affected_bcc_articulation_points.count(s_id) && affected_bcc_articulation_points.count(w_id)) {
      ext_dependency_s_on.at(w_id) = peripheral_subgraphs_order.at(s_id) * peripheral_subgraphs_order.at(w_id);
    }

    for (const auto p_id : predecessors_current.at(w_id)) {
      auto ratio = static_cast<double>(n_shortest_paths_current.at(p_id)) / n_shortest_paths_current.at(w_id);
      dependency_s_on.at(p_id) += ratio * (1 + dependency_s_on.at(w_id));
      if (affected_bcc_articulation_points.count(s_id)) {
        ext_dependency_s_on.at(p_id) += ext_dependency_s_on.at(w_id) * ratio;
      }
    }

    if (s_id != w_id) {
#pragma omp atomic update
      this->node_bc_scores[w_id] += dependency_s_on.at(w_id) / quotient;
    }

    if (affected_bcc_articulation_points.count(s_id)) {
#pragma omp atomic update
      this->node_bc_scores[w_id] += dependency_s_on.at(w_id) * peripheral_subgraphs_order.at(s_id);
#pragma omp atomic update
      this->node_bc_scores[w_id] += ext_dependency_s_on.at(w_id) / quotient;
    }
  }
}

void OnlineBC::iCentral(const mg_graph::GraphView<> &prior_graph, const mg_graph::GraphView<> &current_graph,
                        const Operation operation, const std::pair<std::uint64_t, std::uint64_t> updated_edge,
                        const std::uint64_t threads) {
  const mg_graph::GraphView<> &graph_with_updated_edge =
      (operation == Operation::INSERT_EDGE) ? current_graph : prior_graph;

  std::unordered_set<std::uint64_t> articulation_points_by_bcc;
  std::vector<std::unordered_set<std::uint64_t>> nodes_by_bcc;
  std::vector<std::vector<mg_graph::Edge<>>> edges_by_bcc;
  edges_by_bcc =
      bcc_algorithm::GetBiconnectedComponents(graph_with_updated_edge, articulation_points_by_bcc, nodes_by_bcc);

  const auto [affected_bcc_nodes, affected_bcc_edges, affected_bcc_articulation_points] =
      IsolateAffectedBCC(graph_with_updated_edge, updated_edge);

  const auto distances_first = SSSPLengths(graph_with_updated_edge, updated_edge.first, affected_bcc_nodes);
  const auto distances_second = SSSPLengths(graph_with_updated_edge, updated_edge.second, affected_bcc_nodes);

  const auto peripheral_subgraphs_order =
      PeripheralSubgraphsOrder(prior_graph, affected_bcc_articulation_points, affected_bcc_nodes);

  // OpenMP might throw errors when iterating over STL containers
  auto array_size = affected_bcc_nodes.size();
  std::uint64_t affected_bcc_nodes_array[array_size];
  std::uint64_t i = 0;
  for (const auto node_id : affected_bcc_nodes) {
    affected_bcc_nodes_array[i] = node_id;
    i++;
  }

  omp_set_dynamic(0);
  omp_set_num_threads(threads);
#pragma omp parallel for
  for (std::uint64_t i = 0; i < array_size; i++) {
    auto node_id = affected_bcc_nodes_array[i];
    if (distances_first.at(node_id) != distances_second.at(node_id)) {
      Iteration(prior_graph, current_graph, node_id, affected_bcc_nodes, affected_bcc_articulation_points,
                peripheral_subgraphs_order);
    }
  }
}

std::unordered_map<std::uint64_t, double> OnlineBC::Set(const mg_graph::GraphView<> &graph, const bool directed,
                                                        const bool normalize, const std::uint64_t threads) {
  this->directed = directed;

  BrandesWrapper(graph, threads);
  this->computed = true;

  if (normalize) return NormalizeBC(this->node_bc_scores, graph.Nodes().size());

  return this->node_bc_scores;
}

std::unordered_map<std::uint64_t, double> OnlineBC::Get(const mg_graph::GraphView<> &graph, const bool normalize) {
  if (!this->computed) BrandesWrapper(graph, std::thread::hardware_concurrency());

  if (Inconsistent(graph)) {
    throw std::runtime_error(
        "Graph has been modified and is thus inconsistent with cached betweenness centrality scores; to update them, "
        "please call set/reset!");
  }

  if (normalize) return NormalizeBC(this->node_bc_scores, graph.Nodes().size());

  return this->node_bc_scores;
}

std::unordered_map<std::uint64_t, double> OnlineBC::Update(const mg_graph::GraphView<> &prior_graph,
                                                           const mg_graph::GraphView<> &current_graph,
                                                           const Operation operation, const std::uint64_t updated_node,
                                                           const std::pair<std::uint64_t, std::uint64_t> updated_edge,
                                                           const bool normalize, const std::uint64_t threads) {
  if (!this->computed) {
    BrandesWrapper(current_graph, threads);
    this->computed = true;
  } else {
    if (operation == Operation::INSERT_EDGE || operation == Operation::DELETE_EDGE)
      iCentral(prior_graph, current_graph, operation, updated_edge, threads);
    else if (operation == Operation::INSERT_NODE || operation == Operation::DELETE_NODE)
      BrandesWrapper(current_graph, threads);
  }

  if (normalize) return NormalizeBC(this->node_bc_scores, current_graph.Nodes().size());

  return this->node_bc_scores;
}
}  // namespace online_bc
