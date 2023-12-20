#include "k_weighted_shortest_path.hpp"
#include <cstdint>
#include <mgp.hpp>
#include <queue>
#include <syncstream>
#include <unordered_map>
#include "mg_procedure.h"

struct TempPath {
  double weight;
  std::vector<mgp::Node> vertices;
};

struct DijkstraResult {
  TempPath path;
  std::unordered_map<uint64_t, double> distances;
};

struct CompareTotalPath {
  bool operator()(TempPath const &p1, TempPath const &p2) { return p1.weight > p2.weight; }
};

struct CompareDist {
  bool operator()(std::pair<mgp::Node, double> const &n1, std::pair<mgp::Node, double> const &n2) {
    return n1.second > n2.second;
  }
};

DijkstraResult Dijkstra(mgp::Graph &graph, mgp::Node &source, mgp::Node &sink,
                        const std::set<uint64_t> &ignore_nodes = {},
                        const std::set<std::pair<uint64_t, uint64_t>> &ignore_edges = {}) {
  std::unordered_map<uint64_t, double> distances;
  std::unordered_map<uint64_t, mgp::Id> previous;
  std::priority_queue<std::pair<mgp::Node, double>, std::vector<std::pair<mgp::Node, double>>, CompareDist> queue;
  for (const auto &node : graph.Nodes()) {
    if (node == source) {
      distances[node.Id().AsUint()] = 0;
    } else {
      distances[node.Id().AsUint()] = std::numeric_limits<double>::infinity();
    }

    queue.push({node, distances[node.Id().AsUint()]});
  }

  while (!queue.empty()) {
    std::pair<mgp::Node, double> element = queue.top();
    mgp::Node node = element.first;
    queue.pop();

    for (const auto &relationship : node.OutRelationships()) {
      mgp::Node neighbor = relationship.To();

      // Skip the node if it's in the ignore list
      if (ignore_nodes.find(neighbor.Id().AsUint()) != ignore_nodes.end()) {
        continue;
      }
      // Skip the edge if it's in the ignore list
      if (ignore_edges.find({node.Id().AsUint(), neighbor.Id().AsUint()}) != ignore_edges.end()) {
        continue;
      }
      double weight = relationship.GetProperty("weight").ValueNumeric();
      double alternative_distance = distances[node.Id().AsUint()] + weight;

      if (alternative_distance < distances[neighbor.Id().AsUint()]) {
        distances[neighbor.Id().AsUint()] = alternative_distance;
        previous[neighbor.Id().AsUint()] = node.Id();
        queue.push({neighbor, distances[neighbor.Id().AsUint()]});
      }
    }
  }

  if (previous.find(sink.Id().AsUint()) == previous.end()) {
    // Dijkstra's algorithm didn't find a path from the source to the sink
    return {std::numeric_limits<double>::infinity(), {}};
  }
  TempPath shortest_path = {0, {}};
  for (mgp::Node node = sink; node != source; node = graph.GetNodeById(previous[node.Id().AsUint()])) {
    shortest_path.vertices.push_back(node);
  }
  shortest_path.vertices.push_back(source);
  std::reverse(shortest_path.vertices.begin(), shortest_path.vertices.end());
  shortest_path.weight = distances[sink.Id().AsUint()];

  return {shortest_path, distances};
}

std::vector<TempPath> Yen(mgp::Graph &graph, mgp::Node &source, mgp::Node &sink, int K) {
  std::vector<TempPath> shortest_paths;
  DijkstraResult result = Dijkstra(graph, source, sink);
  TempPath shortest_path = result.path;
  std::unordered_map<uint64_t, double> distances = result.distances;
  shortest_paths.push_back(shortest_path);

  std::priority_queue<TempPath, std::vector<TempPath>, CompareTotalPath> candidates;

  for (int k = 1; k < K; ++k) {
    TempPath prev_shortest_path = shortest_paths[k - 1];
    for (size_t i = 0; i < prev_shortest_path.vertices.size() - 1; ++i) {
      mgp::Node spur_node = prev_shortest_path.vertices[i];
      TempPath root_path = {0, {}};
      root_path.vertices.insert(root_path.vertices.end(), prev_shortest_path.vertices.begin(),
                                prev_shortest_path.vertices.begin() + i + 1);
      root_path.weight = prev_shortest_path.weight - distances[prev_shortest_path.vertices[i + 1].Id().AsUint()];

      std::set<std::pair<uint64_t, uint64_t>> ignore_edges;
      for (const auto &path : shortest_paths) {
        if (root_path.vertices == std::vector<mgp::Node>(path.vertices.begin(), path.vertices.begin() + i + 1)) {
          // Temporarily remove the next edge in path
          ignore_edges.insert({path.vertices[i].Id().AsUint(), path.vertices[i + 1].Id().AsUint()});
        }
      }
      std::set<uint64_t> ignore_nodes;
      for (const auto &node : root_path.vertices) {
        if (node != spur_node) {
          // Temporarily remove all edges from node
          ignore_nodes.insert(node.Id().AsUint());
        }
      }

      DijkstraResult spur_result = Dijkstra(graph, spur_node, sink, ignore_nodes, ignore_edges);
      TempPath spur_path = spur_result.path;
      if (!spur_path.vertices.empty()) {
        TempPath total_path = root_path;
        total_path.weight += spur_path.weight;
        total_path.vertices.insert(total_path.vertices.end(), spur_path.vertices.begin() + 1, spur_path.vertices.end());
        // Add total_path to the candidates
        candidates.push(total_path);
      }

      if (candidates.empty()) {
        break;
      }

      // Find the candidate with the smallest weight and add it to shortest_paths
      TempPath min_weight_candidate = candidates.top();
      candidates.pop();
      shortest_paths.push_back(min_weight_candidate);
    }
  }
  return shortest_paths;
}

void KWeightedShortestPath::KWeightedShortestPath(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result,
                                                  mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto start_node = arguments[0];
    auto end_node = arguments[1];
    auto number_of_weighted_shortest_paths = arguments[2];

    if (!start_node.IsNode()) {
      throw mgp::ValueException("Start node needs to be a node!");
    }
    if (!end_node.IsNode()) {
      throw mgp::ValueException("End node needs to be a node!");
    }
    if (!number_of_weighted_shortest_paths.IsInt()) {
      throw mgp::ValueException("Number of weighted shortest paths needs to be an integer!");
    }
    auto source = start_node.ValueNode();
    auto sync = end_node.ValueNode();
    auto k = number_of_weighted_shortest_paths.ValueInt();

    mgp::Graph graph{memgraph_graph};
    auto dijkstra_result = Dijkstra(graph, source, sync);
    std::string result_string = "Shortest path: " + std::to_string(dijkstra_result.path.weight) + "\n";
    mgp_error er = mgp_log(mgp_log_level::MGP_LOG_LEVEL_CRITICAL, result_string.c_str());
    for (auto node : dijkstra_result.path.vertices) {
      std::string node_string = node.ToString() + "\n";
      mgp_error er = mgp_log(mgp_log_level::MGP_LOG_LEVEL_CRITICAL, node_string.c_str());
    }

    auto k_shortest_paths = Yen(graph, source, sync, k);
    for (auto path : k_shortest_paths) {
      std::string path_string = "Path: " + std::to_string(path.weight) + "\n";
      mgp_error er = mgp_log(mgp_log_level::MGP_LOG_LEVEL_CRITICAL, path_string.c_str());
      for (auto node : path.vertices) {
        std::string node_string = node.ToString() + "\n";
        mgp_error er = mgp_log(mgp_log_level::MGP_LOG_LEVEL_CRITICAL, node_string.c_str());
      }
    }
    // auto record = record_factory.NewRecord();
    // record.Insert(std::string(kProcedureKShortestPath).c_str(), result);
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}