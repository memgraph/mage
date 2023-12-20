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

struct CompareDist {
  bool operator()(std::pair<mgp::Node, double> const &n1, std::pair<mgp::Node, double> const &n2) {
    return n1.second > n2.second;
  }
};

double GetEdgeWeight(mgp::Node &node1, mgp::Node &node2) {
  // Check outgoing relationships of node1
  for (const auto &relationship : node1.OutRelationships()) {
    if (relationship.To() == node2 && relationship.GetProperty("weight").IsNumeric()) {
      return relationship.GetProperty("weight").ValueNumeric();
    }
  }
  // Check incoming relationships of node1
  for (const auto &relationship : node1.InRelationships()) {
    if (relationship.From() == node2 && relationship.GetProperty("weight").IsNumeric()) {
      return relationship.GetProperty("weight").ValueNumeric();
    }
  }
  return 0.0;  // return 0 if no edge exists between node1 and node2, or if the edge has no weight
}

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

struct CompareWeightPath {
  bool operator()(TempPath const &p1, TempPath const &p2) { return p1.weight > p2.weight; }
};

bool ArePathsEqual(const TempPath &path1, const TempPath &path2) {
  // Compare weights with a tolerance
  double tolerance = 1e-9;  // or whatever small value you consider as "close enough"
  if (std::abs(path1.weight - path2.weight) > tolerance) {
    return false;
  }

  // Compare vertices
  if (path1.vertices.size() != path2.vertices.size()) {
    return false;
  }

  // Compare vertices
  std::set<uint64_t> vertices1, vertices2;
  for (const auto &vertex : path1.vertices) {
    vertices1.insert(vertex.Id().AsUint());
  }
  for (const auto &vertex : path2.vertices) {
    vertices2.insert(vertex.Id().AsUint());
  }

  return vertices1 == vertices2;
}

struct ComparePaths {
  bool operator()(const TempPath &path1, const TempPath &path2) const {
    // Compare vertices
    if (path1.vertices.size() != path2.vertices.size()) {
      return path1.vertices.size() < path2.vertices.size();
    }

    for (size_t i = 0; i < path1.vertices.size(); ++i) {
      if (path1.vertices[i].Id().AsUint() != path2.vertices[i].Id().AsUint()) {
        return path1.vertices[i].Id().AsUint() < path2.vertices[i].Id().AsUint();
      }
    }

    // Compare weights with a tolerance
    double tolerance = 1e-9;  // or whatever small value you consider as "close enough"
    return std::abs(path1.weight - path2.weight) > tolerance;
  }
};

std::vector<TempPath> YenKSP(mgp::Graph &graph, mgp::Node &source, mgp::Node &sink, int K) {
  std::vector<TempPath> shortest_paths;
  std::priority_queue<TempPath, std::vector<TempPath>, CompareWeightPath> candidates;

  DijkstraResult result = Dijkstra(graph, source, sink);
  shortest_paths.push_back(result.path);

  for (int k = 1; k < K; ++k) {
    TempPath prev_shortest_path = shortest_paths[k - 1];

    for (size_t i = 0; i < prev_shortest_path.vertices.size() - 1; ++i) {
      mgp::Node spur_node = prev_shortest_path.vertices[i];
      TempPath root_path;
      root_path.weight = 0;

      root_path.vertices.insert(root_path.vertices.end(), prev_shortest_path.vertices.begin(),
                                prev_shortest_path.vertices.begin() + i + 1);

      for (size_t j = 0; j < root_path.vertices.size() - 1; ++j) {
        root_path.weight += GetEdgeWeight(root_path.vertices[j], root_path.vertices[j + 1]);
      }
      std::set<uint64_t> ignore_nodes;
      std::set<std::pair<uint64_t, uint64_t>> ignore_edges;

      for (const TempPath &path : shortest_paths) {
        if (path.vertices.size() > i &&
            std::equal(path.vertices.begin(), path.vertices.begin() + i + 1, root_path.vertices.begin())) {
          ignore_edges.insert({path.vertices[i].Id().AsUint(), path.vertices[i + 1].Id().AsUint()});
        }
      }

      for (size_t j = 0; j < i; ++j) {
        ignore_nodes.insert(prev_shortest_path.vertices[j].Id().AsUint());
      }

      DijkstraResult spur_result = Dijkstra(graph, spur_node, sink, ignore_nodes, ignore_edges);

      if (!spur_result.path.vertices.empty()) {
        TempPath total_path = root_path;
        total_path.weight += spur_result.path.weight;
        total_path.vertices.insert(total_path.vertices.end(), spur_result.path.vertices.begin() + 1,
                                   spur_result.path.vertices.end());

        bool path_exists =
            std::find_if(shortest_paths.begin(), shortest_paths.end(), [&total_path](const TempPath &path) {
              return ArePathsEqual(path, total_path);
            }) != shortest_paths.end();

        if (!path_exists) {
          candidates.push(total_path);
        }
      }
    }

    if (candidates.empty()) {
      break;
    }

    shortest_paths.push_back(candidates.top());
    candidates.pop();
  }

  std::set<TempPath, ComparePaths> shortest_paths_duplicates;
  for (const auto &path : shortest_paths) {
    shortest_paths_duplicates.insert(path);
  }
  std::vector<TempPath> shortest_paths_vector(shortest_paths_duplicates.begin(), shortest_paths_duplicates.end());
  std::sort(shortest_paths_vector.begin(), shortest_paths_vector.end(),
            [](const TempPath &a, const TempPath &b) { return a.weight < b.weight; });

  return shortest_paths_vector;
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
    if (!number_of_weighted_shortest_paths.IsInt() or number_of_weighted_shortest_paths.ValueInt() < 1) {
      throw mgp::ValueException("Number of weighted shortest paths needs to be an integer and bigger then 0!");
    }
    auto source = start_node.ValueNode();
    auto sync = end_node.ValueNode();
    auto k = number_of_weighted_shortest_paths.ValueInt();

    mgp::Graph graph{memgraph_graph};
    auto dijkstra_result = Dijkstra(graph, source, sync);

    for (auto node : dijkstra_result.path.vertices) {
      std::string node_string = node.ToString() + "\n";
      mgp_error er = mgp_log(mgp_log_level::MGP_LOG_LEVEL_CRITICAL, node_string.c_str());
    }
    std::string result_string = "Shortest path: " + std::to_string(dijkstra_result.path.weight) + "\n";
    mgp_error er = mgp_log(mgp_log_level::MGP_LOG_LEVEL_CRITICAL, result_string.c_str());

    auto k_shortest_paths = YenKSP(graph, source, sync, k);
    for (auto path : k_shortest_paths) {
      for (auto node : path.vertices) {
        std::string node_string = node.ToString() + "\n";
        mgp_error er = mgp_log(mgp_log_level::MGP_LOG_LEVEL_CRITICAL, node_string.c_str());
      }
      std::string path_string = "Path: " + std::to_string(path.weight) + "\n";
      mgp_error er = mgp_log(mgp_log_level::MGP_LOG_LEVEL_CRITICAL, path_string.c_str());
    }

    // auto record = record_factory.NewRecord();
    // record.Insert(std::string(kProcedureKShortestPath).c_str(), result);
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
