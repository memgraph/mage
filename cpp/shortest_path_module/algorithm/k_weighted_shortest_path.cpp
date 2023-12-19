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

struct CompareDist {
  bool operator()(std::pair<mgp::Node, double> const &n1, std::pair<mgp::Node, double> const &n2) {
    return n1.second > n2.second;
  }
};

TempPath Dijkstra(mgp::Graph &graph, mgp::Node &source, mgp::Node &sink) {
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
      double weight = relationship.GetProperty("weight").ValueNumeric();
      double alternative_distance = distances[node.Id().AsUint()] + weight;

      if (alternative_distance < distances[neighbor.Id().AsUint()]) {
        distances[neighbor.Id().AsUint()] = alternative_distance;
        previous[neighbor.Id().AsUint()] = node.Id();
        queue.push({neighbor, distances[neighbor.Id().AsUint()]});
      }
    }
  }

  TempPath shortest_path = {0, {}};
  for (mgp::Node node = sink; node != source; node = graph.GetNodeById(previous[node.Id().AsUint()])) {
    shortest_path.vertices.push_back(node);
  }
  shortest_path.vertices.push_back(source);
  std::reverse(shortest_path.vertices.begin(), shortest_path.vertices.end());
  shortest_path.weight = distances[sink.Id().AsUint()];

  return shortest_path;
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

    mgp::Graph graph{memgraph_graph};
    auto shortest_path = Dijkstra(graph, source, sync);
    std::string result_string = "Shortest path: " + std::to_string(shortest_path.weight) + "\n";
    mgp_error er = mgp_log(mgp_log_level::MGP_LOG_LEVEL_CRITICAL, result_string.c_str());
    for (auto node : shortest_path.vertices) {
      std::string node_string = node.ToString() + "\n";
      mgp_error er = mgp_log(mgp_log_level::MGP_LOG_LEVEL_CRITICAL, node_string.c_str());
    }
    // auto record = record_factory.NewRecord();
    // record.Insert(std::string(kProcedureKShortestPath).c_str(), result);
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
