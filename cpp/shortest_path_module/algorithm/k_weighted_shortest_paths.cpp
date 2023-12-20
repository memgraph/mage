#include "k_weighted_shortest_paths.hpp"
#include <cstdint>
#include <mgp.hpp>
#include <queue>
#include <string>
#include <syncstream>
#include <unordered_map>
#include "mg_procedure.h"

double KWeightedShortestPaths::GetEdgeWeight(mgp::Node &node1, mgp::Node &node2, const std::string_view &weight_name) {
  for (const auto &relationship : node1.OutRelationships()) {
    if (relationship.To() == node2 && relationship.GetProperty(std::string(weight_name)).IsNumeric()) {
      return relationship.GetProperty(std::string(weight_name)).ValueNumeric();
    }
  }
  for (const auto &relationship : node1.InRelationships()) {
    if (relationship.From() == node2 && relationship.GetProperty(std::string(weight_name)).IsNumeric()) {
      return relationship.GetProperty(std::string(weight_name)).ValueNumeric();
    }
  }
  return 0.0;
}

KWeightedShortestPaths::DijkstraResult KWeightedShortestPaths::Dijkstra(
    mgp::Graph &graph, mgp::Node &source, mgp::Node &sink, const std::string_view &weight_name,
    const std::set<uint64_t> &ignore_nodes, const std::set<std::pair<uint64_t, uint64_t>> &ignore_edges) {
  std::unordered_map<uint64_t, double> distances;
  std::unordered_map<uint64_t, mgp::Id> previous;
  std::priority_queue<std::pair<mgp::Node, double>, std::vector<std::pair<mgp::Node, double>>,
                      KWeightedShortestPaths::CompareWeightDistance>
      queue;
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

      if (ignore_nodes.find(neighbor.Id().AsUint()) != ignore_nodes.end()) {
        continue;
      }

      if (ignore_edges.find({node.Id().AsUint(), neighbor.Id().AsUint()}) != ignore_edges.end()) {
        continue;
      }
      double weight = relationship.GetProperty(std::string(weight_name)).ValueNumeric();
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
    return {{std::numeric_limits<double>::infinity(), {}}};
  }
  KWeightedShortestPaths::TempPath shortest_path = {0, {}};
  for (mgp::Node node = sink; node != source; node = graph.GetNodeById(previous[node.Id().AsUint()])) {
    shortest_path.vertices.push_back(node);
  }
  shortest_path.vertices.push_back(source);
  std::reverse(shortest_path.vertices.begin(), shortest_path.vertices.end());
  shortest_path.weight = distances[sink.Id().AsUint()];

  return {shortest_path, distances};
}

std::vector<KWeightedShortestPaths::TempPath> KWeightedShortestPaths::YenKSP(mgp::Graph &graph, mgp::Node &source,
                                                                             mgp::Node &sink, int K,
                                                                             const std::string_view &weight_name) {
  std::vector<KWeightedShortestPaths::TempPath> shortest_paths;
  std::priority_queue<KWeightedShortestPaths::TempPath, std::vector<KWeightedShortestPaths::TempPath>,
                      KWeightedShortestPaths::CompareWeightPath>
      candidates;

  KWeightedShortestPaths::DijkstraResult result = Dijkstra(graph, source, sink, weight_name);
  shortest_paths.push_back(result.path);

  for (int k = 1; k < K; ++k) {
    KWeightedShortestPaths::TempPath prev_shortest_path = shortest_paths[k - 1];

    for (size_t i = 0; i < prev_shortest_path.vertices.size() - 1; ++i) {
      mgp::Node spur_node = prev_shortest_path.vertices[i];
      KWeightedShortestPaths::TempPath root_path;
      root_path.weight = 0;

      root_path.vertices.insert(root_path.vertices.end(), prev_shortest_path.vertices.begin(),
                                prev_shortest_path.vertices.begin() + i + 1);

      for (size_t j = 0; j < root_path.vertices.size() - 1; ++j) {
        root_path.weight +=
            KWeightedShortestPaths::GetEdgeWeight(root_path.vertices[j], root_path.vertices[j + 1], weight_name);
      }
      std::set<uint64_t> ignore_nodes;
      std::set<std::pair<uint64_t, uint64_t>> ignore_edges;

      for (const KWeightedShortestPaths::TempPath &path : shortest_paths) {
        if (path.vertices.size() > i &&
            std::equal(path.vertices.begin(), path.vertices.begin() + i + 1, root_path.vertices.begin())) {
          ignore_edges.insert({path.vertices[i].Id().AsUint(), path.vertices[i + 1].Id().AsUint()});
        }
      }

      for (size_t j = 0; j < i; ++j) {
        ignore_nodes.insert(prev_shortest_path.vertices[j].Id().AsUint());
      }

      KWeightedShortestPaths::DijkstraResult spur_result =
          KWeightedShortestPaths::Dijkstra(graph, spur_node, sink, weight_name, ignore_nodes, ignore_edges);

      if (!spur_result.path.vertices.empty()) {
        KWeightedShortestPaths::TempPath total_path = root_path;
        total_path.weight += spur_result.path.weight;
        total_path.vertices.insert(total_path.vertices.end(), spur_result.path.vertices.begin() + 1,
                                   spur_result.path.vertices.end());

        candidates.push(total_path);
      }
    }

    if (candidates.empty()) {
      break;
    }

    shortest_paths.push_back(candidates.top());
    candidates.pop();
  }

  std::set<KWeightedShortestPaths::TempPath, KWeightedShortestPaths::ComparePaths> shortest_paths_duplicates;
  for (const auto &path : shortest_paths) {
    shortest_paths_duplicates.insert(path);
  }
  std::vector<KWeightedShortestPaths::TempPath> shortest_paths_vector(shortest_paths_duplicates.begin(),
                                                                      shortest_paths_duplicates.end());
  std::sort(shortest_paths_vector.begin(), shortest_paths_vector.end(),
            [](const KWeightedShortestPaths::TempPath &a, const KWeightedShortestPaths::TempPath &b) {
              return a.weight < b.weight;
            });

  return shortest_paths_vector;
}

void KWeightedShortestPaths::KWeightedShortestPaths(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result,
                                                    mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  mgp::List paths{};
  try {
    auto start_node = arguments[0];
    auto end_node = arguments[1];
    auto number_of_weighted_shortest_paths = arguments[2];
    auto weight_name = arguments[3];

    if (!start_node.IsNode()) {
      throw mgp::ValueException("Start node needs to be a node!");
    }
    if (!end_node.IsNode()) {
      throw mgp::ValueException("End node needs to be a node!");
    }
    if (!number_of_weighted_shortest_paths.IsInt() or number_of_weighted_shortest_paths.ValueInt() < 1) {
      throw mgp::ValueException("Number of weighted shortest paths needs to be an integer and bigger then 0!");
    }
    if (!weight_name.IsString()) {
      throw mgp::ValueException("Weight name needs to be a string!");
    }
    auto source = start_node.ValueNode();
    auto sync = end_node.ValueNode();
    auto k = number_of_weighted_shortest_paths.ValueInt();
    auto weight = weight_name.ValueString();

    mgp::Graph graph{memgraph_graph};

    auto k_shortest_paths = KWeightedShortestPaths::YenKSP(graph, source, sync, k, weight);

    for (auto path : k_shortest_paths) {
      mgp::Map path_map{};
      path_map.Insert("weight", mgp::Value(path.weight));
      mgp::List path_list;
      for (auto node : path.vertices) {
        path_list.AppendExtend(mgp::Value(node));
      }
      path_map.Insert("path", mgp::Value(path_list));

      paths.AppendExtend(mgp::Value(path_map));
    }
    auto record = record_factory.NewRecord();
    record.Insert(KWeightedShortestPaths::kResultPaths, paths);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
