#pragma once

#include <sys/types.h>
#include <mgp.hpp>
#include <string_view>

namespace KWeightedShortestPaths {

/* KWeightedShortestPath constants */
constexpr const char *kProcedure = "k_weighted_shortest_paths";
constexpr const char *kArgumentStartNode = "start_node";
constexpr const char *kArgumentEndNode = "end_node";
constexpr const char *kArgumentNumberOfWeightedShortestPaths = "number_of_weighted_shortest_paths";
constexpr const char *kArgumentWeightName = "weight_property_name";
constexpr const char *kResultPaths = "paths";
constexpr const int64_t kDefaultNumberOfWeightedShortestPaths = 5;
constexpr const char *kDefaultWeightName = "weight";

struct TempPath {
  double weight;
  std::vector<mgp::Node> vertices;
};

struct ComparePaths {
  bool operator()(const TempPath &path1, const TempPath &path2) const {
    if (path1.vertices.size() != path2.vertices.size()) {
      return path1.vertices.size() < path2.vertices.size();
    }

    for (size_t i = 0; i < path1.vertices.size(); ++i) {
      if (path1.vertices[i].Id().AsUint() != path2.vertices[i].Id().AsUint()) {
        return path1.vertices[i].Id().AsUint() < path2.vertices[i].Id().AsUint();
      }
    }
    // Compare weights with a tolerance
    double tolerance = 1e-9;
    return std::abs(path1.weight - path2.weight) > tolerance;
  }
};

struct CompareWeightDistance {
  bool operator()(std::pair<mgp::Node, double> const &n1, std::pair<mgp::Node, double> const &n2) {
    return n1.second > n2.second;
  }
};
struct CompareWeightPath {
  bool operator()(TempPath const &p1, TempPath const &p2) { return p1.weight > p2.weight; }
};

struct DijkstraResult {
  TempPath path;
  std::unordered_map<uint64_t, double> distances;
};

double GetEdgeWeight(mgp::Node &node1, mgp::Node &node2, const std::string_view &weight_name);

DijkstraResult Dijkstra(mgp::Graph &graph, mgp::Node &source, mgp::Node &sink, const std::string_view &weight_name,
                        const std::set<uint64_t> &ignore_nodes = {},
                        const std::set<std::pair<uint64_t, uint64_t>> &ignore_edges = {});

void KWeightedShortestPaths(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

std::vector<TempPath> YenKSP(mgp::Graph &graph, mgp::Node &source, mgp::Node &sink, int K,
                             const std::string_view &weight_name);

}  // namespace KWeightedShortestPaths
