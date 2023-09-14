#pragma once

#include <unordered_set>

#include <mgp.hpp>

namespace Algo {

class PathFinder {
 public:
  PathFinder(const mgp::Node &start_node, const mgp::Node &end_node, int64_t max_nodes);

  void UpdateRelationshipDirection(const mgp::List &relationship_types);
  void DFS(const mgp::Node &curr_node, mgp::Path &curr_path, std::unordered_set<int64_t> &visited);
  std::vector<mgp::Path> FindAllPaths();

 private:
  const mgp::Node _start_node;
  const mgp::Id _end_node_id;
  std::unordered_map<std::string_view, uint8_t> _rel_direction;
  const int64_t _max_nodes;
  std::vector<mgp::Path> _paths;
};

/* sum_longs constants */
constexpr const std::string_view kProcedureAllSimplePaths = "all_simple_paths";
constexpr const std::string_view kAllSimplePathsArg1 = "start_node";
constexpr const std::string_view kAllSimplePathsArg2 = "end_node";
constexpr const std::string_view kAllSimplePathsArg3 = "relationship_types";
constexpr const std::string_view kAllSimplePathsArg4 = "max_nodes";
constexpr const std::string_view kResultAllSimplePaths = "path";

void AllSimplePaths(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Algo
