#pragma once

#include <unordered_set>

#include <mgp.hpp>
#include <unordered_set>

namespace Algo {

enum class RelDirection { kNone = -1, kAny = 0, kIncoming = 1, kOutgoing = 2, kBoth = 3 };

class PathFinder {
 public:
  PathFinder(const mgp::Node &start_node, const mgp::Node &end_node, int64_t max_length, const mgp::List &rel_types,
             const mgp::RecordFactory &record_factory);

  RelDirection GetDirection(const std::string &rel_type) const;

  void UpdateRelationshipDirection(const mgp::List &relationship_types);
  void DFS(const mgp::Node &curr_node, mgp::Path &curr_path, std::unordered_set<int64_t> &visited);
  void FindAllPaths();

 private:
  const mgp::Node start_node_;
  const mgp::Id end_node_id_;
  const int64_t max_length_;
  bool any_incoming_;
  bool any_outgoing_;

  std::unordered_map<std::string_view, RelDirection> rel_direction_;
  const mgp::RecordFactory &record_factory_;
};

/* all_simple_paths constants */
constexpr const std::string_view kProcedureAllSimplePaths = "all_simple_paths";
constexpr const std::string_view kAllSimplePathsArg1 = "start_node";
constexpr const std::string_view kAllSimplePathsArg2 = "end_node";
constexpr const std::string_view kAllSimplePathsArg3 = "relationship_types";
constexpr const std::string_view kAllSimplePathsArg4 = "max_length";
constexpr const std::string_view kResultAllSimplePaths = "path";

/* cover constants */
constexpr std::string_view kCoverRet1 = "rel";

void AllSimplePaths(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Cover(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Algo
