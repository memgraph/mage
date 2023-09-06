#pragma once

#include <mgp.hpp>
#include <unordered_map>
#include <unordered_set>

namespace Path {

struct LabelSets {
  std::unordered_set<std::string_view> termination_list;
  std::unordered_set<std::string_view> blacklist;
  std::unordered_set<std::string_view> whitelist;
  std::unordered_set<std::string_view> end_list;
};

constexpr std::string_view kReturnSubgraphNodes = "nodes";
constexpr std::string_view kProcedureSubgraphNodes = "subgraph_nodes";
constexpr std::string_view kArgumentsStart = "start_node";
constexpr std::string_view kArgumentsConfig = "config";
constexpr std::string_view kResultSubgraphNodes = "nodes";

constexpr std::string_view kReturnNodesSubgraphAll = "nodes";
constexpr std::string_view kReturnRelsSubgraphAll = "rels";
constexpr std::string_view kProcedureSubgraphAll = "subgraph_all";
constexpr std::string_view kResultNodesSubgraphAll = "nodes";
constexpr std::string_view kResultRelsSubgraphAll = "rels";

void SubgraphNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void SubgraphAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void VisitNode(const mgp::Node node, std::map<mgp::Node, std::int64_t> &visited_nodes, bool is_start,
               const mgp::Map &config, int64_t hop_count, Path::LabelSets &labelFilterSets,
               mgp::List &to_be_returned_nodes);
void ParseLabels(const mgp::List &list_of_labels, LabelSets &labelSets);

}  // namespace Path
