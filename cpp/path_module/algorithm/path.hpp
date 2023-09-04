#pragma once

#include <mgp.hpp>
#include <unordered_map>
#include <unordered_set>

namespace Path {

constexpr std::string_view kReturnSubgraphNodes = "nodes";
constexpr std::string_view kProcedureSubgraphNodes = "subgraph_nodes";
constexpr std::string_view kArgumentsStart = "start_node";
constexpr std::string_view kArgumentsConfig = "config";
constexpr std::string_view kResultSubgraphNodes = "nodes";

void SubgraphNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void VisitNode(const mgp::Node node, std::unordered_map<mgp::Node, std::int64_t> &visited_nodes, bool is_start,
               const mgp::Map &config, int64_t hop_count, const mgp::RecordFactory &record_factory);

}  // namespace Path
