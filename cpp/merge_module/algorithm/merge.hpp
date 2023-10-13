#pragma once

#include <mgp.hpp>
#include <unordered_map>
#include <unordered_set>

namespace Merge {

/* node constants */
constexpr std::string_view kProcedureNode = "node";
constexpr std::string_view kNodeArg1 = "labels";
constexpr std::string_view kNodeArg2 = "identProps";
constexpr std::string_view kNodeArg3 = "createProps";
constexpr std::string_view kNodeArg4 = "matchProps";
constexpr std::string_view kNodeRes = "node";

void Node(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
bool IdentProp(const mgp::Map &identProp, const mgp::Node &node);
bool LabelsContained(const std::unordered_set<std::string_view> &labels, const mgp::Node &node);
}  // namespace Merge
