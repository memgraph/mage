#pragma once

#include <mgp.hpp>

#include <unordered_map>
#include <unordered_set>

namespace Merge {

/* relationship constants */
constexpr const std::string_view kProcedureRelationship = "relationship";
constexpr const std::string_view kRelationshipArg1 = "startNode";
constexpr const std::string_view kRelationshipArg2 = "relationshipType";
constexpr const std::string_view kRelationshipArg3 = "identProps";
constexpr const std::string_view kRelationshipArg4 = "createProps";
constexpr const std::string_view kRelationshipArg5 = "endNode";
constexpr const std::string_view kRelationshipArg6 = "matchProps";
constexpr const std::string_view kRelationshipResult = "rel";

/* node constants */
constexpr std::string_view kProcedureNode = "node";
constexpr std::string_view kNodeArg1 = "labels";
constexpr std::string_view kNodeArg2 = "identProps";
constexpr std::string_view kNodeArg3 = "createProps";
constexpr std::string_view kNodeArg4 = "matchProps";
constexpr std::string_view kNodeRes = "node";

void Relationship(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Node(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
bool IdentProp(const mgp::Map &identProp, const mgp::Node &node);
bool LabelsContained(const std::unordered_set<std::string_view> &labels, const mgp::Node &node);

}  // namespace Merge
