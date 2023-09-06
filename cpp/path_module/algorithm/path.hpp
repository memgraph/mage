#pragma once

#include <mgp.hpp>

#include <unordered_map>
#include <unordered_set>

namespace Path {

/* create constants */
constexpr const std::string_view kProcedureCreate = "create";
constexpr const std::string_view kCreateArg1 = "start_node";
constexpr const std::string_view kCreateArg2 = "relationships";
constexpr const std::string_view kResultCreate = "path";

/* expand constants */
constexpr std::string_view kProcedureExpand = "expand";
constexpr std::string_view kArgumentStartExpand = "start";
constexpr std::string_view kArgumentRelationshipsExpand = "relationships";
constexpr std::string_view kArgumentLabelsExpand = "labels";
constexpr std::string_view kArgumentMinHopsExpand = "min_hops";
constexpr std::string_view kArgumentMaxHopsExpand = "max_hops";
constexpr std::string_view kResultExpand = "result";

/* subgraph_nodes constants */
constexpr std::string_view kReturnSubgraphNodes = "nodes";
constexpr std::string_view kProcedureSubgraphNodes = "subgraph_nodes";
constexpr std::string_view kArgumentsStart = "start_node";
constexpr std::string_view kArgumentsConfig = "config";
constexpr std::string_view kResultSubgraphNodes = "nodes";

/* subgraph_all constants */
constexpr std::string_view kReturnNodesSubgraphAll = "nodes";
constexpr std::string_view kReturnRelsSubgraphAll = "rels";
constexpr std::string_view kProcedureSubgraphAll = "subgraph_all";
constexpr std::string_view kResultNodesSubgraphAll = "nodes";
constexpr std::string_view kResultRelsSubgraphAll = "rels";

struct LabelSets {
  std::unordered_set<std::string_view> termination_list;
  std::unordered_set<std::string_view> blacklist;
  std::unordered_set<std::string_view> whitelist;
  std::unordered_set<std::string_view> end_list;
};

struct LabelBools {
  bool blacklisted = false;
  bool terminated = false;
  bool end_node = false;
  bool whitelisted = false;
};

struct LabelBoolsStatus {
  bool end_node_activated = false;
  bool whitelist_empty = false;
  bool termination_activated = false;
};

struct RelationshipSets {
  std::unordered_set<std::string> outgoing_rel;
  std::unordered_set<std::string> incoming_rel;
  std::unordered_set<std::string> any_rel;
};

void Create(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void FilterLabelBoolStatus(const LabelSets &labelSets, LabelBoolsStatus &labelStatus);

bool ShouldExpand(const LabelBools &labelBools, const LabelBoolsStatus &labelStatus);

void Expand(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void FilterLabel(const std::string_view label, const LabelSets &labelFilters, LabelBools &labelBools);

void ParseLabels(const mgp::List &list_of_labels, LabelSets &labelFilters);

void ParseRelationships(const mgp::List &list_of_relationships, RelationshipSets &relationshipSets, bool &any_outgoing,
                        bool &any_incoming);
bool PathSizeOk(const int64_t path_size, const int64_t &max_hops, const int64_t &min_hops);

bool RelationshipAllowed(const std::string &rel_type, const RelationshipSets &relationshipSets, bool &any_outgoing,
                         bool &any_incoming, bool outgoing);

bool Whitelisted(const bool &whitelisted, const bool &whitelist_empty);

void PathDFS(mgp::Path path, std::unordered_set<mgp::Relationship> &relationships_set,
             const mgp::RecordFactory &record_factory, int64_t path_size, const int64_t min_hops,
             const int64_t max_hops, const LabelSets &labelFilters, const LabelBoolsStatus &labelStatus,
             const RelationshipSets &relationshipSets, bool &any_outgoing, bool &any_incoming);

void DfsByDirection(mgp::Path &path, std::unordered_set<mgp::Relationship> &relationships_set,
                    const mgp::RecordFactory &record_factory, int64_t path_size, const int64_t min_hops,
                    const int64_t max_hops, const LabelSets &labelFilters, const LabelBoolsStatus &labelStatus,
                    const RelationshipSets &relationshipSets, bool &any_outgoing, bool &any_incoming, bool outgoing);

void StartFunction(const mgp::Node &node, const mgp::RecordFactory &record_factory, int64_t path_size,
                   const int64_t min_hops, const int64_t max_hops, const LabelSets &labelSets,
                   const LabelBoolsStatus &labelStatus, const RelationshipSets &relationshipSets, bool &any_outgoing,
                   bool &any_incoming);

void SubgraphNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void SubgraphAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void VisitNode(const mgp::Node node, std::map<mgp::Node, std::int64_t> &visited_nodes, bool is_start,
               const mgp::Map &config, int64_t hop_count, Path::LabelSets &labelFilterSets,
               mgp::List &to_be_returned_nodes);

}  // namespace Path
