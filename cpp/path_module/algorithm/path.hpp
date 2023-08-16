#pragma once

#include <mgp.hpp>
#include <unordered_set>
namespace Path {
constexpr std::string_view kProcedureExpand = "expand";
constexpr std::string_view kArgumentStartExpand = "start";
constexpr std::string_view kArgumentRelationshipsExpand = "relationships";
constexpr std::string_view kArgumentLabelsExpand = "labels";
constexpr std::string_view kArgumentMinHopsExpand = "min_hops";
constexpr std::string_view kArgumentMaxHopsExpand = "max_hops";
constexpr std::string_view kResultExpand = "result";

struct LabelSets {
  std::unordered_set<std::string> termination_list;
  std::unordered_set<std::string> blacklist;
  std::unordered_set<std::string> whitelist;
  std::unordered_set<std::string> end_list;
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
}  // namespace Path
