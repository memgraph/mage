#pragma once

#include <mgp.hpp>
#include <unordered_set>
namespace Path {
    constexpr std::string_view kProcedureExpand = "expand";
    constexpr std::string_view kArgumentStartExpand = "start";
    constexpr std::string_view kArgumentRelationshipsExpand = "relationships";
    constexpr std::string_view kArgumentLabelsExpand = "labels";
    constexpr std::string_view kArgumentMinHopsExpand= "min_hops";
    constexpr std::string_view kArgumentMaxHopsExpand = "max_hops";
    constexpr std::string_view kResultExpand = "result";

    void Expand(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

    void FilterLabel(const std::string_view label, const std::unordered_set<std::string> &termination_list, const std::unordered_set<std::string> &blacklist,
    const std::unordered_set<std::string> &whitelist, const std::unordered_set<std::string> &end_list,
    bool &blacklisted, bool &terminated, bool &end_node, bool &whitelisted);

    void ParseLabels(const mgp::List &list_of_labels,std::unordered_set<std::string> &termination_list, std::unordered_set<std::string> &blacklist,
    std::unordered_set<std::string> &whitelist, std::unordered_set<std::string> &end_list);

    void Parse_Relationships(const mgp::List &list_of_relationships, std::unordered_set<std::string> &outgoing_rel, std::unordered_set<std::string> &incoming_rel
    ,std::unordered_set<std::string> &any_rel, bool &any_outgoing, bool &any_incoming);

    void Path_DFS(mgp::Path path, std::unordered_set<mgp::Relationship> relationships_set, std::vector<mgp::Path> &path_list,
     size_t path_size, const int64_t min_hops, const int64_t max_hops,
    const std::unordered_set<std::string> &termination_list, const std::unordered_set<std::string> &blacklist,
    const std::unordered_set<std::string> &whitelist, const std::unordered_set<std::string> &end_list, 
    const bool &end_node_activated, const bool &whitelist_empty, const bool &termination_activated,
    std::unordered_set<std::string> &outgoing_rel, std::unordered_set<std::string> &incoming_rel
    ,std::unordered_set<std::string> &any_rel, bool &any_outgoing, bool &any_incoming);
}  // namespace Path