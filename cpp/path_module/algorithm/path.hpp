#pragma once

#include <mgp.hpp>

#include <queue>
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
  bool blacklisted = false;  // no node in the path will be blacklisted
  bool terminated = false;   // returned paths end with a termination node but don't continue to be expanded further,
                             // takes precedence over end nodes
  bool end_node = false;     // returned paths end with an end node but continue to be expanded further
  bool whitelisted = false;  // all nodes in the path will be whitelisted (except end and termination nodes)
  // end and termination nodes don't have to respect whitelists and blacklists
};

struct LabelBoolsStatus {
  bool end_node_activated = false;  // true if there is an end node -> only paths ending with it can be saved as result,
                                    // but they can be expanded further
  bool whitelist_empty = false;     // true if no whitelist is given -> all nodes are whitelisted
  bool termination_activated = false;  // true if there is a termination node -> only paths ending with it are allowed
};

enum class RelDirection { kNone = -1, kAny = 0, kIncoming = 1, kOutgoing = 2, kBoth = 3 };

struct Config {
  LabelBoolsStatus label_bools_status;
  std::unordered_map<std::string, RelDirection> relationship_sets;
  LabelSets label_sets;
  int64_t min_hops = 0;
  int64_t max_hops = std::numeric_limits<int64_t>::max();
  bool any_incoming = false;
  bool any_outgoing = false;
  bool filter_start_node = true;
  bool begin_sequence_at_start = true;
  bool bfs = false;
};

class PathHelper {
 public:
  explicit PathHelper(const mgp::List &labels, const mgp::List &relationships, int64_t min_hops, int64_t max_hops);
  explicit PathHelper(const mgp::Map &config);

  RelDirection GetDirection(std::string &rel_type) const;
  LabelBools GetLabelBools(const mgp::Node &node) const;

  bool AnyDirected(bool outgoing) const { return outgoing ? config_.any_outgoing : config_.any_incoming; }
  bool IsNotStartOrSupportsStartNode(bool is_start) const { return (config_.filter_start_node || !is_start); }
  bool IsNotStartOrSupportsStartRel(bool is_start) const { return (config_.begin_sequence_at_start || !is_start); }

  bool AreLabelsValid(const LabelBools &label_bools) const;
  bool ContinueExpanding(const LabelBools &label_bools, size_t path_size) const;

  bool PathSizeOk(int64_t path_size) const;
  bool PathTooBig(int64_t path_size) const;
  bool Whitelisted(bool whitelisted) const;

  // methods for parsing config
  void FilterLabelBoolStatus();
  void FilterLabel(std::string_view label, LabelBools &label_bools) const;
  void ParseLabels(const mgp::List &list_of_labels);
  void ParseRelationships(const mgp::List &list_of_relationships);

 private:
  Config config_;
};

class PathExpand {
 public:
  explicit PathExpand(PathHelper &&helper, const mgp::RecordFactory &record_factory, const mgp::Graph &graph)
      : helper_(helper), record_factory_(record_factory), graph_(graph) {}

  void ExpandPath(mgp::Path &path, const mgp::Relationship &relationship, int64_t path_size);
  void ExpandFromRelationships(mgp::Path &path, mgp::Relationships relationships, bool outgoing, int64_t path_size,
                               std::set<std::pair<std::string_view, int64_t>> &seen);
  void StartAlgorithm(mgp::Node node);
  void Parse(const mgp::Value &value);
  void DFS(mgp::Path &path, int64_t path_size);
  void RunAlgorithm();

 private:
  PathHelper helper_;
  const mgp::RecordFactory &record_factory_;
  const mgp::Graph &graph_;
  std::unordered_set<int64_t> visited_;
  std::unordered_set<mgp::Node> start_nodes_;
};

class PathSubgraph {
 public:
  explicit PathSubgraph(PathHelper &&helper, const mgp::RecordFactory &record_factory, const mgp::Graph &graph)
      : helper_(helper), record_factory_(record_factory), graph_(graph) {}

  void ExpandFromRelationships(const std::pair<mgp::Node, int64_t> &pair, mgp::Relationships relationships,
                               bool outgoing, std::queue<std::pair<mgp::Node, int64_t>> &queue,
                               std::set<std::pair<std::string_view, int64_t>> &seen);
  void Parse(const mgp::Value &value);
  void TryInsertNode(const mgp::Node &node, int64_t hop_count, LabelBools &label_bools);
  mgp::List BFS();

 private:
  PathHelper helper_;
  const mgp::RecordFactory &record_factory_;
  const mgp::Graph &graph_;
  std::unordered_set<int64_t> visited_;
  std::unordered_set<mgp::Node> start_nodes_;
  mgp::List to_be_returned_nodes_;
};

void Create(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Expand(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void SubgraphNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void SubgraphAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Path
