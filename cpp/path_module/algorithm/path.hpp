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

enum RelDirection { kNone = -1, kAny = 0, kIncoming = 1, kOutgoing = 2, kBoth = 3 };

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

  RelDirection GetDirection(std::string &rel_type);
  LabelBools GetLabelBools(const mgp::Node &node);

  bool AnyDirected(bool outgoing) const { return outgoing ? config_.any_outgoing : config_.any_incoming; }
  bool FilterNodes(bool is_start) const { return (config_.filter_start_node || !is_start); }
  bool FilterRelationships(bool is_start) const { return (config_.begin_sequence_at_start || !is_start); }

  bool AreLabelsValid(const LabelBools &label_bools) const;
  bool ContinueExpanding(const LabelBools &label_bools, size_t path_size) const;

  bool PathSizeOk(int64_t path_size) const;
  bool PathTooBig(int64_t path_size) const;
  bool Whitelisted(bool whitelisted) const;

  // methods for parsing config
  void FilterLabelBoolStatus();
  void FilterLabel(std::string_view label, LabelBools &label_bools);
  void ParseLabels(const mgp::List &list_of_labels);
  void ParseRelationships(const mgp::List &list_of_relationships);

 private:
  Config config_;
};

class PathData {
 public:
  explicit PathData(const PathHelper &helper, const mgp::RecordFactory &record_factory, const mgp::Graph &graph)
      : helper_(helper), record_factory_(record_factory), graph_(graph) {}

 protected:
  PathHelper helper_;
  const mgp::RecordFactory &record_factory_;
  const mgp::Graph &graph_;
  std::unordered_set<int64_t> visited_;
};

class PathExpand : PathData {
 public:
  explicit PathExpand(const PathHelper &helper, const mgp::RecordFactory &record_factory, const mgp::Graph &graph)
      : PathData(helper, record_factory, graph) {}

  void ExpandPath(mgp::Path &path, const mgp::Relationship &relationship, int64_t path_size);
  void Expand(mgp::Path &path, mgp::Relationships relationships, bool outgoing, int64_t path_size,
              std::set<std::pair<std::string_view, int64_t>> &seen);
  void StartAlgorithm(mgp::Node node);
  void Parse(const mgp::Value &value);
  void DFS(mgp::Path &path, int64_t path_size);
};

class PathSubgraph : PathData {
 public:
  explicit PathSubgraph(const PathHelper &helper, const mgp::RecordFactory &record_factory, const mgp::Graph &graph)
      : PathData(helper, record_factory, graph) {}

  struct Pair {
    mgp::Node node;
    int64_t hop_count;
  };

  void Expand(Pair &pair, mgp::Relationships relationships, bool outgoing, std::queue<Pair> &queue,
              std::set<std::pair<std::string_view, int64_t>> &seen);
  void Parse(const mgp::Value &value);
  void InsertNode(const mgp::Node &node, int64_t hop_count, LabelBools &label_bools);
  mgp::List BFS();

 private:
  std::unordered_set<mgp::Node> start_nodes_;
  mgp::List to_be_returned_nodes_;
};

void Create(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void Expand(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void SubgraphNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void SubgraphAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Path
