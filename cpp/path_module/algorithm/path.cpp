#include "path.hpp"

void GetStartNodes(const mgp::Value element, const mgp::Graph &graph, std::unordered_set<mgp::Node> &start_nodes) {
  if (!(element.IsNode() || element.IsInt())) {
    throw mgp::ValueException("First argument must be type node, id or a list of those.");
  }
  if (element.IsNode()) {
    start_nodes.insert(element.ValueNode());
    return;
  }
  start_nodes.insert(graph.GetNodeById(mgp::Id::FromInt(element.ValueInt())));
}

bool RelFilterAllows(const mgp::Map &config, std::string_view type, bool ingoing) {
  mgp::List list_of_types = config.At("relationshipFilter").ValueList();
  if (list_of_types.Size() == 0) {
    return true;
  }
  for (const auto element : list_of_types) {
    auto string_rel_type = element.ValueString();
    if (string_rel_type.front() == '<' && ingoing) {
      string_rel_type.remove_prefix(1);
    }
    if (string_rel_type.back() == '>' && !ingoing) {
      string_rel_type.remove_suffix(1);
    }
    if (string_rel_type == type || string_rel_type.size() == 0) {
      return true;
    }
  }
  return false;
}

void Path::VisitNode(const mgp::Node node, std::unordered_map<mgp::Node, std::int64_t> &visited_nodes, bool is_start,
                     const mgp::Map &config, int64_t hop_count, const mgp::RecordFactory &record_factory) {
  if (hop_count > config.At("maxLevel").ValueInt()) {
    return;
  }
  try {
    if (visited_nodes.at(node) <= hop_count) {
      return;
    }
  } catch (const std::out_of_range &e) {
    // it's okay, the node is not in visited nodes map
    if (!is_start || config.At("minLevel").ValueInt() != 1) {  // what if they send me not int or nothing at all
      auto record = record_factory.NewRecord();
      record.Insert(std::string(kResultSubgraphNodes).c_str(), node);
    }
  }
  visited_nodes.insert({node, hop_count});
  for (const auto in_rel : node.InRelationships()) {
    if (RelFilterAllows(config, in_rel.Type(), true)) {
      VisitNode(in_rel.From(), visited_nodes, false, config, hop_count + 1, record_factory);
    }
  }
  for (const auto out_rel : node.OutRelationships()) {
    if (RelFilterAllows(config, out_rel.Type(), false)) {
      VisitNode(out_rel.To(), visited_nodes, false, config, hop_count + 1, record_factory);
    }
  }
}

void Path::SubgraphNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto config = arguments[1].ValueMap();

    std::unordered_set<mgp::Node> start_nodes;
    if (arguments[0].IsList()) {
      for (const auto element : arguments[0].ValueList()) {
        GetStartNodes(element, graph, start_nodes);
      }
    } else {
      GetStartNodes(arguments[0], graph, start_nodes);
    }

    std::unordered_map<mgp::Node, std::int64_t> visited_nodes;
    for (const auto node : start_nodes) {
      VisitNode(node, visited_nodes, true, config, 0, record_factory);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
