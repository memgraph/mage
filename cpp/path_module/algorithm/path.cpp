#include "path.hpp"

#include <list>

void getStartNodes(const mgp::Value element, const mgp::Graph &graph, std::list<mgp::Node> &startNodes) {
  if (!(element.IsNode() || element.IsInt())) {
    throw mgp::ValueException("First argument must be type node, id or a list of those.");
  }
  if (element.IsNode()) {
    startNodes.push_back(element.ValueNode());
    return;
  }
  startNodes.push_back(graph.GetNodeById(mgp::Id::FromInt(element.ValueInt())));
}

void Path::SubgraphAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::List returnNodes = mgp::List();
    mgp::List returnRels = mgp::List();

    const auto config = arguments[1].ValueMap();
    std::list<mgp::Node> startNodes;
    if (arguments[0].IsList()) {
      for (const auto element : arguments[0].ValueList()) {
        getStartNodes(element, graph, startNodes);
      }
    } else {
      getStartNodes(arguments[0], graph, startNodes);
    }

    for (const auto node : startNodes) {
      returnNodes.AppendExtend(mgp::Value(node));
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultSubgraphAll).c_str(), std::move(returnNodes));
    record.Insert(std::string(kResultSubgraphAll).c_str(), std::move(returnRels));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
