#include "text.hpp"
#include "mgp.hpp"

void Text::Join(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    auto relationship{arguments[0].ValueRelationship()};
    const auto node{arguments[1].ValueNode()};

    mgp::Graph graph{memgraph_graph};
    graph.ChangeRelationshipFrom(relationship, node);

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultJoin).c_str(), relationship);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
