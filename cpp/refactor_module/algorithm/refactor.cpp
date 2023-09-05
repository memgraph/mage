#include "refactor.hpp"

#include "mgp.hpp"

void Refactor::From(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Relationship relationship{arguments[0].ValueRelationship()};
    const mgp::Node new_from{arguments[1].ValueNode()};
    mgp::Graph graph{memgraph_graph};

    graph.ChangeRelationshipFrom(relationship, new_from);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Refactor::To(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Relationship relationship{arguments[0].ValueRelationship()};
    const mgp::Node new_to{arguments[1].ValueNode()};
    mgp::Graph graph{memgraph_graph};

    graph.ChangeRelationshipTo(relationship, new_to);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}