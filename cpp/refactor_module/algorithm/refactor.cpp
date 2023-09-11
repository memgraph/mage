#include "refactor.hpp"

void Refactor::InvertRel(mgp::Graph &graph, mgp::Relationship &rel) {
  const auto old_from = rel.From();
  const auto old_to = rel.To();
  graph.SetFrom(rel, old_to);
  graph.SetTo(rel, old_from);
}

void Refactor::Invert(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph = mgp::Graph(memgraph_graph);
    mgp::Relationship rel = arguments[0].ValueRelationship();

    InvertRel(graph, rel);
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kReturnIdInvert).c_str(), rel.Id().AsInt());
    record.Insert(std::string(kReturnRelationshipInvert).c_str(), rel);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}