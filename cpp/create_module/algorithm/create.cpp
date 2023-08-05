#include "create.hpp"

void Create::RemoveElementLabels(mgp::Node &element, const mgp::List &labels,
                                 const mgp::RecordFactory &record_factory) {
  for (auto label : labels) {
    element.RemoveLabel(label.ValueString());
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRemoveLabels).c_str(), std::move(element));
  }
}

void Create::ProcessElement(const mgp::Value &element, const mgp::Graph graph, const mgp::List &labels,
                            const mgp::RecordFactory &record_factory) {
  if (element.IsNode()) {
    auto node = element.ValueNode();
    RemoveElementLabels(node, labels, record_factory);
    return;
  }
  if (element.IsInt()) {
    auto node = graph.GetNodeById(mgp::Id::FromInt(element.ValueInt()));
    RemoveElementLabels(node, labels, record_factory);
    return;
  }
  throw mgp::ValueException("First argument must be type node, relationship, id or a list of those.");
}

void Create::RemoveLabels(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto labels = arguments[1].ValueList();

    if (arguments[0].IsList()) {
      for (const auto element : arguments[0].ValueList()) {
        ProcessElement(element, graph, labels, record_factory);
      }
      return;
    }
    ProcessElement(arguments[0], graph, labels, record_factory);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
