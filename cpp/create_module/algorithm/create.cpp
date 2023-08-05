#include "create.hpp"

void Create::SetElementProp(mgp::Relationship &element, const mgp::List &prop_key_list,
                            const mgp::List &prop_value_list, const mgp::RecordFactory &record_factory) {
  for (size_t i = 0; i < prop_key_list.Size(); i++) {
    element.SetProperty(std::string(prop_key_list[i].ValueString()), prop_value_list[i]);
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRelProp).c_str(), std::move(element));
  }
}

void Create::ProcessElement(const mgp::Value &element, const mgp::Graph graph, const mgp::List &prop_key_list,
                            const mgp::List &prop_value_list, const mgp::RecordFactory &record_factory,
                            std::unordered_set<mgp::Id> &relIds) {
  if (element.IsRelationship()) {
    auto rel = element.ValueRelationship();
    SetElementProp(rel, prop_key_list, prop_value_list, record_factory);
    return;
  }
  if (element.IsInt()) {
    relIds.insert(mgp::Id::FromInt(element.ValueInt()));
    return;
  }
  throw mgp::ValueException("First argument must be node, relationship, id or a list of those.");
}

void Create::SetRelProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::List prop_key_list = mgp::List();
    prop_key_list.AppendExtend(arguments[1]);
    mgp::List prop_value_list = mgp::List();
    prop_value_list.AppendExtend(arguments[2]);

    std::unordered_set<mgp::Id> relIds;

    if (arguments[0].IsList()) {
      for (const auto element : arguments[0].ValueList()) {
        ProcessElement(element, graph, prop_key_list, prop_value_list, record_factory, relIds);
      }
    } else {
      ProcessElement(arguments[0], graph, prop_key_list, prop_value_list, record_factory, relIds);
    }
    for (auto rel : graph.Relationships()) {
      if (relIds.contains(rel.Id())) {
        SetElementProp(rel, prop_key_list, prop_value_list, record_factory);
      }
    }
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
