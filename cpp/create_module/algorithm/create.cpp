#include "create.hpp"

void Create::SetRelProperty(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    std::set<mgp::Id> relIds;

    if (arguments[0].IsList()) {
      for (const auto element : arguments[0].ValueList()) {
        if (element.IsRelationship()) {
          relIds.insert(std::move(element.ValueRelationship().Id()));
        } else if (element.IsInt()) {
          relIds.insert(std::move(mgp::Id::FromInt(element.ValueInt())));
        } else {
          throw mgp::ValueException("First argument must be a relationsip, relationship's id or a list of those.");
        }
      }
    } else if (arguments[0].IsRelationship()) {
      relIds.insert(std::move(arguments[0].ValueRelationship().Id()));
    } else if (arguments[0].IsInt()) {
      relIds.insert(std::move(mgp::Id::FromInt(arguments[0].ValueInt())));
    } else {
      throw mgp::ValueException("First argument must be a relationsip, relationship's id or a list of those.");
    }
    // this part of code is very similar so I could extract it in a function but then lots of type casting has to be
    // done so I'm not sure if it is worth it?

    const auto prop_key = arguments[1].ValueString();
    const auto prop_value = arguments[2];

    for (auto rel : graph.Relationships()) {
      if (relIds.contains(rel.Id())) {
        rel.SetProperty((std::string)prop_key, prop_value);
        auto record = record_factory.NewRecord();
        record.Insert(std::string(kResultRelProp).c_str(), std::move(rel));
      }
    }
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
