#include "create.hpp"
#include "_mgp.hpp"
#include "mg_procedure.h"
#include "mgp.hpp"

void Create::Relationship(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    auto node_from{arguments[0].ValueNode()};
    const auto relationship_type{arguments[1].ValueString()};
    const auto properties{arguments[2].ValueMap()};
    auto node_to{arguments[3].ValueNode()};

    mgp::Graph graph{memgraph_graph};
    auto relationship = graph.CreateRelationship(node_from, node_to, relationship_type);

    for (const auto &map_item : properties) {
      relationship.SetProperty(std::string(map_item.key), map_item.value);
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRelationship).c_str(), relationship);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
