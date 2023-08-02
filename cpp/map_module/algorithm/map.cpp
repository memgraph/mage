#include "map.hpp"
#include "mg_utils.hpp"
#include "mgp.hpp"

void Map::FromNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto label{arguments[0].ValueString()};
    const auto property{arguments[1].ValueString()};
    mgp::Map map_result{};

    const auto all_nodes = mgp::Graph(memgraph_graph).Nodes();
    for (const auto node : all_nodes) {
      if (!node.HasLabel(label) || !node.Properties().contains(std::string(property))) continue;

      std::ostringstream oss;
      oss << node.GetProperty(std::string(property));
      const auto key = oss.str();

      mgp::Map map{};
      map.Update("identity", mgp::Value(node.Id().AsInt()));

      mgp::List labels{};
      for (const auto &label : node.Labels()) {
        labels.AppendExtend(mgp::Value(label));
      }
      map.Update("labels", mgp::Value(std::move(labels)));

      const auto property_map = node.Properties();
      mgp::Map properties{};
      for (const auto &[key, value] : property_map) {
        properties.Insert(key, value);
      }
      map.Update("properties", mgp::Value(std::move(properties)));

      map_result.Update(key, mgp::Value(std::move(map)));
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultFromNodes).c_str(), map_result);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
