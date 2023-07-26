#include "map.hpp"
#include "mg_utils.hpp"
#include "mgp.hpp"

void Map::FromNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto &label{arguments[0].ValueString()};
    const auto &property{arguments[1].ValueString()};
    mgp::Map map_result{};

    const auto &all_nodes = mgp::Graph(memgraph_graph).Nodes();
    for(const auto &node : all_nodes){
      if(!node.HasLabel(label))
          continue;

      mgp::Map map{};
      const mgp::Value id_value{node.Id().AsInt()};
      map.Update("identity", id_value);

      mgp::List labels{};
      for(const auto &label : node.Labels()){
          labels.AppendExtend(mgp::Value(label));
      }
      const mgp::Value labels_value{std::move(labels)};
      map.Update("labels", labels_value);

      const auto property_map = node.Properties();
      mgp::Map properties{};
      for (const auto &[key, value] : property_map) {
        properties.Insert(key, value);
      }
      const mgp::Value properties_value{std::move(properties)};
      map.Update("properties", properties_value);

      const mgp::Value map_value{std::move(map)};
      map_result.Update(node.GetProperty(std::string(property)).ValueString(), map_value);
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultFromNodes).c_str(), map_result);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
