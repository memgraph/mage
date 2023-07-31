#include "map.hpp"
#include "mg_utils.hpp"
#include "mgp.hpp"

namespace {
std::string value_to_string(const mgp::Value &value) {
  switch (value.Type()) {
    case mgp::Type::Null:
      return "";
    case mgp::Type::Any:
      return "";
    case mgp::Type::Bool:
      return (value.ValueBool() ? "true" : "false");
    case mgp::Type::Int:
      return std::to_string(value.ValueInt());
    case mgp::Type::Double:
      return std::to_string(value.ValueDouble());
    case mgp::Type::String:
      return std::string(value.ValueString());
    case mgp::Type::List:
      return "";
    case mgp::Type::Map:
      return "";
    case mgp::Type::Node:
      return "Node[" + std::to_string(value.ValueNode().Id().AsInt()) + "]";
    case mgp::Type::Relationship:
      return "Relationship[" + std::to_string(value.ValueRelationship().Id().AsInt()) + "]";
    case mgp::Type::Path:
      return "";
    case mgp::Type::Date: {
      const auto date{value.ValueDate()};
      return std::to_string(date.Year()) + "-" + std::to_string(date.Month()) + "-" + std::to_string(date.Day());
    }
    case mgp::Type::LocalTime: {
      const auto localTime{value.ValueLocalTime()};
      return std::to_string(localTime.Hour()) + ":" + std::to_string(localTime.Minute()) + ":" +
             std::to_string(localTime.Second()) + "," + std::to_string(localTime.Millisecond()) +
             std::to_string(localTime.Microsecond());
    }
    case mgp::Type::LocalDateTime: {
      const auto localDateTime = value.ValueLocalDateTime();
      return std::to_string(localDateTime.Year()) + "-" + std::to_string(localDateTime.Month()) + "-" +
             std::to_string(localDateTime.Day()) + "T" + std::to_string(localDateTime.Hour()) + ":" +
             std::to_string(localDateTime.Minute()) + ":" + std::to_string(localDateTime.Second()) + "," +
             std::to_string(localDateTime.Millisecond()) + std::to_string(localDateTime.Microsecond());
    }
    case mgp::Type::Duration:
      return std::to_string(value.ValueDuration().Microseconds()) + "ms";
    default:
      throw mgp::ValueException("Unknown value type");
  }
}
}  // namespace

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

      auto key{value_to_string(node.GetProperty(std::string(property)))};
      if (key.empty()) {
        continue;
      }

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
