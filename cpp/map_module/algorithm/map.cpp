#include "map.hpp"
#include <string>
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
      throw mgp::ValueException("Unknown value type");
  }
  return "";
}
}  // namespace

void Map::FromValues(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto values{arguments[0].ValueList()};
    mgp::Map map{};

    if (values.Size() % 2) {
      throw mgp::ValueException("List needs to have an even number of elements");
    }

    auto iterator = values.begin();
    while (iterator != values.end()) {
      const auto key = value_to_string(*iterator);
      ++iterator;
      if (key != "") {
        map.Update(key, *iterator);
      }
      ++iterator;
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultFromValues).c_str(), map);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
