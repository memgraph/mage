#include "schema.hpp"

/*we have << operator for type in Cpp API, but in it we return somewhat different strings than I would like in this
module, so I implemented a small function here*/
std::string Schema::TypeOf(const mgp::Type &type) {
  switch (type) {
    case mgp::Type::Null:
      return "Null";
    case mgp::Type::Bool:
      return "Bool";
    case mgp::Type::Int:
      return "Int";
    case mgp::Type::Double:
      return "Double";
    case mgp::Type::String:
      return "String";
    case mgp::Type::List:
      return "List[Any]";
    case mgp::Type::Map:
      return "Map[Any]";
    case mgp::Type::Node:
      return "Vertex";
    case mgp::Type::Relationship:
      return "Edge";
    case mgp::Type::Path:
      return "Path";
    case mgp::Type::Date:
      return "Date";
    case mgp::Type::LocalTime:
      return "LocalTime";
    case mgp::Type::LocalDateTime:
      return "LocalDateTime";
    case mgp::Type::Duration:
      return "Duration";
    default:
      throw mgp::ValueException("Unsupported type");
  }
}

void Schema::ProcessPropertiesNode(mgp::Record &record, const mgp::List &labels, const std::string &propertyName,
                                   const std::string &propertyType, const bool &mandatory) {
  record.Insert(std::string(kReturnLabels).c_str(), labels);
  record.Insert(std::string(kReturnPropertyName).c_str(), propertyName);
  record.Insert(std::string(kReturnPropertyType).c_str(), propertyType);
  record.Insert(std::string(kReturnMandatory).c_str(), mandatory);
}

void Schema::ProcessPropertiesRel(mgp::Record &record, const std::string_view &type, const std::string &propertyName,
                                  const std::string &propertyType, const bool &mandatory) {
  record.Insert(std::string(kReturnRelType).c_str(), type);
  record.Insert(std::string(kReturnPropertyName).c_str(), propertyName);
  record.Insert(std::string(kReturnPropertyType).c_str(), propertyType);
  record.Insert(std::string(kReturnMandatory).c_str(), mandatory);
}

void Schema::NodeTypeProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const mgp::Graph graph = mgp::Graph(memgraph_graph);

    for (auto node : graph.Nodes()) {
      mgp::List labels = mgp::List();
      for (auto label : node.Labels()) {
        labels.AppendExtend(mgp::Value(label));
      }

      if (node.Properties().size() == 0) {
        auto record = record_factory.NewRecord();
        ProcessPropertiesNode(record, labels, "", "", false);
        continue;
      }

      for (auto &[key, prop] : node.Properties()) {
        auto record = record_factory.NewRecord();
        ProcessPropertiesNode(record, labels, key, TypeOf(prop.Type()), true);
      }
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Schema::RelTypeProperties(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const mgp::Graph graph = mgp::Graph(memgraph_graph);

    for (auto rel : graph.Relationships()) {
      if (rel.Properties().size() == 0) {
        auto record = record_factory.NewRecord();
        ProcessPropertiesRel(record, rel.Type(), "", "", false);
        continue;
      }

      for (auto &[key, prop] : rel.Properties()) {
        auto record = record_factory.NewRecord();
        ProcessPropertiesRel(record, rel.Type(), key, TypeOf(prop.Type()), true);
      }
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
