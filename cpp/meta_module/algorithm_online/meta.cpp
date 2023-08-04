#include "meta.hpp"
#include <string_view>
#include "mgp.hpp"

namespace Meta {

namespace {

class Metadata {
 public:
  int64_t node_cnt;
  int64_t relationship_cnt;
  std::unordered_map<std::string_view, int64_t> labels;
  std::unordered_map<std::string_view, int64_t> property_key_cnt;
  std::unordered_map<std::string_view, int64_t> relationship_types;
  std::unordered_map<std::string_view, int64_t> relationship_types_cnt;

  int64_t get_label_count() const { return static_cast<int64_t>(labels.size()); }
  int64_t get_relationship_type_count() const { return static_cast<int64_t>(relationship_types.size()); }
  int64_t get_property_key_count() const { return static_cast<int64_t>(property_key_cnt.size()); }

  void update_labels(const mgp::Node &node, int add);
  void update_property_key_cnt(const mgp::Node &node, int add);
  void update_property_key_cnt(const mgp::Relationship &relationship, int add);
  void update_relationship_types(const mgp::Relationship &relationship, int add);
  void update_relationship_types_cnt(const mgp::Relationship &relationship, int add);
};

// Global variable
// This is unsafe in the multithreaded environment, the workaround would be building a thread-safe dynamic storage
// implementation
Metadata metadata;

void insert(std::unordered_map<std::string_view, int64_t> &map, std::string_view key, int64_t add) {
  auto iterator = map.find(key);
  if (iterator != map.end()) {
    (*iterator).second += add;
    if ((*iterator).second == 0) {
      map.erase(iterator);
    }
  } else {
    map[key] = 1;
  }
}

void Metadata::update_labels(const mgp::Node &node, int add) {
  for (const auto &label : node.Labels()) {
    insert(labels, label, add);
  }
}

void Metadata::update_property_key_cnt(const mgp::Node &node, int add) {
  for (const auto &[property, _] : node.Properties()) {
    insert(property_key_cnt, property, add);
  }
}

void Metadata::update_property_key_cnt(const mgp::Relationship &relationship, int add) {
  for (const auto &[property, _] : relationship.Properties()) {
    insert(property_key_cnt, property, add);
  }
}

void Metadata::update_relationship_types(const mgp::Relationship &relationship, int add) {
  const auto type = relationship.Type();

  for (const auto &label : relationship.From().Labels()) {
    auto key = "(:" + std::string(label) + ")-[:" + std::string(type) + "]->()";
    insert(relationship_types, key, add);
  }
  for (const auto &label : relationship.To().Labels()) {
    auto key = "()-[:" + std::string(type) + "]->(:" + std::string(label) + ")";
    insert(relationship_types, key, add);
  }
}

void Metadata::update_relationship_types_cnt(const mgp::Relationship &relationship, int add) {
  insert(relationship_types_cnt, relationship.Type(), add);
}

}  // namespace

void Update(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto created_objects{arguments[0].ValueList()};
    const auto deleted_objects{arguments[1].ValueList()};
    const auto removed_vertex_properties{arguments[2].ValueList()};
    const auto removed_edge_properties{arguments[3].ValueList()};
    const auto set_vertex_labels{arguments[4].ValueList()};
    const auto removed_vertex_labels{arguments[5].ValueList()};

    for (const auto &object : created_objects) {
      const auto event{object.ValueMap()};

      const auto event_type = event["event_type"].ValueString();
      if (event_type == "created_vertex") {
        Meta::metadata.node_cnt++;
        const auto vertex = event["vertex"].ValueNode();
        Meta::metadata.update_labels(vertex, 1);
        Meta::metadata.update_property_key_cnt(vertex, 1);
      } else if (event_type == "created_edge") {
        Meta::metadata.relationship_cnt++;
        const auto edge = event["edge"].ValueRelationship();
        Meta::metadata.update_relationship_types(edge, 1);
        Meta::metadata.update_relationship_types_cnt(edge, 1);
        Meta::metadata.update_property_key_cnt(edge, 1);
      } else {
        throw mgp::ValueException("Unexpected event type");
      }
    }

    for (const auto &object : deleted_objects) {
      const auto event{object.ValueMap()};

      const auto event_type = event["event_type"].ValueString();
      if (event_type == "deleted_vertex") {
        Meta::metadata.node_cnt--;
        const auto vertex = event["vertex"].ValueNode();
        Meta::metadata.update_labels(vertex, -1);
        Meta::metadata.update_property_key_cnt(vertex, -1);
      } else if (event_type == "deleted_edge") {
        Meta::metadata.relationship_cnt--;
        const auto edge = event["edge"].ValueRelationship();
        Meta::metadata.update_relationship_types(edge, -1);
        Meta::metadata.update_relationship_types_cnt(edge, -1);
        Meta::metadata.update_property_key_cnt(edge, -1);
      } else {
        throw mgp::ValueException("Unexpected event type");
      }
    }

    for (const auto &object : removed_vertex_properties) {
      const auto event{object.ValueMap()};
      insert(Meta::metadata.property_key_cnt, event["key"].ValueString(), -1);
    }

    for (const auto &object : removed_edge_properties) {
      const auto event{object.ValueMap()};
      insert(Meta::metadata.property_key_cnt, event["key"].ValueString(), -1);
    }

    for (const auto &object : set_vertex_labels) {
      const auto event{object.ValueMap()};
      insert(Meta::metadata.labels, event["label"].ValueString(), event["vertices"].ValueList().Size());
    }

    for (const auto &object : removed_vertex_labels) {
      const auto event{object.ValueMap()};
      insert(Meta::metadata.labels, event["label"].ValueString(), event["vertices"].ValueList().Size());
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Stats(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    mgp::Map stats{};
    stats.Insert("labelCount", mgp::Value(metadata.get_label_count()));
    stats.Insert("relTypeCount", mgp::Value(metadata.get_relationship_type_count()));
    stats.Insert("propertyKeyCount", mgp::Value(metadata.get_property_key_count()));
    stats.Insert("nodeCount", mgp::Value(metadata.node_cnt));
    stats.Insert("relCount", mgp::Value(metadata.relationship_cnt));

    auto create_map = [](const auto &map) {
      mgp::Map result;
      for (const auto &[key, value] : map) {
        result.Insert(key, mgp::Value(value));
      }
      return result;
    };

    stats.Insert("labels", mgp::Value(create_map(metadata.labels)));
    stats.Insert("relTypes", mgp::Value(create_map(metadata.relationship_types)));
    stats.Insert("relTypesCount", mgp::Value(create_map(metadata.relationship_types_cnt)));

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kReturnStats).c_str(), stats);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

}  // namespace Meta
