#include "meta.hpp"

#include <atomic>
#include <mutex>

#include "mgp.hpp"

namespace Meta {

namespace {

struct Metadata {
  Metadata() : node_cnt{0}, relationship_cnt{0} {}

  void Reset();
  void UpdateLabels(const mgp::Node &node, int add, bool online);
  void UpdatePropertyKeyCnt(const auto &node_or_relationship, int add, bool online);
  void UpdateRelationshipTypes(const mgp::Relationship &relationship, int add, bool online);
  void UpdateRelationshipTypesCnt(const mgp::Relationship &relationship, int add, bool online);

  std::atomic<int64_t> node_cnt;
  std::atomic<int64_t> relationship_cnt;

  std::unordered_map<std::string, int64_t> labels;
  std::unordered_map<std::string, int64_t> property_key_cnt;
  std::unordered_map<std::string, int64_t> relationship_types;
  std::unordered_map<std::string, int64_t> relationship_types_cnt;
};

struct MetadataMutex {
  std::mutex labels_mutex;
  std::mutex property_key_cnt_mutex;
  std::mutex relationship_types_mutex;
  std::mutex relationship_types_cnt_mutex;
};

// Global variable
Metadata metadata;
MetadataMutex metadata_mutex;

void LockAndExecute(std::mutex &mutex, bool online, auto Function) {
  if (online) {
    const std::lock_guard<std::mutex> lock(mutex);
    Function();
    return;
  }
  Function();
}

auto LockAndExecuteReturn(std::mutex &mutex, bool online, auto Function) {
  if (online) {
    const std::lock_guard<std::mutex> lock(mutex);
    return Function();
  }
  return Function();
}

void Insert(std::unordered_map<std::string, int64_t> &map, std::string &key, int64_t add) {
  auto iterator = map.find(key);
  if (iterator != map.end()) {
    (*iterator).second += add;
    if ((*iterator).second == 0) {
      map.erase(iterator);
    }
  } else {
    map[key] = add;
  }
}

void Metadata::Reset() {
  node_cnt = 0;
  relationship_cnt = 0;

  auto ClearMap = [](std::mutex &mutex, auto &map) { LockAndExecute(mutex, true, [&]() { map.clear(); }); };

  ClearMap(metadata_mutex.labels_mutex, labels);
  ClearMap(metadata_mutex.property_key_cnt_mutex, property_key_cnt);
  ClearMap(metadata_mutex.relationship_types_mutex, relationship_types);
  ClearMap(metadata_mutex.relationship_types_cnt_mutex, relationship_types_cnt);
}

void Metadata::UpdateLabels(const mgp::Node &node, int add, bool online) {
  auto IterateAndInsert = [&]() {
    for (const auto &label : node.Labels()) {
      auto key = std::string(label);
      Insert(labels, key, add);
    }
  };

  LockAndExecute(metadata_mutex.labels_mutex, online, IterateAndInsert);
}

void Metadata::UpdatePropertyKeyCnt(const auto &node_or_relationship, int add, bool online) {
  auto IterateAndInsert = [&]() {
    for (const auto &[property, _] : node_or_relationship.Properties()) {
      auto key = std::string(property);
      Insert(property_key_cnt, key, add);
    }
  };

  LockAndExecute(metadata_mutex.property_key_cnt_mutex, online, IterateAndInsert);
}

void Metadata::UpdateRelationshipTypes(const mgp::Relationship &relationship, int add, bool online) {
  auto IterateAndInsert = [&]() {
    const auto type = relationship.Type();

    for (const auto &label : relationship.From().Labels()) {
      auto key = "(:" + std::string(label) + ")-[:" + std::string(type) + "]->()";
      Insert(relationship_types, key, add);
    }
    for (const auto &label : relationship.To().Labels()) {
      auto key = "()-[:" + std::string(type) + "]->(:" + std::string(label) + ")";
      Insert(relationship_types, key, add);
    }

    auto key = "()-[:" + std::string(type) + "]->()";
    Insert(relationship_types, key, add);
  };

  LockAndExecute(metadata_mutex.relationship_types_mutex, online, IterateAndInsert);
}

void Metadata::UpdateRelationshipTypesCnt(const mgp::Relationship &relationship, int add, bool online) {
  auto IterateAndInsert = [&]() {
    auto key = std::string(relationship.Type());
    Insert(relationship_types_cnt, key, add);
  };
  LockAndExecute(metadata_mutex.relationship_types_cnt_mutex, online, IterateAndInsert);
}

void UpdateAllMetadata(mgp_graph *memgraph_graph, Metadata &data, bool online) {
  const mgp::Graph graph{memgraph_graph};

  for (const auto node : graph.Nodes()) {
    data.node_cnt++;
    data.UpdateLabels(node, 1, online);
    data.UpdatePropertyKeyCnt(node, 1, online);
  }

  for (const auto relationship : graph.Relationships()) {
    data.relationship_cnt++;
    data.UpdateRelationshipTypes(relationship, 1, online);
    data.UpdateRelationshipTypesCnt(relationship, 1, online);
    data.UpdatePropertyKeyCnt(relationship, 1, online);
  }
}

void OutputFromMetadata(Metadata &data, const mgp::RecordFactory &record_factory, bool online) {
  auto record = record_factory.NewRecord();
  mgp::Map stats{};

  int64_t label_count = LockAndExecuteReturn(metadata_mutex.labels_mutex, online,
                                             [&]() { return static_cast<int64_t>(data.labels.size()); });
  record.Insert(std::string(kReturnStats1).c_str(), label_count);
  stats.Insert("labelCount", mgp::Value(label_count));

  int64_t relationship_type_count = LockAndExecuteReturn(metadata_mutex.relationship_types_cnt_mutex, online, [&]() {
    return static_cast<int64_t>(data.relationship_types_cnt.size());
  });
  record.Insert(std::string(kReturnStats2).c_str(), relationship_type_count);
  stats.Insert("relationshipTypeCount", mgp::Value(relationship_type_count));

  int64_t property_key_count = LockAndExecuteReturn(metadata_mutex.property_key_cnt_mutex, online, [&]() {
    return static_cast<int64_t>(data.property_key_cnt.size());
  });
  record.Insert(std::string(kReturnStats3).c_str(), property_key_count);
  stats.Insert("propertyKeyCount", mgp::Value(property_key_count));

  record.Insert(std::string(kReturnStats4).c_str(), data.node_cnt);
  stats.Insert("nodeCount", mgp::Value(data.node_cnt));

  record.Insert(std::string(kReturnStats5).c_str(), data.relationship_cnt);
  stats.Insert("relationshipCount", mgp::Value(data.relationship_cnt));

  auto CreateMap = [](const auto &map) {
    mgp::Map result;
    for (const auto &[key, value] : map) {
      result.Insert(key, mgp::Value(value));
    }
    return result;
  };

  auto labels_map = LockAndExecuteReturn(metadata_mutex.labels_mutex, online, [&]() { return CreateMap(data.labels); });
  record.Insert(std::string(kReturnStats6).c_str(), labels_map);
  stats.Insert("labels", mgp::Value(std::move(labels_map)));

  auto relationship_types_map = LockAndExecuteReturn(metadata_mutex.relationship_types_mutex, online,
                                                     [&]() { return CreateMap(data.relationship_types); });
  record.Insert(std::string(kReturnStats7).c_str(), relationship_types_map);
  stats.Insert("relationshipTypes", mgp::Value(std::move(relationship_types_map)));

  auto relationship_types_count_map = LockAndExecuteReturn(metadata_mutex.relationship_types_cnt_mutex, online,
                                                           [&]() { return CreateMap(data.relationship_types_cnt); });
  record.Insert(std::string(kReturnStats8).c_str(), relationship_types_count_map);
  stats.Insert("relationshipTypesCount", mgp::Value(std::move(relationship_types_count_map)));

  record.Insert(std::string(kReturnStats9).c_str(), stats);
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

    auto UpdateObject = [](const auto &objects, std::string_view vertex_event, std::string_view edge_event, int add) {
      for (const auto &object : objects) {
        const auto event{object.ValueMap()};

        const auto event_type = event["event_type"].ValueString();
        if (event_type == vertex_event) {
          Meta::metadata.node_cnt += add;
          const auto vertex = event["vertex"].ValueNode();
          Meta::metadata.UpdateLabels(vertex, add, true);
          Meta::metadata.UpdatePropertyKeyCnt(vertex, add, true);
        } else if (event_type == edge_event) {
          Meta::metadata.relationship_cnt += add;
          const auto edge = event["edge"].ValueRelationship();
          Meta::metadata.UpdateRelationshipTypes(edge, add, true);
          Meta::metadata.UpdateRelationshipTypesCnt(edge, add, true);
          Meta::metadata.UpdatePropertyKeyCnt(edge, add, true);
        } else {
          throw mgp::ValueException("Unexpected event type");
        }
      }
    };

    UpdateObject(created_objects, "created_vertex", "created_edge", 1);
    UpdateObject(deleted_objects, "deleted_vertex", "deleted_edge", -1);

    for (const auto &object : removed_vertex_properties) {
      const auto event{object.ValueMap()};
      LockAndExecute(metadata_mutex.property_key_cnt_mutex, true, [&]() {
        auto key = std::string(event["key"].ValueString());
        Insert(Meta::metadata.property_key_cnt, key, -1);
      });
    }

    for (const auto &object : removed_edge_properties) {
      const auto event{object.ValueMap()};
      LockAndExecute(metadata_mutex.property_key_cnt_mutex, true, [&]() {
        auto key = std::string(event["key"].ValueString());
        Insert(Meta::metadata.property_key_cnt, key, -1);
      });
    }

    for (const auto &object : set_vertex_labels) {
      const auto event{object.ValueMap()};
      LockAndExecute(metadata_mutex.property_key_cnt_mutex, true, [&]() {
        auto key = std::string(event["label"].ValueString());
        Insert(Meta::metadata.labels, key, static_cast<int64_t>(event["vertices"].ValueList().Size()));
      });
    }

    for (const auto &object : removed_vertex_labels) {
      const auto event{object.ValueMap()};
      LockAndExecute(metadata_mutex.property_key_cnt_mutex, true, [&]() {
        auto key = std::string(event["label"].ValueString());
        Insert(Meta::metadata.labels, key, static_cast<int64_t>(-event["vertices"].ValueList().Size()));
      });
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void StatsOnline(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    bool update_stats = arguments[0].ValueBool();

    if (update_stats) {
      Meta::metadata.Reset();
      UpdateAllMetadata(memgraph_graph, metadata, true);
      OutputFromMetadata(metadata, record_factory, true);
    } else {
      OutputFromMetadata(metadata, record_factory, true);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void StatsOffline(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    Metadata data;
    UpdateAllMetadata(memgraph_graph, data, false);
    OutputFromMetadata(data, record_factory, false);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Reset(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto record_factory = mgp::RecordFactory(result);

  try {
    metadata.Reset();

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

}  // namespace Meta
