#include "merge.hpp"

bool Merge::LabelsContained(const std::unordered_set<std::string_view> &labels, const mgp::Node &node) {
  bool contained = false;
  auto size = labels.size();  // this works if labels are unique, which they are
  size_t counter = 0;

  for (const auto label : node.Labels()) {
    if (labels.find(label) != labels.end()) {
      counter++;
    }
  }

  if (counter == size) {
    contained = true;
  }

  return contained;
}

bool Merge::IdentProp(const mgp::Map &identProp, const mgp::Node &node) {
  bool match = true;
  for (const auto &[key, value] : identProp) {
    if (value != node.GetProperty(std::string(key))) {
      match = false;
    }
  }
  return match;
}

void Merge::Node(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto graph = mgp::Graph(memgraph_graph);
    auto labels = arguments[0].ValueList();
    auto identProp = arguments[1].ValueMap();
    auto createProp = arguments[2].ValueMap();
    auto matchProp = arguments[3].ValueMap();
    std::unordered_set<std::string_view> label_set;
    std::unordered_map<std::string_view, mgp::Value> identProp_map;
    std::unordered_map<std::string_view, mgp::Value> createProp_map;
    std::unordered_map<std::string_view, mgp::Value> matchProp_map;
    /*conversion of mgp::Maps to unordered_map for easier use of SetProperties(it expects an unordered map as
     * argument)*/
    for (const auto &[key, value] : identProp) {
      identProp_map.emplace(key, value);
    }
    for (const auto &[key, value] : createProp) {
      createProp_map.emplace(key, value);
    }
    for (const auto &[key, value] : matchProp) {
      matchProp_map.emplace(key, value);
    }
    /*creating a set of labels for O(1) check of labels*/
    for (const auto elem : labels) {
      const auto label = elem.ValueString();
      if (label == "") {
        throw mgp::ValueException("List of labels cannot contain empty string!");
      }
      label_set.insert(elem.ValueString());
    }

    bool matched = false;
    for (auto node : graph.Nodes()) {
      /*check if node already exists, if true, merge, if not, create*/
      if (LabelsContained(label_set, node) && IdentProp(identProp, node)) {
        matched = true;
        node.SetProperties(matchProp_map);
        auto record = record_factory.NewRecord();
        record.Insert(std::string(kNodeRes).c_str(), node);
      }
    }
    if (!matched) {
      auto node = graph.CreateNode();
      for (const auto label : label_set) {
        node.AddLabel(label);
      }
      node.SetProperties(identProp_map);  // when merge creates it creates identyfing props also
      node.SetProperties(createProp_map);
      auto record = record_factory.NewRecord();
      record.Insert(std::string(kNodeRes).c_str(), node);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
