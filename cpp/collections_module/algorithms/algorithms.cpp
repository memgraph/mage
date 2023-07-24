#include <list>
#include <unordered_set>

#include "algorithms.hpp"

void Collections::Union(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto &list1 = arguments[0].ValueList();
    const auto &list2 = arguments[1].ValueList();

    std::unordered_map<int64_t, std::vector<mgp::Value>> unionMap;

    for (const auto value : list1) {
      if (auto search = unionMap.find(std::hash<mgp::Value>{}(value)); search != unionMap.end()) {
        if (std::find(search->second.begin(), search->second.end(), value) != search->second.end()) {
          continue;
        }
        search->second.push_back(value);
      }
      unionMap.insert({std::hash<mgp::Value>{}(value), std::vector<mgp::Value>{value}});
    }
    for (const auto value : list2) {
      if (auto search = unionMap.find(std::hash<mgp::Value>{}(value)); search != unionMap.end()) {
        if (std::find(search->second.begin(), search->second.end(), value) != search->second.end()) {
          continue;
        }
        search->second.push_back(value);
      }
      unionMap.insert({std::hash<mgp::Value>{}(value), std::vector<mgp::Value>{value}});
    }

    mgp::List unionList;

    for (auto pair : unionMap) {
      for (auto value : pair.second) {
        unionList.AppendExtend(std::move(value));
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultUnion).c_str(), unionList);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
