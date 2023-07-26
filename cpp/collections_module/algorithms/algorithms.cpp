#include <list>
#include <vector>
#include <algorithm>
#include <unordered_set>

#include "algorithms.hpp"

void Collections::Sum(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    double sum{0};
    const auto list = arguments[0].ValueList();

    for (const auto value : list) {
      if (!value.IsNumeric()) {
        throw std::invalid_argument("One of the list elements is not a number.");
      }
      sum += value.ValueNumeric();
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultSum).c_str(), sum);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Union(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto list1 = arguments[0].ValueList();
    const auto list2 = arguments[1].ValueList();

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

void Collections::Sort(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto list = arguments[0].ValueList();
    std::vector<mgp::Value> sorted;

    for (const auto value : list) {
      sorted.push_back(std::move(value));
    }

    std::sort(sorted.begin(), sorted.end());

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultSort).c_str(), mgp::List(std::move(sorted)));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Collections::ContainsSorted(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    bool contains{false};
    const auto list = arguments[0].ValueList();
    const auto element = arguments[1];

    int left{0};
    int right{list.Size() - 1};
    int check;

    while (left <= right) {
      check = (left + right) / 2;
      if (list[check] == element) {
        contains = true;
        break;
      } else if (element < list[check]) {
        right = check - 1;
      } else {
        left = check + 1;
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultCS).c_str(), contains);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Max(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto list = arguments[0].ValueList();

    if (list.Empty()) {
      throw mgp::ValueException("Empty input list.");
    }

    mgp::Value max = mgp::Value(list[0]);

    for (const auto value : list) {
      if (max < value) {  // this will throw an error in case values can't be compared
        max = value;
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultMax).c_str(), max);
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Split(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;

  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto inputList = arguments[0].ValueList();
    const auto delimiter = arguments[1];

    if (inputList.Empty()) {
      auto record = record_factory.NewRecord();
      record.Insert(std::string(Collections::kResultSplit).c_str(), inputList);
      return;
    }

    mgp::List part = mgp::List();
    for (const auto value : inputList) {
      if (value != delimiter) {
        part.AppendExtend(value);
        continue;
      }
      if (part.Empty()) {
        continue;
      }
      auto record = record_factory.NewRecord();
      record.Insert(std::string(Collections::kResultSplit).c_str(), part);
      part = mgp::List();
    }
    if (part.Empty()) {
      return;
    }
    auto record = record_factory.NewRecord();
    record.Insert(std::string(Collections::kResultSplit).c_str(), std::move(part));
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Pairs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  auto record = record_factory.NewRecord();
  try {
    mgp::List pairsList = mgp::List();

    const auto inputList = arguments[0].ValueList();

    if (inputList.Size() == 0) {
      record.Insert(std::string(kResultPairs).c_str(), pairsList);
      return;
    }
    for (size_t i = 0; i < inputList.Size() - 1; i++) {
      mgp::List helper = mgp::List();
      helper.AppendExtend(inputList[i]);
      helper.AppendExtend(inputList[i + 1]);
      pairsList.AppendExtend(mgp::Value(std::move(helper)));
    }
    mgp::List helper = mgp::List();
    helper.AppendExtend(inputList[inputList.Size() - 1]);
    helper.AppendExtend(mgp::Value());
    pairsList.AppendExtend(mgp::Value(std::move(helper)));

    record.Insert(std::string(kResultPairs).c_str(), pairsList);
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
