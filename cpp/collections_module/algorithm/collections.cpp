#include "collections.hpp"

#include <sstream>
#include <unordered_set>
#include "mgp.hpp"

void Collections::SumLongs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    int64_t sum{0};
    const auto &list = arguments[0].ValueList();

    for (const auto list_item : list) {
      if (!list_item.IsNumeric()) {
        std::ostringstream oss;
        oss << list_item.Type();
        throw mgp::ValueException("Unsupported type for this operation, received type: " + oss.str());
      }
      sum += static_cast<int64_t>(list_item.ValueNumeric());
    }
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultSumLongs).c_str(), sum);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Avg(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    double average{0};
    const auto &list = arguments[0].ValueList();

    for (const auto list_item : list) {
      if (!list_item.IsNumeric()) {
        std::ostringstream oss;
        oss << list_item.Type();
        throw mgp::ValueException("Unsupported type for this operation, received type: " + oss.str());
      }
      average += list_item.ValueNumeric();
    }
    average /= list.Size();

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultAvg).c_str(), average);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Collections::ContainsAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto list1{arguments[0].ValueList()};
    std::unordered_set<mgp::Value> set{list1.begin(), list1.end()};
    const auto list2{arguments[1].ValueList()};
    std::unordered_set<mgp::Value> values{list2.begin(), list2.end()};

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultContainsAll).c_str(),
                  std::all_of(values.begin(), values.end(), [&](const auto &x) { return set.contains(x); }));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Intersection(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto list1{arguments[0].ValueList()};
    std::unordered_set<mgp::Value> set1{list1.begin(), list1.end()};
    const auto list2{arguments[1].ValueList()};
    std::unordered_set<mgp::Value> set2{list2.begin(), list2.end()};

    if (set1.size() > set2.size()) {
      std::swap(set1, set2);
    }

    mgp::List intersection{};
    for (const auto &element : set1) {
      if (set2.contains(element)) {
        intersection.AppendExtend(std::move(element));
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultIntersection).c_str(), intersection);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
