#include "algorithm.hpp"
#include <sstream>
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
    record.Insert(std::string(kResultAverage).c_str(), average);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
