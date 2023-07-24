#include "algorithm.hpp"

void Collections::Avg(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    double average{0};
    const auto &list = arguments[0].ValueList();

    for (const auto list_item : list) {
      if (!list_item.IsNumeric()) {
        // TODO: got element of type x expected number
        throw std::invalid_argument("One of the list elements is not a number");
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
