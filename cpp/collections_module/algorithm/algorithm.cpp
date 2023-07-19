#include "algorithm.hpp"

void Collections::SumLongs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    int64_t sum = 0;
    const auto &list = arguments[0].ValueList();
    
    for (const auto list_item : list){
      if(!list_item.IsNumeric())
        // TODO: got element of type x expected number
        throw std::invalid_argument("One of the list elements is not a number");
      sum += static_cast<int64_t>(list_item.ValueNumeric());
    }
    auto record = record_factory.NewRecord();
    record.Insert(kResultSumLongs, sum);
    
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}