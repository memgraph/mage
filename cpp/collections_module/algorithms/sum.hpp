#include <list>

#include <mgp.hpp>

void Sum(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    double sum = 0;
    const auto &list = arguments[0].ValueList();

    for (const auto value : list) {
      if (!value.IsNumeric()) {
        // throw std::invalid_argument(std::format("List element {} is not a number.", value));
        throw std::invalid_argument("One of the list elements is not a number.");
      }
      sum += value.ValueNumeric();
    }

    auto record = record_factory.NewRecord();
    record.Insert("sum", sum);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
