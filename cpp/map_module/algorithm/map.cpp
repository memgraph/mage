#include "map.hpp"
#include <string>
#include "mg_utils.hpp"
#include "mgp.hpp"

void Map::FromValues(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto values{arguments[0].ValueList()};
    mgp::Map map{};

    if (values.Size() % 2) {
      throw mgp::ValueException("List needs to have an even number of elements");
    }

    auto iterator = values.begin();
    while (iterator != values.end()) {
      std::ostringstream oss;
      oss << *iterator;
      const auto key = oss.str();

      ++iterator;
      map.Update(key, *iterator);
      ++iterator;
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultFromValues).c_str(), map);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
