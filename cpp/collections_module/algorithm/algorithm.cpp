#include "algorithm.hpp"



void Collections::Min(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const mgp::List &list = arguments[0].ValueList();
    if (list.Empty()) {
      throw mgp::ValueException("Empty input list");
    }
    const mgp::Type &type = list[0].Type();
    auto record = record_factory.NewRecord();

    if (type == mgp::Type::Map || type == mgp::Type::Path || type == mgp::Type::List) {
      std::ostringstream oss;
      oss << type;
      std::string s = oss.str();
      throw mgp::ValueException("Unsuppported type for this operation, receieved type: " + s);
    }

    bool isListNumeric = list[0].IsNumeric();
    mgp::Value min{std::move(list[0])};
    for (size_t i = 0; i < list.Size(); i++) {
      if (list[i].Type() != type && !(isListNumeric && list[i].IsNumeric())) {
        throw mgp::ValueException("All elements must be of the same type!");
      }

      if (list[i] < min) {
        min = std::move(list[i]);
      }
    }

    record.Insert(std::string(kReturnValueMin).c_str(), min);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

