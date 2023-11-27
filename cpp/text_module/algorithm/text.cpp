#include "text.hpp"

#include <fmt/args.h>
#include <fmt/format.h>

void Text::Join(mgp_list *args, mgp_graph * /*memgraph_graph*/, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  ;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto list{arguments[0].ValueList()};
    const auto delimiter{arguments[1].ValueString()};

    std::string result;
    if (list.Empty()) {
      auto record = record_factory.NewRecord();
      record.Insert(std::string(kResultJoin).c_str(), result);
      return;
    }

    auto iterator = list.begin();
    result += (*iterator).ValueString();

    for (++iterator; iterator != list.end(); ++iterator) {
      result += delimiter;
      result += (*iterator).ValueString();
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultJoin).c_str(), result);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Text::Format(mgp_list *args, mgp_graph * /*memgraph_graph*/, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  ;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    std::string text{arguments[0].ValueString()};
    const auto params{arguments[1].ValueList()};

    fmt::dynamic_format_arg_store<fmt::format_context> storage;
    std::for_each(params.begin(), params.end(),
                  [&storage](const mgp::Value &value) { storage.push_back(value.ToString()); });

    std::string result = fmt::vformat(text, storage);

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultFormat).c_str(), result);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
