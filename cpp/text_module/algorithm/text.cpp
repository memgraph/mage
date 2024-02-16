#include "text.hpp"

#include <regex>
#include <vector>

#include <fmt/args.h>
#include <fmt/format.h>

void Text::Join(mgp_list *args, mgp_graph * /*memgraph_graph*/, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
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
  }
}

void Text::Format(mgp_list *args, mgp_graph * /*memgraph_graph*/, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
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
  }
}

void Text::RegexGroups(mgp_list *args, mgp_graph * /*memgraph_graph*/, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const std::string input{arguments[0].ValueString()};
    const std::string regex_str{arguments[1].ValueString()};

    std::regex regex{regex_str};

    size_t mark_count = regex.mark_count();
    std::vector<int> submatches;
    for (size_t i = 0; i <= mark_count; ++i) {
      submatches.emplace_back(i);
    }

    mgp::List all_results;
    mgp::List local_results;
    uint32_t cnt = 1;
    auto words_begin = std::sregex_token_iterator(input.begin(), input.end(), regex, submatches);
    auto words_end = std::sregex_token_iterator();

    for (auto it = words_begin; it != words_end; it++, cnt++) {
      if (auto match = it->str(); !match.empty()) {
        local_results.AppendExtend(mgp::Value(match));
      }
      if (cnt % (mark_count + 1) == 0) {
        all_results.AppendExtend(mgp::Value(local_results));
        local_results = mgp::List();
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRegexGroups).c_str(), all_results);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
  }
}
