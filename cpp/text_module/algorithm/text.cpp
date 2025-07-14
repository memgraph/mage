#include "text.hpp"

#include <algorithm>
#include <regex>
#include <vector>

#include <fmt/args.h>
#include <fmt/format.h>
#include <unicode/normalizer2.h>
#include <unicode/unistr.h>

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

void Text::Replace(mgp_list *args, mgp_func_context * /*ctx*/, mgp_func_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  mgp::Result result_obj(result);

  try {
    auto text = std::string(arguments[0].ValueString());
    const auto regex = std::string(arguments[1].ValueString());
    const auto replacement = std::string(arguments[2].ValueString());

    if (regex.size() == 0) {
      result_obj.SetValue(text);
      return;
    }

    size_t pos = 0;
    while ((pos = text.find(regex, pos)) != std::string::npos) {
      text.replace(pos, regex.length(), replacement);
      pos += replacement.length();
    }

    result_obj.SetValue(std::move(text));
  } catch (const std::exception &e) {
    result_obj.SetErrorMessage(e.what());
    return;
  }
}

void Text::RegReplace(mgp_list *args, mgp_func_context * /*ctx*/, mgp_func_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  mgp::Result result_obj(result);

  try {
    const auto text = std::string(arguments[0].ValueString());
    const auto regex = std::string(arguments[1].ValueString());
    const auto replacement = std::string(arguments[2].ValueString());

    if (regex.size() == 0) {
      result_obj.SetValue(text);
      return;
    }

    std::regex pattern(regex);
    std::string result_str = std::regex_replace(text, pattern, replacement);

    result_obj.SetValue(std::move(result_str));
  } catch (const std::exception &e) {
    result_obj.SetErrorMessage(e.what());
    return;
  }
}

void Text::Distance(mgp_list *args, mgp_func_context * /*ctx*/, mgp_func_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  mgp::Result result_obj(result);

  const auto normalize = [](const std::string &input) -> std::string {
    UErrorCode error = U_ZERO_ERROR;

    // Get the NFD normalizer (decomposes é → e + ́)
    const icu::Normalizer2 *normalizer = icu::Normalizer2::getNFDInstance(error);
    if (U_FAILURE(error)) return input;

    // Convert UTF-8 input to UnicodeString
    icu::UnicodeString unicode_input = icu::UnicodeString::fromUTF8(input);
    icu::UnicodeString normalized;
    normalizer->normalize(unicode_input, normalized, error);
    if (U_FAILURE(error)) return input;

    // Create UnicodeSet for non-spacing marks (diacritics)
    icu::UnicodeSet diacritics;
    diacritics.applyPattern(icu::UnicodeString::fromUTF8("[:Nonspacing Mark:]"), error);
    if (U_FAILURE(error)) return input;

    // Build result string by skipping characters in diacritics set
    icu::UnicodeString cleaned;
    for (int32_t i = 0; i < normalized.length();) {
      UChar32 c = normalized.char32At(i);
      if (!diacritics.contains(c)) {
        cleaned.append(c);
      }
      i += U16_LENGTH(c);
    }

    // Convert back to UTF-8
    std::string result;
    cleaned.toUTF8String(result);
    return result;
  };

  try {
    const auto text1 = normalize(std::string(arguments[0].ValueString()));
    const auto text2 = normalize(std::string(arguments[1].ValueString()));

    const size_t m = text1.length();
    const size_t n = text2.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));

    for (size_t i = 0; i <= m; i++) {
      dp[i][0] = i;
    }
    for (size_t j = 0; j <= n; j++) {
      dp[0][j] = j;
    }

    for (size_t i = 1; i <= m; i++) {
      for (size_t j = 1; j <= n; j++) {
        if (text1[i - 1] == text2[j - 1]) {
          dp[i][j] = dp[i - 1][j - 1];
        } else {
          dp[i][j] = 1 + std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
        }
      }
    }

    result_obj.SetValue(static_cast<int64_t>(dp[m][n]));
  } catch (const std::exception &e) {
    result_obj.SetErrorMessage(e.what());
    return;
  }
}
