#include <fmt/core.h>
#include <chrono>
#include <mg_exceptions.hpp>
#include <mgp.hpp>
#include <string>
#include <string_view>
#include <thread>

constexpr char *kProcedureExecuteQuery = "query";
constexpr char *kArgumentInputQuery = "input_query";
constexpr char *kArgumentParameters = "parameters";
constexpr char *kArgumentConfig = "config";
constexpr char *kReturnSuccess = "success";
constexpr char *kReturnNumberOfRetries = "number_of_retries";

constexpr char *kConfigKeyMaxRetries = "max_retries";
constexpr char *kConfigKeyRetryType = "retry_type";
constexpr char *kCOnfigKeyInitialBackoff = "initial_backoff";

constexpr char *kConfigKeyExponentialBackoff = "EXPONENTIAL";
constexpr char *kCOnfigKeyLinearBackoff = "LINEAR";

void replaceString(std::string &subject, const std::string &search, const std::string &replace) {
  size_t pos = 0;
  while ((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
}

void ExecuteQuery(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);

  const auto record_factory = mgp::RecordFactory(result);
  auto record = record_factory.NewRecord();

  auto input_query = std::string(arguments[0].ValueString());
  const auto parameters = arguments[1].ValueMap();
  const auto config = arguments[2].ValueMap();

  auto max_retries = 0;
  auto backoff = "EXPONENTIAL";
  auto initial_backoff = 10;
  if (config.KeyExists(kConfigKeyMaxRetries)) {
    if (!config.At(kConfigKeyMaxRetries).IsInt()) {
      record_factory.SetErrorMessage("max_retries parameter needs to be an integer!");
      record.Insert(kReturnSuccess, false);
      return;
    }
    max_retries = config.At(kConfigKeyMaxRetries).ValueInt();
  }

  if (max_retries != 0) {
    if (config.KeyExists(kConfigKeyRetryType)) {
      if (!config.At(kConfigKeyRetryType).IsString()) {
        record_factory.SetErrorMessage("retry_type parameter needs to be an string!");
        record.Insert(kReturnSuccess, false);
        return;
      }
      auto retry_type = std::string(config.At(kConfigKeyRetryType).ValueString());
      if (retry_type != kConfigKeyExponentialBackoff && retry_type != kCOnfigKeyLinearBackoff) {
        record_factory.SetErrorMessage("retry_type parameter needs to either EXPONENTIAL or LINEAR!");
        record.Insert(kReturnSuccess, false);
        return;
      }
    }
    if (config.KeyExists(kCOnfigKeyInitialBackoff)) {
      if (!config.At(kCOnfigKeyInitialBackoff).IsInt()) {
        record_factory.SetErrorMessage("initial_backoff parameter needs to be an integer!");
        record.Insert(kReturnSuccess, false);
        return;
      }
      initial_backoff = config.At(kCOnfigKeyInitialBackoff).ValueInt();
    }
  }

  for (const auto &[key, value] : parameters) {
    if (value.IsString()) {
      replaceString(input_query, fmt::format("${}", key), fmt::format("{}", value.ValueString()));
    } else if (value.IsBool()) {
      replaceString(input_query, fmt::format("${}", key), fmt::format("{}", value.ValueBool()));
    } else if (value.IsInt()) {
      replaceString(input_query, fmt::format("${}", key), fmt::format("{}", value.ValueInt()));
    } else if (value.IsDouble()) {
      replaceString(input_query, fmt::format("${}", key), fmt::format("{}", value.ValueDouble()));
    } else if (value.IsNull()) {
      replaceString(input_query, fmt::format("${}", key), "null");
    }
  }

  int64_t number_of_retries = 0;
  do {
    try {
      auto input_query_execution = mgp::QueryExecution(memgraph_graph);
      auto execution_result = input_query_execution.ExecuteQuery(input_query);

      while (execution_result.PullOne()) {
      }

      record.Insert(kReturnSuccess, true);
      record.Insert(kReturnNumberOfRetries, number_of_retries);
      return;
    } catch (const mg_exception::RetryBasicException &e) {
      number_of_retries++;

      if (number_of_retries <= max_retries) {
        std::this_thread::sleep_for(std::chrono::milliseconds(initial_backoff));
        if (backoff == kConfigKeyExponentialBackoff) {
          initial_backoff *= 2;
        }
      }
    } catch (const std::exception &e) {
      record_factory.SetErrorMessage(e.what());
      record.Insert(kReturnSuccess, false);
      record.Insert(kReturnNumberOfRetries, number_of_retries);
      return;
    }
  } while (number_of_retries <= max_retries);

  record_factory.SetErrorMessage(
      fmt::format("Did not successfully execute query! Number of retries: {}.", max_retries));
  record.Insert(kReturnSuccess, false);
  return;
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};
    mgp::AddProcedure(
        ExecuteQuery, kProcedureExecuteQuery, mgp::ProcedureType::Read,
        {mgp::Parameter(kArgumentInputQuery, mgp::Type::String), mgp::Parameter(kArgumentParameters, mgp::Type::Map),
         mgp::Parameter(kArgumentConfig, mgp::Type::Map)},
        {mgp::Return(kReturnSuccess, mgp::Type::Bool), mgp::Return(kReturnNumberOfRetries, mgp::Type::Int)}, module,
        memory);
  } catch (const std::exception &e) {
    return 1;
  }
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
