#include <fmt/core.h>
#include <mgp.hpp>
#include <string>
#include <string_view>

constexpr char *kProcedureExecuteQuery = "query";
constexpr char *kArgumentInputQuery = "input_query";
constexpr char *kArgumentParameters = "parameters";
constexpr char *kReturnSuccess = "success";

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

  try {
    auto input_query_execution = mgp::QueryExecution(memgraph_graph);
    auto execution_result = input_query_execution.ExecuteQuery(input_query);

    while (const auto maybe_result = execution_result.PullOne()) {
    }

    record.Insert(kReturnSuccess, true);
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    record.Insert(kReturnSuccess, false);
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};
    mgp::AddProcedure(
        ExecuteQuery, kProcedureExecuteQuery, mgp::ProcedureType::Read,
        {mgp::Parameter(kArgumentInputQuery, mgp::Type::String), mgp::Parameter(kArgumentParameters, mgp::Type::Map)},
        {mgp::Return(kReturnSuccess, mgp::Type::Bool)}, module, memory);
  } catch (const std::exception &e) {
    return 1;
  }
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
