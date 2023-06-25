#include <fmt/core.h>
#include <mgp.hpp>

#include "mgclient.hpp"

const char *kProcedureCase = "case";
const char *kArgumentConditionals = "conditionals";
const char *kArgumentElseQuery = "else_query";
const char *kArgumentParams = "params";

const char *kProcedureWhen = "when";
const char *kArgumentCondition = "condition";
const char *kArgumentIfQuery = "if_query";

const char *kReturnValue = "value";

const std::vector<std::string_view> kGlobalOperations = {"CREATE INDEX ON",
                                                         "DROP INDEX ON",
                                                         "CREATE CONSTRAINT ON",
                                                         "DROP CONSTRAINT ON",
                                                         "SET GLOBAL TRANSACTION ISOLATION LEVEL",
                                                         "STORAGE MODE IN_MEMORY_TRANSACTIONAL",
                                                         "STORAGE MODE IN_MEMORY_ANALYTICAL"};

struct ParamNames {
  std::vector<std::string> node_names;
  std::vector<std::string> relationship_names;
  std::vector<std::string> primitive_names;
};

struct QueryResults {
  std::vector<std::string> columns;
  std::vector<std::vector<mg::Value>> results;
};

ParamNames ExtractParamNames(const mgp::Map &parameters) {
  std::vector<std::string> node_names;
  std::vector<std::string> relationship_names;
  std::vector<std::string> primitive_names;

  for (const auto &map_item : parameters) {
    if (map_item.value.IsNode()) {
      node_names.push_back(std::string(map_item.key));
    } else if (map_item.value.IsRelationship()) {
      relationship_names.push_back(std::string(map_item.key));
    } else {
      primitive_names.push_back(std::string(map_item.key));
    }
  }

  return ParamNames{.node_names = std::move(node_names),
                    .relationship_names = std::move(relationship_names),
                    .primitive_names = std::move(primitive_names)};
}

auto Join(std::vector<std::string> &strings, const std::string &delimiter) {
  std::string s;

  if (!strings.size()) {
    return s;
  }

  s += strings[0];

  for (size_t i = 1; i < strings.size(); i++) {
    s += delimiter + strings[i];
  }

  return s;
}

auto ConstructQueryPreffix(const ParamNames &names) {
  if (!names.node_names.size() && !names.relationship_names.size() && !names.primitive_names.size()) {
    return std::string();
  }

  std::vector<std::string> with_entity_vector;
  for (const auto &node_name : names.node_names) {
    with_entity_vector.push_back(fmt::format("${} AS __{}_id", node_name, node_name));
  }
  for (const auto &rel_name : names.relationship_names) {
    with_entity_vector.push_back(fmt::format("${} AS __{}_id", rel_name, rel_name));
  }
  for (const auto &prim_name : names.primitive_names) {
    with_entity_vector.push_back(fmt::format("${} AS {}", prim_name, prim_name));
  }

  auto with_variables = fmt::format("WITH {}", Join(with_entity_vector, ", "));

  std::string match_string = "";
  std::vector<std::string> match_by_id_vector;
  for (const auto &node_name : names.node_names) {
    match_by_id_vector.push_back(fmt::format("MATCH ({}) WHERE ID({}) = __{}_id", node_name, node_name, node_name));
  }
  for (const auto &rel_name : names.relationship_names) {
    match_by_id_vector.push_back(fmt::format("MATCH ()-[{}]->() WHERE ID({}) = __{}_id", rel_name, rel_name, rel_name));
  }

  if (match_by_id_vector.size()) {
    match_string = Join(match_by_id_vector, " ");
  }

  return fmt::format("{} {}", with_variables, match_string);
}

auto ConstructPreffixQuery(const mgp::Map &parameters) {
  const auto param_names = ExtractParamNames(parameters);

  return ConstructQueryPreffix(param_names);
}

auto ConstructFinalQuery(const std::string &running_query, const std::string &preffix_query) {
  return fmt::format("{} {}", preffix_query, running_query);
}

auto ConstructParams(const ParamNames &param_names, const mgp::Map &parameters) {
  mg::Map new_params{parameters.Size()};

  for (const auto &map_item : parameters) {
    if (map_item.value.IsNode()) {
      new_params.Insert(map_item.key, mg::Value(static_cast<int64_t>(map_item.value.ValueNode().Id().AsInt())));
    } else if (map_item.value.IsRelationship()) {
      new_params.Insert(map_item.key, mg::Value(static_cast<int64_t>(map_item.value.ValueRelationship().Id().AsInt())));
    } else if (map_item.value.IsBool()) {
      new_params.Insert(map_item.key, mg::Value(map_item.value.ValueBool()));
    } else if (map_item.value.IsString()) {
      new_params.Insert(map_item.key, mg::Value(map_item.value.ValueString()));
    } else if (map_item.value.IsInt()) {
      new_params.Insert(map_item.key, mg::Value(map_item.value.ValueInt()));
    } else if (map_item.value.IsDouble()) {
      new_params.Insert(map_item.key, mg::Value(map_item.value.ValueDouble()));
    } else {
      // Temporal types and paths not yet supported
      throw std::runtime_error("Can't parse some of the arguments!");
    }
  }

  return new_params;
}

auto ExecuteQuery(const std::string &query, const mgp::Map &query_parameters) {
  mg::Client::Init();
  mg::Client::Params params{.host = "localhost", .port = 7687};

  auto client = mg::Client::Connect(params);

  if (!client) {
    throw std::runtime_error("Unable to connect to client!");
  }

  auto param_names = ExtractParamNames(query_parameters);
  auto preffix_query = ConstructQueryPreffix(param_names);
  auto final_query = ConstructFinalQuery(query, preffix_query);

  const auto final_parameters = ConstructParams(param_names, query_parameters);

  if (!client->Execute(final_query, final_parameters.AsConstMap())) {
    throw std::runtime_error("Error while executing do module!");
  }

  auto columns = client->GetColumns();

  std::vector<std::vector<mg::Value>> results;
  while (const auto maybe_result = client->FetchOne()) {
    if ((*maybe_result).size() == 0) {
      break;
    }

    auto result = *maybe_result;
    results.push_back(std::move(result));
  }

  client->DiscardAll();

  mg::Client::Finalize();

  return QueryResults{.columns = std::move(columns), .results = std::move(results)};
}

void InsertConditionalResults(const mgp::RecordFactory &record_factory, const QueryResults &query_results) {
  for (const auto &row : query_results.results) {
    auto result_map = mgp::Map();

    for (size_t i = 0; i < query_results.columns.size(); i++) {
      const auto &result = row[i];
      const auto &column = query_results.columns[i];

      if (result.type() == mg::Value::Type::Bool) {
        result_map.Insert(column, mgp::Value(result.ValueBool()));
      } else if (result.type() == mg::Value::Type::String) {
        auto string_value = mgp::Value(std::string(result.ValueString()));
        result_map.Insert(column, string_value);
      } else if (result.type() == mg::Value::Type::Int) {
        result_map.Insert(column, mgp::Value(result.ValueInt()));
      } else if (result.type() == mg::Value::Type::Double) {
        result_map.Insert(column, mgp::Value(result.ValueDouble()));
      } else if (result.type() == mg::Value::Type::Node) {
        throw std::runtime_error("Returning nodes in do procedures not yet supported.");
      } else if (result.type() == mg::Value::Type::Relationship) {
        throw std::runtime_error("Returning relationships in do procedures not yet supported.");
      } else if (result.type() == mg::Value::Type::Path) {
        throw std::runtime_error("Returning paths in do procedures not yet supported.");
      } else {
        throw std::runtime_error(
            fmt::format("Returning type in column {} in do procedures not yet supported!", column));
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(kReturnValue, result_map);
  }
}

bool IsGlobalOperation(const std::string_view &query) {
  for (const auto &global_op : kGlobalOperations) {
    if (query.starts_with(global_op)) {
      return true;
    }
  }
  return false;
}

void When(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;

  const auto arguments = mgp::List(args);
  const auto condition = arguments[0].ValueBool();

  const auto query_to_execute =
      condition ? std::string(arguments[1].ValueString()) : std::string(arguments[2].ValueString());
  const auto params = arguments[3].ValueMap();

  const auto record_factory = mgp::RecordFactory(result);

  try {
    if (IsGlobalOperation(query_to_execute)) {
      throw std::runtime_error(fmt::format(
          "The query {} isn’t supported by `do.when` because it would execute a global operation.", query_to_execute));
    }

    const auto query_results = ExecuteQuery(query_to_execute, params);
    InsertConditionalResults(record_factory, query_results);
    return;
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Case(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;

  const auto arguments = mgp::List(args);
  const auto conditionals = arguments[0].ValueList();
  const auto else_query = std::string(arguments[1].ValueString());
  const auto params = arguments[2].ValueMap();

  if (conditionals.Empty()) {
    throw std::runtime_error("Conditionals list must not be empty!");
  }

  const auto conditionals_size = conditionals.Size();

  if (conditionals_size % 2) {
    throw std::runtime_error("Size of the conditionals size must be even!");
  }

  for (size_t i = 0; i < conditionals_size; i++) {
    if (!(i % 2) && !conditionals[i].IsBool()) {
      throw std::runtime_error(fmt::format("Argument on index {} in do.case conditionals is not bool!", i));
    } else if (i % 2 && !conditionals[i].IsString()) {
      throw std::runtime_error(fmt::format("Argument on index {} in do.case conditionals is not string!", i));
    }
  }

  auto found_true_conditional = -1;
  for (size_t i = 0; i < conditionals_size; i += 2) {
    const auto conditional = conditionals[i].ValueBool();
    if (conditional) {
      found_true_conditional = i;
      break;
    }
  }

  const auto query_to_execute =
      found_true_conditional != -1 ? std::string(conditionals[found_true_conditional + 1].ValueString()) : else_query;

  const auto record_factory = mgp::RecordFactory(result);

  try {
    if (IsGlobalOperation(query_to_execute)) {
      throw std::runtime_error(fmt::format(
          "The query {} isn’t supported by `do.case` because it would execute a global operation.", query_to_execute));
    }

    const auto query_results = ExecuteQuery(query_to_execute, params);
    InsertConditionalResults(record_factory, query_results);
    return;
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    mgp::AddProcedure(Case, kProcedureCase, mgp::ProcedureType::Read,
                      {mgp::Parameter(kArgumentConditionals, {mgp::Type::List, mgp::Type::Any}),
                       mgp::Parameter(kArgumentElseQuery, mgp::Type::String),
                       mgp::Parameter(kArgumentParams, mgp::Type::Map, mgp::Value(mgp::Map()))},
                      {mgp::Return(kReturnValue, mgp::Type::Map)}, module, memory);

    mgp::AddProcedure(
        When, kProcedureWhen, mgp::ProcedureType::Read,
        {mgp::Parameter(kArgumentCondition, mgp::Type::Bool), mgp::Parameter(kArgumentIfQuery, mgp::Type::String),
         mgp::Parameter(kArgumentElseQuery, mgp::Type::String),
         mgp::Parameter(kArgumentParams, mgp::Type::Map, mgp::Value(mgp::Map()))},
        {mgp::Return(kReturnValue, mgp::Type::Map)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
