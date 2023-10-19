#include <string_view>

#include <fmt/core.h>
#include <mgp.hpp>

#include "mgclient.hpp"

constexpr std::string_view kProcedureCase = "case";
constexpr std::string_view kArgumentConditionals = "conditionals";
constexpr std::string_view kArgumentElseQuery = "else_query";
constexpr std::string_view kArgumentParams = "params";

constexpr std::string_view kProcedureWhen = "when";
constexpr std::string_view kArgumentCondition = "condition";
constexpr std::string_view kArgumentIfQuery = "if_query";

constexpr std::string_view kReturnValue = "value";

constexpr std::string_view kMgHost = "MG_HOST";
constexpr std::string_view kMgPort = "MG_PORT";
constexpr std::string_view kMgUsername = "MG_USERNAME";
constexpr std::string_view kMgPassword = "MG_PASSWORD";

constexpr std::string_view kDefaultHost = "localhost";
constexpr uint16_t kDefaultPort = 7687;

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
  ParamNames res;
  for (const auto &map_item : parameters) {
    switch (map_item.value.Type()) {
      case mgp::Type::Node:
        res.node_names.emplace_back(map_item.key);
        break;
      case mgp::Type::Relationship:
        res.relationship_names.emplace_back(map_item.key);
        break;
      default:
        res.primitive_names.emplace_back(map_item.key);
    }
  }

  return res;
}

std::string Join(std::vector<std::string> const &strings, std::string_view delimiter) {
  if (!strings.size()) {
    return "";
  }

  auto joined_strings_size = strings[0].size();
  for (size_t i = 1; i < strings.size(); i++) {
    joined_strings_size += strings[i].size();
  }

  std::string joined_strings;
  joined_strings.reserve(joined_strings_size + delimiter.size() * (strings.size() - 1));

  joined_strings += std::move(strings[0]);
  for (size_t i = 1; i < strings.size(); i++) {
    joined_strings += delimiter;
    joined_strings += std::move(strings[i]);
  }

  return joined_strings;
}

std::string GetGraphFirstClassEntityAlias(const std::string &entity_name) {
  return fmt::format("${0} AS __{0}_id", entity_name);
}

std::string GetPrimitiveEntityAlias(const std::string &primitive_name) {
  return fmt::format("${0} AS {0}", primitive_name);
}

std::string ConstructWithStatement(const ParamNames &names) {
  std::vector<std::string> with_entity_vector;
  for (const auto &node_name : names.node_names) {
    with_entity_vector.emplace_back(GetGraphFirstClassEntityAlias(node_name));
  }
  for (const auto &rel_name : names.relationship_names) {
    with_entity_vector.emplace_back(GetGraphFirstClassEntityAlias(rel_name));
  }
  for (const auto &prim_name : names.primitive_names) {
    with_entity_vector.emplace_back(GetPrimitiveEntityAlias(prim_name));
  }

  return fmt::format("WITH {}", Join(std::move(with_entity_vector), ", "));
}

std::string ConstructMatchingNodeById(const std::string &node_name) {
  return fmt::format("MATCH ({0}) WHERE ID({0}) = __{0}_id", node_name);
}

std::string ConstructMatchingRelationshipById(const std::string &rel_name) {
  return fmt::format("MATCH ()-[{0}]->() WHERE ID({0}) = __{0}_id", rel_name);
}

std::string ConstructMatchGraphEntitiesById(const ParamNames &names) {
  std::string match_string{""};
  std::vector<std::string> match_by_id_vector;
  for (const auto &node_name : names.node_names) {
    match_by_id_vector.emplace_back(ConstructMatchingNodeById(node_name));
  }
  for (const auto &rel_name : names.relationship_names) {
    match_by_id_vector.emplace_back(ConstructMatchingRelationshipById(rel_name));
  }

  if (match_by_id_vector.size()) {
    match_string = Join(match_by_id_vector, " ");
  }

  return match_string;
}

std::string ConstructQueryPreffix(const ParamNames &names) {
  if (!names.node_names.size() && !names.relationship_names.size() && !names.primitive_names.size()) {
    return std::string();
  }

  auto with_variables = ConstructWithStatement(names);
  auto match_string = ConstructMatchGraphEntitiesById(names);

  return fmt::format("{} {}", with_variables, match_string);
}

std::string ConstructPreffixQuery(const mgp::Map &parameters) {
  const auto param_names = ExtractParamNames(parameters);

  return ConstructQueryPreffix(param_names);
}

std::string ConstructFinalQuery(const std::string &running_query, const std::string &preffix_query) {
  return fmt::format("{} {}", preffix_query, running_query);
}

mg::Map ConstructParams(const ParamNames &param_names, const mgp::Map &parameters) {
  mg::Map new_params{parameters.Size()};

  for (const auto &map_item : parameters) {
    switch (map_item.value.Type()) {
      case mgp::Type::Node:
        new_params.Insert(map_item.key, mg::Value(static_cast<int64_t>(map_item.value.ValueNode().Id().AsInt())));
        break;
      case mgp::Type::Relationship:
        new_params.Insert(map_item.key,
                          mg::Value(static_cast<int64_t>(map_item.value.ValueRelationship().Id().AsInt())));
        break;
      case mgp::Type::Bool:
        new_params.Insert(map_item.key, mg::Value(map_item.value.ValueBool()));
        break;
      case mgp::Type::String:
        new_params.Insert(map_item.key, mg::Value(map_item.value.ValueString()));
        break;
      case mgp::Type::Int:
        new_params.Insert(map_item.key, mg::Value(map_item.value.ValueInt()));
        break;
      case mgp::Type::Double:
        new_params.Insert(map_item.key, mg::Value(map_item.value.ValueDouble()));
        break;
      default:
        // Temporal types and paths not yet supported
        throw std::runtime_error("Can't parse some of the arguments!");
        break;
    }
  }

  return new_params;
}

mg::Client::Params GetClientParams() {
  auto host = std::string(kDefaultHost);
  auto port = kDefaultPort;
  auto *username = "";
  auto *password = "";

  auto *maybe_host = std::getenv(std::string(kMgHost).c_str());
  if (maybe_host) {
    host = std::move(maybe_host);
  }

  const auto *maybe_port = std::getenv(std::string(kMgPort).c_str());
  if (maybe_port) {
    port = static_cast<uint16_t>(std::move(*maybe_port));
  }

  const auto *maybe_username = std::getenv(std::string(kMgUsername).c_str());
  if (maybe_username) {
    username = std::move(maybe_username);
  }

  const auto *maybe_password = std::getenv(std::string(kMgPassword).c_str());
  if (maybe_password) {
    password = std::move(maybe_password);
  }

  return mg::Client::Params{.host = std::move(host),
                            .port = std::move(port),
                            .username = std::move(username),
                            .password = std::move(password)};
}

QueryResults ExecuteQuery(const std::string &query, const mgp::Map &query_parameters) {
  mg::Client::Init();

  auto client = mg::Client::Connect(GetClientParams());

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

      switch (result.type()) {
        case mg::Value::Type::Bool:
          result_map.Insert(column, mgp::Value(result.ValueBool()));
          break;
        case mg::Value::Type::String: {
          auto string_value = mgp::Value(std::string(result.ValueString()));
          result_map.Insert(column, string_value);
          break;
        }
        case mg::Value::Type::Int:
          result_map.Insert(column, mgp::Value(result.ValueInt()));
          break;
        case mg::Value::Type::Double:
          result_map.Insert(column, mgp::Value(result.ValueDouble()));
          break;
        case mg::Value::Type::Node:
          throw std::runtime_error("Returning nodes in do procedures not yet supported.");
          break;
        case mg::Value::Type::Relationship:
          throw std::runtime_error("Returning relationships in do procedures not yet supported.");
          break;
        case mg::Value::Type::Path:
          throw std::runtime_error("Returning paths in do procedures not yet supported.");
          break;
        default:
          throw std::runtime_error(
              fmt::format("Returning type in column {} in do procedures not yet supported!", column));
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kReturnValue).c_str(), result_map);
  }
}

bool IsGlobalOperation(std::string_view query) {
  for (const auto &global_op : kGlobalOperations) {
    if (query.starts_with(global_op)) {
      return true;
    }
  }
  return false;
}

void When(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};;

  const auto arguments = mgp::List(args);
  const auto condition = arguments[0].ValueBool();

  const auto &query_to_execute = std::string(arguments[condition ? 1 : 2].ValueString());
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
  mgp::MemoryDispatcherGuard guard{memory};;

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
    mgp::MemoryDispatcherGuard guard{memory};;

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
