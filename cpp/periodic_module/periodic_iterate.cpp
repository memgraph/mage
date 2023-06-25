#include <fmt/core.h>
#include <mgp.hpp>

#include "mgclient.hpp"

const char *kProcedurePeriodic = "iterate";
const char *kArgumentInputQuery = "input_query";
const char *kArgumentRunningQuery = "running_query";
const char *kArgumentConfig = "config";
const char *kConfigKeyBatchSize = "batch_size";
const char *kBatchInternalName = "__batch";
const char *kBatchRowInternalName = "__batch_row";
const char *kReturnSuccess = "success";
const char *kReturnNumBatches = "number_of_executed_batches";

const char *kMgHost = "MG_HOST";
const char *kMgPort = "MG_PORT";
const char *kMgUsername = "MG_USERNAME";
const char *kMgPassword = "MG_PASSWORD";

const char *kDefaultHost = "localhost";
const uint16_t kDefaultPort = 7687;

struct ParamNames {
  std::vector<std::string> node_names;
  std::vector<std::string> relationship_names;
  std::vector<std::string> primitive_names;
};

auto ExtractParamNames(const std::vector<std::string> &columns, const std::vector<mg::Value> &batch_row) {
  ParamNames res;
  for (size_t i = 0; i < columns.size(); i++) {
    if (batch_row[i].type() == mg::Value::Type::Node) {
      res.node_names.push_back(columns[i]);
    } else if (batch_row[i].type() == mg::Value::Type::Relationship) {
      res.relationship_names.push_back(columns[i]);
    } else {
      res.primitive_names.push_back(columns[i]);
    }
  }

  return res;
}

auto Join(const std::vector<std::string> &strings, const std::string &delimiter) {
  if (!strings.size()) {
    return std::string();
  }

  auto result = strings[0];

  for (size_t i = 1; i < strings.size(); i++) {
    result += delimiter + strings[i];
  }

  return result;
}

auto GetGraphFirstClassEntityAlias(const std::string &internal_name, const std::string &entity_name) {
  return fmt::format("{}.{} AS __{}_id", internal_name, entity_name, entity_name);
}

auto GetPrimitiveEntityAlias(const std::string &internal_name, const std::string &primitive_name) {
  return fmt::format("{}.{} AS {}", internal_name, primitive_name, primitive_name);
}

auto ConstructWithStatement(const ParamNames &names) {
  std::vector<std::string> with_entity_vector;
  for (const auto &node_name : names.node_names) {
    with_entity_vector.push_back(GetGraphFirstClassEntityAlias(kBatchRowInternalName, node_name));
  }
  for (const auto &rel_name : names.relationship_names) {
    with_entity_vector.push_back(GetGraphFirstClassEntityAlias(kBatchRowInternalName, rel_name));
  }
  for (const auto &prim_name : names.primitive_names) {
    with_entity_vector.push_back(GetPrimitiveEntityAlias(kBatchRowInternalName, prim_name));
  }

  auto with_variables = fmt::format("WITH {}", Join(with_entity_vector, ", "));

  return with_variables;
}

auto ConstructMatchingNodeById(const std::string &node_name) {
  return fmt::format("MATCH ({}) WHERE ID({}) = __{}_id", node_name, node_name, node_name);
}

auto ConstructMatchingRelationshipById(const std::string &rel_name) {
  return fmt::format("MATCH ()-[{}]->() WHERE ID({}) = __{}_id", rel_name, rel_name, rel_name);
}

auto ConstructMatchGraphEntitiesById(const ParamNames &names) {
  std::string match_string = "";
  std::vector<std::string> match_by_id_vector;
  for (const auto &node_name : names.node_names) {
    match_by_id_vector.push_back(ConstructMatchingNodeById(node_name));
  }
  for (const auto &rel_name : names.relationship_names) {
    match_by_id_vector.push_back(ConstructMatchingRelationshipById(rel_name));
  }

  if (match_by_id_vector.size()) {
    match_string = Join(match_by_id_vector, " ");
  }

  return match_string;
}

auto ConstructQueryPrefix(const ParamNames &names) {
  if (!names.node_names.size() && !names.relationship_names.size() && !names.primitive_names.size()) {
    return std::string();
  }

  auto unwind_batch = fmt::format("UNWIND ${} AS {}", kBatchInternalName, kBatchRowInternalName);
  auto with_variables = ConstructWithStatement(names);
  auto match_string = ConstructMatchGraphEntitiesById(names);

  return fmt::format("{} {} {}", unwind_batch, with_variables, match_string);
}

auto ConstructQueryParams(const std::vector<std::string> &columns, const std::vector<std::vector<mg::Value>> &batch) {
  mg::Map params(1);
  mg::List list_value(batch.size());

  auto param_row_size = columns.size();

  for (size_t row = 0; row < batch.size(); row++) {
    mg::Map constructed_row(param_row_size);

    for (size_t i = 0; i < param_row_size; i++) {
      if (batch[row][i].type() == mg::Value::Type::Node) {
        constructed_row.Insert(columns[i], mg::Value(static_cast<int64_t>(batch[row][i].ValueNode().id().AsInt())));
      } else if (batch[row][i].type() == mg::Value::Type::Relationship) {
        constructed_row.Insert(columns[i],
                               mg::Value(static_cast<int64_t>(batch[row][i].ValueRelationship().id().AsInt())));
      } else {
        constructed_row.Insert(columns[i], batch[row][i]);
      }
    }

    list_value.Append(mg::Value(std::move(constructed_row)));
  }

  params.Insert(kBatchInternalName, mg::Value(std::move(list_value)));

  return params;
}

auto ConstructFinalQuery(const std::string &running_query, const std::string &prefix_query) {
  return fmt::format("{} {}", prefix_query, running_query);
}

void ExecuteRunningQuery(const std::string running_query, const std::vector<std::string> &columns,
                         const std::vector<std::vector<mg::Value>> &batch) {
  if (!batch.size()) {
    return;
  }

  auto param_names = ExtractParamNames(columns, batch[0]);
  auto prefix_query = ConstructQueryPrefix(param_names);
  auto final_query = ConstructFinalQuery(running_query, prefix_query);

  auto query_params = ConstructQueryParams(columns, batch);

  mg::Client::Params session_params{.host = "localhost", .port = 7687};
  auto client = mg::Client::Connect(session_params);
  if (!client) {
    throw std::runtime_error("Unable to connect to client!");
  }
  if (!client->Execute(final_query, query_params.AsConstMap())) {
    throw std::runtime_error("Error while executing periodic iterate!");
  }

  client->DiscardAll();
}

void ValidateBatchSize(const mgp::Value &batch_size_value) {
  if (batch_size_value.IsNull()) {
    throw std::runtime_error(fmt::format("Configuration parameter {} is not set.", kConfigKeyBatchSize));
  }
  if (!batch_size_value.IsInt()) {
    throw std::runtime_error("Batch size not provided as an integer in the periodic iterate configuration!");
  }

  const auto batch_size = batch_size_value.ValueInt();
  if (batch_size <= 0) {
    throw std::runtime_error("Batch size must be a non-negative number!");
  }
}

mg::Client::Params GetClientParams() {
  auto *host = kDefaultHost;
  auto port = kDefaultPort;
  auto *username = "";
  auto *password = "";

  auto *maybe_host = std::getenv(kMgHost);
  if (maybe_host) {
    host = std::move(maybe_host);
  }

  const auto *maybe_port = std::getenv(kMgPort);
  if (maybe_port) {
    port = static_cast<uint16_t>(std::move(*maybe_port));
  }

  const auto *maybe_username = std::getenv(kMgUsername);
  if (maybe_username) {
    username = std::move(maybe_username);
  }

  const auto *maybe_password = std::getenv(kMgPassword);
  if (maybe_password) {
    password = std::move(maybe_password);
  }

  return mg::Client::Params{.host = std::move(host),
                            .port = std::move(port),
                            .username = std::move(username),
                            .password = std::move(password)};
}

void PeriodicIterate(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);

  auto num_of_executed_batches = 0;
  const auto record_factory = mgp::RecordFactory(result);
  auto record = record_factory.NewRecord();

  const auto input_query = std::string(arguments[0].ValueString());
  const auto running_query = std::string(arguments[1].ValueString());
  const auto config = arguments[2].ValueMap();

  const auto batch_size_value = config.At(kConfigKeyBatchSize);

  try {
    ValidateBatchSize(batch_size_value);

    const auto batch_size = batch_size_value.ValueInt();

    mg::Client::Init();

    auto params = GetClientParams();
    auto client = mg::Client::Connect(params);

    if (!client) {
      throw std::runtime_error("Unable to connect to client!");
    }

    if (!client->Execute(input_query)) {
      record.Insert(kReturnSuccess, false);
      return;
    }

    auto columns = client->GetColumns();

    std::vector<std::vector<mg::Value>> batch;
    batch.reserve(batch_size);
    int rows = 0;
    while (const auto maybe_result = client->FetchOne()) {
      if ((*maybe_result).size() == 0) {
        break;
      }

      batch.push_back(std::move(*maybe_result));
      rows++;

      if (rows == batch_size) {
        ExecuteRunningQuery(running_query, columns, batch);
        num_of_executed_batches++;
        rows = 0;
        batch.clear();
      }
    }

    if (batch.size()) {
      ExecuteRunningQuery(running_query, columns, batch);
      num_of_executed_batches++;
    }

    mg::Client::Finalize();

    record.Insert(kReturnSuccess, true);
    record.Insert(kReturnNumBatches, static_cast<int64_t>(num_of_executed_batches));
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    record.Insert(kReturnSuccess, false);
    record.Insert(kReturnNumBatches, static_cast<int64_t>(num_of_executed_batches));
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    mgp::AddProcedure(
        PeriodicIterate, kProcedurePeriodic, mgp::ProcedureType::Read,
        {mgp::Parameter(kArgumentInputQuery, mgp::Type::String),
         mgp::Parameter(kArgumentRunningQuery, mgp::Type::String), mgp::Parameter(kArgumentConfig, mgp::Type::Map)},
        {mgp::Return(kReturnSuccess, mgp::Type::Bool), mgp::Return(kReturnNumBatches, mgp::Type::Int)}, module, memory);
  } catch (const std::exception &e) {
    return 1;
  }
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
