#include <fmt/core.h>
#include <mgp.hpp>
#include <string>
#include <string_view>

#include "mgclient.hpp"

const char *kProcedurePeriodicIterate = "iterate";
const char *kProcedurePeriodicDelete = "delete";
const char *kArgumentInputQuery = "input_query";
const char *kArgumentRunningQuery = "running_query";
const char *kArgumentConfig = "config";
const char *kConfigKeyBatchSize = "batch_size";
const char *kBatchInternalName = "__batch";
const char *kBatchRowInternalName = "__batch_row";
const char *kConfigKeyLabels = "labels";
const char *kConfigKeyEdgeTypes = "edge_types";

const char *kReturnSuccess = "success";
const char *kReturnNumBatches = "number_of_executed_batches";
const char *kReturnNumDeletedNodes = "number_of_deleted_nodes";
const char *kReturnNumDeletedRelationships = "number_of_deleted_relationships";

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

struct DeletionInfo {
  uint64_t batch_size{0};
  std::vector<std::string> labels{};
  std::vector<std::string> edge_types{};
};

struct DeletionResult {
  uint64_t num_batches{0};
  uint64_t num_deleted_nodes{0};
  uint64_t num_deleted_relationships{0};
};

mg::Client::Params GetClientParams() {
  mg::Client::Params mg_params = {.host = "", .port = 0, .username = "", .password = ""};

  auto *maybe_host = std::getenv(kMgHost);
  if (maybe_host) {
    mg_params.host = std::string(maybe_host);
  }

  const auto *maybe_port = std::getenv(kMgPort);
  if (maybe_port) {
    mg_params.port = static_cast<uint16_t>(std::stoi(std::string(maybe_port)));
  }

  const auto *maybe_username = std::getenv(kMgUsername);
  if (maybe_username) {
    mg_params.username = std::string(maybe_username);
  }

  const auto *maybe_password = std::getenv(kMgPassword);
  if (maybe_password) {
    mg_params.password = std::string(maybe_password);
  }

  return mg_params;
}

ParamNames ExtractParamNames(const std::vector<std::string> &columns, const std::vector<mg::Value> &batch_row) {
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

std::string Join(const std::vector<std::string> &strings, const std::string &delimiter) {
  if (!strings.size()) {
    return "";
  }

  auto joined_strings_size = 0;
  for (const auto &string : strings) {
    joined_strings_size += string.size();
  }

  std::string joined_strings;
  joined_strings.reserve(joined_strings_size + delimiter.size() * (strings.size() - 1));

  joined_strings += strings[0];
  for (size_t i = 1; i < strings.size(); i++) {
    joined_strings += delimiter + strings[i];
  }

  return joined_strings;
}

std::string GetGraphFirstClassEntityAlias(const std::string &internal_name, const std::string &entity_name) {
  return fmt::format("{}.{} AS __{}_id", internal_name, entity_name, entity_name);
}

std::string GetPrimitiveEntityAlias(const std::string &internal_name, const std::string &primitive_name) {
  return fmt::format("{}.{} AS {}", internal_name, primitive_name, primitive_name);
}

std::string ConstructWithStatement(const ParamNames &names) {
  std::vector<std::string> with_entity_vector;
  for (const auto &node_name : names.node_names) {
    with_entity_vector.emplace_back(GetGraphFirstClassEntityAlias(kBatchRowInternalName, node_name));
  }
  for (const auto &rel_name : names.relationship_names) {
    with_entity_vector.emplace_back(GetGraphFirstClassEntityAlias(kBatchRowInternalName, rel_name));
  }
  for (const auto &prim_name : names.primitive_names) {
    with_entity_vector.emplace_back(GetPrimitiveEntityAlias(kBatchRowInternalName, prim_name));
  }

  return fmt::format("WITH {}", Join(with_entity_vector, ", "));
}

std::string ConstructMatchingNodeById(const std::string &node_name) {
  return fmt::format("MATCH ({}) WHERE ID({}) = __{}_id", node_name, node_name, node_name);
}

std::string ConstructMatchingRelationshipById(const std::string &rel_name) {
  return fmt::format("MATCH ()-[{}]->() WHERE ID({}) = __{}_id", rel_name, rel_name, rel_name);
}

std::string ConstructMatchGraphEntitiesById(const ParamNames &names) {
  std::string match_string = "";
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

std::string ConstructQueryPrefix(const ParamNames &names) {
  if (!names.node_names.size() && !names.relationship_names.size() && !names.primitive_names.size()) {
    return std::string();
  }

  auto unwind_batch = fmt::format("UNWIND ${} AS {}", kBatchInternalName, kBatchRowInternalName);
  auto with_variables = ConstructWithStatement(names);
  auto match_string = ConstructMatchGraphEntitiesById(names);

  return fmt::format("{} {} {}", unwind_batch, with_variables, match_string);
}

mg::Map ConstructQueryParams(const std::vector<std::string> &columns,
                             const std::vector<std::vector<mg::Value>> &batch) {
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

std::string ConstructFinalQuery(const std::string &running_query, const std::string &prefix_query) {
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

void ValidateDeletionConfigEntities(const mgp::Map &config, std::string config_key) {
  auto key = std::string_view(config_key);
  if (!config.KeyExists(key)) {
    return;
  }

  auto value = config.At(key);
  if (!value.IsString() && !value.IsList()) {
    throw std::runtime_error(fmt::format("Invalid config for config parameter {}!", config_key));
  }

  if (value.IsString()) {
    return;
  }

  if (value.IsList()) {
    auto list_value = value.ValueList();
    for (auto elem : list_value) {
      if (!elem.IsString()) {
        throw std::runtime_error(fmt::format("Invalid config for config parameter {}!", config_key));
      }
    }
  }
}

void ValidateDeletionConfig(const mgp::Map &config) {
  auto batch_size_key = std::string(kConfigKeyBatchSize);
  auto labels_key = std::string(kConfigKeyLabels);
  auto edge_types_key = std::string(kConfigKeyEdgeTypes);

  if (!config.KeyExists(batch_size_key)) {
    throw std::runtime_error("Periodic.delete() did not specify config parameter batch_size!");
  }

  auto batch_size_value = config.At(batch_size_key);
  if (!batch_size_value.IsInt()) {
    throw std::runtime_error("Batch size needs to be an integer!");
  }

  if (batch_size_value.ValueInt() <= 0) {
    throw std::runtime_error("Batch size can't be a non-negative integer!");
  }

  ValidateDeletionConfigEntities(config, labels_key);
  ValidateDeletionConfigEntities(config, edge_types_key);
}

void EmplaceFromConfig(const mgp::Map &config, std::vector<std::string> &vec, std::string &config_key) {
  auto key = std::string_view(config_key);
  if (config.KeyExists(key)) {
    auto value = config.At(key);
    if (value.IsString()) {
      vec.emplace_back(std::string(value.ValueString()));
    } else if (value.IsList()) {
      auto list_value = value.ValueList();
      for (const auto elem : list_value) {
        vec.emplace_back(elem.ValueString());
      }
    }
  }
}

DeletionInfo GetDeletionInfo(const mgp::Map &config) {
  std::vector<std::string> labels, edge_types;

  ValidateDeletionConfig(config);

  auto batch_size_key = std::string(kConfigKeyBatchSize);
  auto labels_key = std::string(kConfigKeyLabels);
  auto edge_types_key = std::string(kConfigKeyEdgeTypes);

  auto batch_size = config.At(batch_size_key).ValueInt();

  EmplaceFromConfig(config, labels, labels_key);
  EmplaceFromConfig(config, edge_types, edge_types_key);

  return {.batch_size = static_cast<uint64_t>(batch_size),
          .labels = std::move(labels),
          .edge_types = std::move(edge_types)};
}

void ExecutePeriodicDelete(DeletionInfo deletion_info, DeletionResult &deletion_result) {
  auto delete_all = deletion_info.edge_types.empty() && deletion_info.labels.empty();
  auto delete_nodes = delete_all || !deletion_info.labels.empty();
  auto delete_edges = delete_all || !deletion_info.labels.empty() || !deletion_info.edge_types.empty();

  auto labels_formatted = deletion_info.labels.empty() ? "" : fmt::format(":{}", Join(deletion_info.labels, ":"));
  auto edge_types_formatted =
      deletion_info.edge_types.empty() ? "" : fmt::format(":{}", Join(deletion_info.edge_types, "|"));

  auto relationships_deletion_query =
      fmt::format("MATCH (n{})-[r{}]-(m) WITH DISTINCT r LIMIT {} DELETE r RETURN count(r) AS num_deleted",
                  labels_formatted, edge_types_formatted, deletion_info.batch_size);
  auto nodes_deletion_query =
      fmt::format("MATCH (n{}) WITH DISTINCT n LIMIT {} DETACH DELETE n RETURN count(n) AS num_deleted",
                  labels_formatted, deletion_info.batch_size);

  auto client = mg::Client::Connect(GetClientParams());
  if (!client) {
    throw std::runtime_error("Unable to connect to client!");
  }

  if (delete_edges) {
    while (true) {
      if (!client->Execute(relationships_deletion_query)) {
        throw std::runtime_error("Error while executing periodic iterate!");
      }

      auto result = client->FetchOne();
      if (!result || (*result).size() != 1) {
        throw std::runtime_error("No result received from periodic delete!");
      }

      client->DiscardAll();

      uint64_t num_deleted = static_cast<uint64_t>((*result)[0].ValueInt());
      deletion_result.num_batches++;
      deletion_result.num_deleted_relationships += num_deleted;
      if (static_cast<uint64_t>(num_deleted) < deletion_info.batch_size) {
        break;
      }
    }
  }

  if (delete_nodes) {
    while (true) {
      if (!client->Execute(nodes_deletion_query)) {
        throw std::runtime_error("Error while executing periodic iterate!");
      }

      auto result = client->FetchOne();
      if (!result || (*result).size() != 1) {
        throw std::runtime_error("No result received from periodic delete!");
      }

      client->DiscardAll();

      uint64_t num_deleted = static_cast<uint64_t>((*result)[0].ValueInt());
      deletion_result.num_batches++;
      deletion_result.num_deleted_nodes += num_deleted;
      if (static_cast<uint64_t>(num_deleted) < deletion_info.batch_size) {
        break;
      }
    }
  }
}

void PeriodicDelete(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);

  const auto record_factory = mgp::RecordFactory(result);
  auto record = record_factory.NewRecord();

  const auto config = arguments[0].ValueMap();

  DeletionResult deletion_result;

  try {
    mg::Client::Init();

    auto client = mg::Client::Connect(GetClientParams());

    if (!client) {
      throw std::runtime_error("Unable to connect to client!");
    }

    auto deletion_info = GetDeletionInfo(config);

    ExecutePeriodicDelete(std::move(deletion_info), deletion_result);

    mg::Client::Finalize();

    record.Insert(kReturnSuccess, true);
    record.Insert(kReturnNumBatches, static_cast<int64_t>(deletion_result.num_batches));
    record.Insert(kReturnNumDeletedNodes, static_cast<int64_t>(deletion_result.num_deleted_nodes));
    record.Insert(kReturnNumDeletedRelationships, static_cast<int64_t>(deletion_result.num_deleted_relationships));
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    record.Insert(kReturnSuccess, false);
    record.Insert(kReturnNumBatches, static_cast<int64_t>(deletion_result.num_batches));
    record.Insert(kReturnNumDeletedNodes, static_cast<int64_t>(deletion_result.num_deleted_nodes));
    record.Insert(kReturnNumDeletedRelationships, static_cast<int64_t>(deletion_result.num_deleted_relationships));
  }
}

void PeriodicIterate(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
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

    auto client = mg::Client::Connect(GetClientParams());

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
    mgp::MemoryDispatcherGuard guard{memory};
    mgp::AddProcedure(
        PeriodicIterate, kProcedurePeriodicIterate, mgp::ProcedureType::Read,
        {mgp::Parameter(kArgumentInputQuery, mgp::Type::String),
         mgp::Parameter(kArgumentRunningQuery, mgp::Type::String), mgp::Parameter(kArgumentConfig, mgp::Type::Map)},
        {mgp::Return(kReturnSuccess, mgp::Type::Bool), mgp::Return(kReturnNumBatches, mgp::Type::Int)}, module, memory);

    mgp::AddProcedure(PeriodicDelete, kProcedurePeriodicDelete, mgp::ProcedureType::Read,
                      {mgp::Parameter(kArgumentConfig, mgp::Type::Map)},
                      {mgp::Return(kReturnSuccess, mgp::Type::Bool), mgp::Return(kReturnNumBatches, mgp::Type::Int),
                       mgp::Return(kReturnNumDeletedNodes, mgp::Type::Int),
                       mgp::Return(kReturnNumDeletedRelationships, mgp::Type::Int)},
                      module, memory);
  } catch (const std::exception &e) {
    return 1;
  }
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
