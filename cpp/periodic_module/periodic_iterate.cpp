#include <fmt/core.h>
#include <boost/algorithm/string/join.hpp>
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

const char *kProcedureFormatQuery = "format_query";
const char *kArgumentQuery = "query";
const char *kReturnQuery = "query";

struct ParamNames {
  std::vector<std::string> node_names;
  std::vector<std::string> relationship_names;
  std::vector<std::string> primitive_names;
};

ParamNames ExtractParamNames(std::vector<std::string> columns, std::vector<mg::Value> &batch_row) {
  std::vector<std::string> node_names;
  std::vector<std::string> relationship_names;
  std::vector<std::string> primitive_names;

  for (size_t i = 0; i < columns.size(); i++) {
    if (batch_row[i].type() == mg::Value::Type::Node) {
      node_names.push_back(columns[i]);
    } else if (batch_row[i].type() == mg::Value::Type::Relationship) {
      relationship_names.push_back(columns[i]);
    } else {
      primitive_names.push_back(columns[i]);
    }
  }

  return ParamNames{
      .node_names = node_names, .relationship_names = relationship_names, .primitive_names = primitive_names};
}

std::vector<std::string> ConstructColumns(std::vector<std::string> columns, std::vector<mg::Value> &batch_row) {
  std::vector<std::string> new_columns;

  for (size_t i = 0; i < columns.size(); i++) {
    if (batch_row[i].type() == mg::Value::Type::Node || batch_row[i].type() == mg::Value::Type::Relationship) {
      new_columns.push_back(fmt::format("__{}_id", columns[i]));
    } else {
      new_columns.push_back(columns[i]);
    }
  }

  return new_columns;
}

std::string ConstructQueryPreffix(ParamNames names) {
  if (!names.node_names.size() && !names.relationship_names.size() && !names.primitive_names.size()) {
    return "";
  }

  auto unwind_batch = fmt::format("UNWIND ${} AS {}", kBatchInternalName, kBatchRowInternalName);

  std::vector<std::string> with_entity_vector;
  for (const auto &node_name : names.node_names) {
    with_entity_vector.push_back(fmt::format("{}.{} AS __{}_id", kBatchRowInternalName, node_name, node_name));
  }
  for (const auto &rel_name : names.relationship_names) {
    with_entity_vector.push_back(fmt::format("{}.{} AS __{}_id", kBatchRowInternalName, rel_name, rel_name));
  }
  for (const auto &prim_name : names.primitive_names) {
    with_entity_vector.push_back(fmt::format("{}.{} AS {}", kBatchRowInternalName, prim_name, prim_name));
  }

  auto with_variables = fmt::format("WITH {}", boost::algorithm::join(with_entity_vector, ", "));

  std::string match_string = "";
  std::vector<std::string> match_by_id_vector;
  for (const auto &node_name : names.node_names) {
    match_by_id_vector.push_back(fmt::format("MATCH ({}) WHERE ID({}) = __{}_id", node_name, node_name, node_name));
  }
  for (const auto &rel_name : names.relationship_names) {
    match_by_id_vector.push_back(fmt::format("MATCH ()-[{}]->() WHERE ID({}) = __{}_id", rel_name, rel_name, rel_name));
  }

  if (match_by_id_vector.size()) {
    match_string = boost::algorithm::join(match_by_id_vector, " ");
  }

  return fmt::format("{} {} {}", unwind_batch, with_variables, match_string);
}

mg::Map ConstructParams(std::vector<std::string> columns, std::vector<std::vector<mg::Value>> &batch) {
  auto params = mg::Map(1);
  auto list_value = mg::List(batch.size());

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

std::string ConstructFinalQuery(std::string running_query, std::string preffix_query) {
  return fmt::format("{} {}", preffix_query, running_query);
}

void ExecuteRunningQuery(std::string running_query, std::vector<std::string> columns,
                         std::vector<std::vector<mg::Value>> &batch) {
  if (!batch.size()) {
    return;
  }

  auto param_names = ExtractParamNames(columns, batch[0]);
  auto query_columns = ConstructColumns(columns, batch[0]);

  auto preffix_query = ConstructQueryPreffix(param_names);
  auto final_query = ConstructFinalQuery(running_query, preffix_query);

  auto query_params = ConstructParams(columns, batch);

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

void PeriodicIterate(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);

  const auto input_query = std::string(arguments[0].ValueString());
  const auto running_query = std::string(arguments[1].ValueString());
  const auto config = arguments[2].ValueMap();

  const auto batch_size_value = config.At(kConfigKeyBatchSize);
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

  const auto record_factory = mgp::RecordFactory(result);
  auto record = record_factory.NewRecord();
  try {
    mg::Client::Init();

    mg::Client::Params params{.host = "localhost", .port = 7687};
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

      auto result = *maybe_result;

      if (rows == batch_size) {
        ExecuteRunningQuery(running_query, columns, batch);
        rows = 0;
        batch.clear();
      }

      batch.push_back(std::move(result));
    }

    ExecuteRunningQuery(running_query, columns, batch);

    mg::Client::Finalize();

    record.Insert(kReturnSuccess, true);
    return;
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    record.Insert(kReturnSuccess, false);
    return;
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    mgp::AddProcedure(
        PeriodicIterate, kProcedurePeriodic, mgp::ProcedureType::Read,
        {mgp::Parameter(kArgumentInputQuery, mgp::Type::String),
         mgp::Parameter(kArgumentRunningQuery, mgp::Type::String), mgp::Parameter(kArgumentConfig, mgp::Type::Map)},
        {mgp::Return(kReturnSuccess, mgp::Type::Bool)}, module, memory);
  } catch (const std::exception &e) {
    return 1;
  }
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
