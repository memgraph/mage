#include <fmt/core.h>
#include <boost/algorithm/string/join.hpp>
#include <mgp.hpp>

#include "mgclient.hpp"

const char *kProcedureCase = "case";
const char *kArgumentConditionals = "conditionals";
const char *kArgumentElseQuery = "else_query";
const char *kArgumentParams = "params";

const char *kProcedureWhen = "when";
const char *kArgumentCondition = "condition";
const char *kArgumentIfQuery = "if_query";

const char *kReturnSuccess = "success";

struct ParamNames {
  std::vector<std::string> node_names;
  std::vector<std::string> relationship_names;
  std::vector<std::string> primitive_names;
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

  return ParamNames{
      .node_names = node_names, .relationship_names = relationship_names, .primitive_names = primitive_names};
}

std::string ConstructQueryPreffix(ParamNames names) {
  if (!names.node_names.size() && !names.relationship_names.size() && !names.primitive_names.size()) {
    return "";
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
  auto new_params = mg::Map(parameters.Size());

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

void ExecuteQuery(const std::string &query, const mgp::Map &query_parameters) {
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

  client->DiscardAll();

  mg::Client::Finalize();
}

void When(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;

  const auto arguments = mgp::List(args);
  const auto condition = arguments[0].ValueBool();

  const auto query_to_execute =
      condition ? std::string(arguments[1].ValueString()) : std::string(arguments[2].ValueString());
  const auto params = arguments[3].ValueMap();

  const auto record_factory = mgp::RecordFactory(result);
  auto record = record_factory.NewRecord();

  try {
    ExecuteQuery(query_to_execute, params);
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    record.Insert(kReturnSuccess, false);
    return;
  }

  record.Insert(kReturnSuccess, true);
  return;
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
  auto record = record_factory.NewRecord();

  try {
    ExecuteQuery(query_to_execute, params);
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    record.Insert(kReturnSuccess, false);
    return;
  }

  record.Insert(kReturnSuccess, true);
  return;
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    mgp::AddProcedure(
        Case, kProcedureCase, mgp::ProcedureType::Read,
        {mgp::Parameter(kArgumentConditionals, {mgp::Type::List, mgp::Type::Any}),
         mgp::Parameter(kArgumentElseQuery, mgp::Type::String), mgp::Parameter(kArgumentParams, mgp::Type::Map)},
        {mgp::Return(kReturnSuccess, mgp::Type::Bool)}, module, memory);

    mgp::AddProcedure(
        When, kProcedureWhen, mgp::ProcedureType::Read,
        {mgp::Parameter(kArgumentCondition, mgp::Type::Bool), mgp::Parameter(kArgumentIfQuery, mgp::Type::String),
         mgp::Parameter(kArgumentElseQuery, mgp::Type::String), mgp::Parameter(kArgumentParams, mgp::Type::Map)},
        {mgp::Return(kReturnSuccess, mgp::Type::Bool)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
