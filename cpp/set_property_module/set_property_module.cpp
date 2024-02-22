#include <map>
#include <unordered_set>

#include <fmt/core.h>
#include <mgp.hpp>

#include "mgclient.hpp"

constexpr static std::string_view kResult = "result";
constexpr static std::string_view kQueryExecuted = "queryExecuted";
constexpr static std::string_view kSourceProperties = "sourceProperties";
constexpr static std::string_view kSourceVariable = "sourceVariable";
constexpr static std::string_view kSourceNode = "sourceNode";
constexpr static std::string_view kSourceRel = "sourceRel";
constexpr static std::string_view kTargetProperties = "targetProperties";
constexpr static std::string_view kTargetVariable = "targetVariable";
constexpr static std::string_view kTargetNode = "targetNode";
constexpr static std::string_view kTargetRel = "targetRel";

constexpr char *kMgHost = "MG_HOST";
constexpr char *kMgPort = "MG_PORT";
constexpr char *kMgUsername = "MG_USERNAME";
constexpr char *kMgPassword = "MG_PASSWORD";

constexpr char *kDefaultHost = "localhost";
constexpr uint16_t kDefaultPort = 7687;

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

void ExecuteSetPropertiesQuery(const std::string query) {
  auto client = mg::Client::Connect(GetClientParams());
  if (!client) {
    throw std::runtime_error("Unable to connect to client!");
  }
  if (!client->Execute(query)) {
    throw std::runtime_error("Error while executing set property query, please look at the logs!");
  }

  client->DiscardAll();
}

// -------------------------------------------------------------------------------
// ---------------------- OPTIMIZED METHODS --------------------------------------
// -------------------------------------------------------------------------------

void CopyPropertyNode2Node(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard(memory);

  auto arguments = mgp::List(args);
  auto record_factory = mgp::RecordFactory(result);
  auto record = record_factory.NewRecord();

  try {
    if (!arguments[0].IsNode()) {
      throw std::runtime_error("CopyPropertyNode2Node argument source entity is not a node!");
    }
    if (!arguments[2].IsNode()) {
      throw std::runtime_error("CopyPropertyNode2Node argument target entity is not a node!");
    }

    auto source_node = arguments[0].ValueNode();
    auto source_properties = arguments[1].ValueList();

    auto target_node = arguments[2].ValueNode();
    auto target_properties = arguments[3].ValueList();

    if (source_properties.Empty() && target_properties.Empty()) {
      record.Insert(kResult.data(), true);
      return;
    }

    if (source_properties.Size() != target_properties.Size()) {
      throw std::runtime_error(
          "CopyPropertyNode2Node source properties and target properties are not of the same size!");
    }

    std::unordered_map<std::string, mgp::Value> source_prop_map = source_node.Properties();
    std::unordered_map<std::string_view, mgp::Value> target_prop_map;
    for (size_t i = 0, size = source_properties.Size(); i < size; i++) {
      target_prop_map[target_properties[i].ValueString()] =
          source_prop_map[std::string(source_properties[i].ValueString())];
    }

    target_node.SetProperties(target_prop_map);

    record.Insert(kResult.data(), true);
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    record.Insert(kResult.data(), false);
  }
}

void CopyPropertyNode2Rel(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard(memory);

  auto arguments = mgp::List(args);
  auto record_factory = mgp::RecordFactory(result);
  auto record = record_factory.NewRecord();

  try {
    if (!arguments[0].IsNode()) {
      throw std::runtime_error("CopyPropertyNode2Rel argument source entity is not a node!");
    }
    if (!arguments[2].IsRelationship()) {
      throw std::runtime_error("CopyPropertyNode2Rel argument target entity is not a relationship!");
    }

    auto source_node = arguments[0].ValueNode();
    auto source_properties = arguments[1].ValueList();

    auto target_rel = arguments[2].ValueRelationship();
    auto target_properties = arguments[3].ValueList();

    if (source_properties.Empty() && target_properties.Empty()) {
      record.Insert(kResult.data(), true);
      return;
    }

    if (source_properties.Size() != target_properties.Size()) {
      throw std::runtime_error(
          "CopyPropertyNode2Rel source properties and target properties are not of the same size!");
    }

    std::unordered_map<std::string, mgp::Value> source_prop_map = source_node.Properties();
    std::unordered_map<std::string_view, mgp::Value> target_prop_map;
    for (size_t i = 0, size = source_properties.Size(); i < size; i++) {
      target_prop_map[target_properties[i].ValueString()] =
          source_prop_map[std::string(source_properties[i].ValueString())];
    }

    target_rel.SetProperties(target_prop_map);

    record.Insert(kResult.data(), true);
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    record.Insert(kResult.data(), false);
  }
}

void CopyPropertyRel2Node(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard(memory);

  auto arguments = mgp::List(args);
  auto record_factory = mgp::RecordFactory(result);
  auto record = record_factory.NewRecord();

  try {
    if (!arguments[0].IsRelationship()) {
      throw std::runtime_error("CopyPropertyRel2Node argument source entity is not a relationship!");
    }
    if (!arguments[2].IsNode()) {
      throw std::runtime_error("CopyPropertyRel2Node argument target entity is not a node!");
    }

    auto source_rel = arguments[0].ValueRelationship();
    auto source_properties = arguments[1].ValueList();

    auto target_node = arguments[2].ValueNode();
    auto target_properties = arguments[3].ValueList();

    if (source_properties.Empty() && target_properties.Empty()) {
      record.Insert(kResult.data(), true);
      return;
    }

    if (source_properties.Size() != target_properties.Size()) {
      throw std::runtime_error(
          "CopyPropertyRel2Node source properties and target properties are not of the same size!");
    }

    std::unordered_map<std::string, mgp::Value> source_prop_map = source_rel.Properties();
    std::unordered_map<std::string_view, mgp::Value> target_prop_map;
    for (size_t i = 0, size = source_properties.Size(); i < size; i++) {
      target_prop_map[target_properties[i].ValueString()] =
          source_prop_map[std::string(source_properties[i].ValueString())];
    }

    target_node.SetProperties(target_prop_map);

    record.Insert(kResult.data(), true);
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    record.Insert(kResult.data(), false);
  }
}

void CopyPropertyRel2Rel(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard(memory);

  auto arguments = mgp::List(args);
  auto record_factory = mgp::RecordFactory(result);
  auto record = record_factory.NewRecord();

  try {
    if (!arguments[0].IsRelationship()) {
      throw std::runtime_error("CopyPropertyRel2Rel argument source entity is not a relationship!");
    }
    if (!arguments[2].IsRelationship()) {
      throw std::runtime_error("CopyPropertyRel2Rel argument target entity is not a relationship!");
    }

    auto source_rel = arguments[0].ValueRelationship();
    auto source_properties = arguments[1].ValueList();

    auto target_rel = arguments[2].ValueRelationship();
    auto target_properties = arguments[3].ValueList();

    if (source_properties.Empty() && target_properties.Empty()) {
      record.Insert(kResult.data(), true);
      return;
    }

    if (source_properties.Size() != target_properties.Size()) {
      throw std::runtime_error("CopyPropertyRel2Rel source properties and target properties are not of the same size!");
    }

    std::unordered_map<std::string, mgp::Value> source_prop_map = source_rel.Properties();
    std::unordered_map<std::string_view, mgp::Value> target_prop_map;
    for (size_t i = 0, size = source_properties.Size(); i < size; i++) {
      target_prop_map[target_properties[i].ValueString()] =
          source_prop_map[std::string(source_properties[i].ValueString())];
    }

    target_rel.SetProperties(target_prop_map);

    record.Insert(kResult.data(), true);
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    record.Insert(kResult.data(), false);
  }
}

std::string TransformIntoSetPropertiesClause(std::string_view source_variable, mgp::List source_props,
                                             std::string_view target_variable, mgp::List target_props) {
  std::unordered_set<std::string> prop_key_set;
  std::vector<std::string> set_prop_strings;
  set_prop_strings.reserve(source_props.Size());

  for (size_t i = 0, size = source_props.Size(); i < size; i++) {
    if (prop_key_set.contains(std::string(target_props[i].ValueString()))) {
      continue;
    }

    set_prop_strings.push_back(
        fmt::format("{}: {}.{}", target_props[i].ValueString(), source_variable, source_props[i].ValueString()));

    prop_key_set.insert(std::string(target_props[i].ValueString()));
  }

  std::string sp;
  for (size_t i = 0, size = set_prop_strings.size(); i < size; i++) {
    sp += set_prop_strings[i];
    if (i < size - 1) {
      sp += ", ";
    }
  }

  return fmt::format("SET {} += {{ {} }}", target_variable, sp);
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard(memory);
    AddProcedure(CopyPropertyNode2Node, "copyPropertyNode2Node", mgp::ProcedureType::Write,
                 {mgp::Parameter(kSourceNode, mgp::Type::Node),
                  mgp::Parameter(kSourceProperties, {mgp::Type::List, mgp::Type::String}),
                  mgp::Parameter(kTargetNode, mgp::Type::Node),
                  mgp::Parameter(kTargetProperties, {mgp::Type::List, mgp::Type::String})},
                 {mgp::Return(kResult, mgp::Type::Bool)}, module, memory);

    AddProcedure(CopyPropertyNode2Rel, "copyPropertyNode2Rel", mgp::ProcedureType::Write,
                 {mgp::Parameter(kSourceNode, mgp::Type::Node),
                  mgp::Parameter(kSourceProperties, {mgp::Type::List, mgp::Type::String}),
                  mgp::Parameter(kTargetRel, mgp::Type::Relationship),
                  mgp::Parameter(kTargetProperties, {mgp::Type::List, mgp::Type::String})},
                 {mgp::Return(kResult, mgp::Type::Bool)}, module, memory);

    AddProcedure(CopyPropertyRel2Node, "copyPropertyRel2Node", mgp::ProcedureType::Write,
                 {mgp::Parameter(kSourceRel, mgp::Type::Relationship),
                  mgp::Parameter(kSourceProperties, {mgp::Type::List, mgp::Type::String}),
                  mgp::Parameter(kTargetNode, mgp::Type::Node),
                  mgp::Parameter(kTargetProperties, {mgp::Type::List, mgp::Type::String})},
                 {mgp::Return(kResult, mgp::Type::Bool)}, module, memory);

    AddProcedure(CopyPropertyRel2Rel, "copyPropertyRel2Rel", mgp::ProcedureType::Write,
                 {mgp::Parameter(kSourceRel, mgp::Type::Relationship),
                  mgp::Parameter(kSourceProperties, {mgp::Type::List, mgp::Type::String}),
                  mgp::Parameter(kTargetRel, mgp::Type::Relationship),
                  mgp::Parameter(kTargetProperties, {mgp::Type::List, mgp::Type::String})},
                 {mgp::Return(kResult, mgp::Type::Bool)}, module, memory);
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }