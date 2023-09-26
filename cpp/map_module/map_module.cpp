#include <mgp.hpp>

#include "algorithm/map.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    mgp::AddFunction(Map::Flatten, std::string(Map::kProcedureFlatten).c_str(),
                     {mgp::Parameter(std::string(Map::kArgumentMapFlatten).c_str(), {mgp::Type::Map, mgp::Type::Any}),
                      mgp::Parameter(std::string(Map::kArgumentDelimiterFlatten).c_str(), {mgp::Type::String}, ".")},
                     module, memory);

    mgp::AddFunction(
        Map::FromLists, std::string(Map::kProcedureFromLists).c_str(),
        {mgp::Parameter(std::string(Map::kArgumentListKeysFromLists).c_str(), {mgp::Type::List, mgp::Type::String}),
         mgp::Parameter(std::string(Map::kArgumentListValuesFromLists).c_str(), {mgp::Type::List, mgp::Type::Any})},
        module, memory);

    mgp::AddFunction(
        Map::RemoveKeys, std::string(Map::kProcedureRemoveKeys).c_str(),
        {mgp::Parameter(std::string(Map::kArgumentsInputMapRemoveKeys).c_str(), mgp::Type::Map),
         mgp::Parameter(std::string(Map::kArgumentsKeysListRemoveKeys).c_str(), {mgp::Type::List, mgp::Type::String}),
         mgp::Parameter(std::string(Map::kArgumentsRecursiveRemoveKeys).c_str(), mgp::Type::Map,
                        mgp::Value(mgp::Map()))},
        module, memory);

    mgp::AddFunction(
        Map::RemoveKey, Map::kProcedureRemoveKey,
        {mgp::Parameter(Map::kArgumentsInputMap, mgp::Type::Map), mgp::Parameter(Map::kArgumentsKey, mgp::Type::String),
         mgp::Parameter(Map::kArgumentsIsRecursive, mgp::Type::Map, mgp::Value(mgp::Map()))},
        module, memory);

    mgp::AddFunction(Map::FromPairs, Map::kProcedureFromPairs,
                     {mgp::Parameter(Map::kArgumentsInputList, {mgp::Type::List, mgp::Type::List})}, module, memory);

    mgp::AddFunction(Map::Merge, Map::kProcedureMerge,
                     {mgp::Parameter(Map::kArgumentsInputMap1, mgp::Type::Map),
                      mgp::Parameter(Map::kArgumentsInputMap2, mgp::Type::Map)},
                     module, memory);

    AddProcedure(Map::FromNodes, Map::kProcedureFromNodes, mgp::ProcedureType::Read,
                 {mgp::Parameter(Map::kFromNodesArg1, mgp::Type::String),
                  mgp::Parameter(Map::kFromNodesArg2, mgp::Type::String)},
                 {mgp::Return(Map::kResultFromNodes, mgp::Type::Map)}, module, memory);

    mgp::AddFunction(Map::FromValues, Map::kProcedureFromValues,
                     {mgp::Parameter(Map::kFromValuesArg1, {mgp::Type::List, mgp::Type::Any})}, module, memory);

    mgp::AddFunction(
        Map::SetKey, Map::kProcedureSetKey,
        {mgp::Parameter(Map::kSetKeyArg1, mgp::Type::Map), mgp::Parameter(Map::kSetKeyArg2, mgp::Type::String),
         mgp::Parameter(Map::kSetKeyArg3, mgp::Type::Any)},
        module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
