#include <mgp.hpp>

#include "algorithm/map.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Map::Flatten, std::string(Map::kProcedureFlatten).c_str(), mgp::ProcedureType::Read,
                 {mgp::Parameter(std::string(Map::kArgumentMapFlatten).c_str(), {mgp::Type::Map, mgp::Type::Any}),
                  mgp::Parameter(std::string(Map::kArgumentDelimiterFlatten).c_str(), {mgp::Type::String}, ".")},
                 {mgp::Return(std::string(Map::kReturnValueFlatten).c_str(), {mgp::Type::Map, mgp::Type::Any})}, module,
                 memory);
    AddProcedure(
        Map::FromLists, std::string(Map::kProcedureFromLists).c_str(), mgp::ProcedureType::Read,
        {mgp::Parameter(std::string(Map::kArgumentListKeysFromLists).c_str(), {mgp::Type::List, mgp::Type::String}),
         mgp::Parameter(std::string(Map::kArgumentListValuesFromLists).c_str(), {mgp::Type::List, mgp::Type::Any})},
        {mgp::Return(std::string(Map::kReturnListFromLists).c_str(), {mgp::Type::Map, mgp::Type::Any})}, module,
        memory);

    AddProcedure(
        Map::RemoveKeys, std::string(Map::kProcedureRemoveKeys).c_str(), mgp::ProcedureType::Read,
        {mgp::Parameter(std::string(Map::kArgumentsInputMapRemoveKeys).c_str(), mgp::Type::Map),
         mgp::Parameter(std::string(Map::kArgumentsKeysListRemoveKeys).c_str(), {mgp::Type::List, mgp::Type::String}),
         mgp::Parameter(std::string(Map::kArgumentsRecursiveRemoveKeys).c_str(), mgp::Type::Bool, false)},
        {mgp::Return(std::string(Map::kReturnRemoveKeys).c_str(), mgp::Type::Map)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
