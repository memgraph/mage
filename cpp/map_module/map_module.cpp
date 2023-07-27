#include <mgp.hpp>

#include "algorithm/map.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(
        Map::RemoveKeys, std::string(Map::kProcedureRemoveKeys).c_str(), mgp::ProcedureType::Read,
        {mgp::Parameter(std::string(Map::kArgumentsInputMapRemoveKeys).c_str(), mgp::Type::Map),
         mgp::Parameter(std::string(Map::kArgumentsKeysListRemoveKeys).c_str(), {mgp::Type::List, mgp::Type::Any}),
         mgp::Parameter(std::string(Map::kArgumentsRecursiveRemoveKeys).c_str(), mgp::Type::Bool, false)},
        {mgp::Return(std::string(Map::kReturnRemoveKeys).c_str(), mgp::Type::Map)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
