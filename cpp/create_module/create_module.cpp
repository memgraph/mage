#include <mgp.hpp>

#include "algorithm/create.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Create::RemoveProperties, std::string(Create::kProcedureRemoveProperties).c_str(),
                 mgp::ProcedureType::Write,
                 {
                     mgp::Parameter(std::string(Create::kArgumentNodeRemoveProperties).c_str(), mgp::Type::Node),
                     mgp::Parameter(std::string(Create::kArgumentKeysRemoveProperties).c_str(),
                                    {mgp::Type::List, mgp::Type::String}),
                 },
                 {mgp::Return(std::string(Create::kReturntRemoveProperties).c_str(), mgp::Type::Node)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
