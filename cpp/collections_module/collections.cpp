#include <mgp.hpp>

#include "algorithms/max.hpp"

constexpr std::string_view kReturnMax = "max";

constexpr std::string_view kProcedureMax = "max";

constexpr std::string_view kArgumentsInputList = "input_list";

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  mgp::memory = memory;
  try {
    AddProcedure(Max, kProcedureMax, mgp::ProcedureType::Read,
                 {mgp::Parameter(kArgumentsInputList, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(kReturnMax, mgp::Type::Double)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
