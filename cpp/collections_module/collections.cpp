#include <mgp.hpp>

#include "algorithms/sum.hpp"

constexpr std::string_view kReturnSum = "sum";

constexpr std::string_view kProcedureSum = "sum";

constexpr std::string_view kInputList = "input_list";

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  mgp::memory = memory;
  try {
    AddProcedure(Sum, kProcedureSum, mgp::ProcedureType::Read,
                 {mgp::Parameter(kInputList, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(kReturnSum, mgp::Type::Double)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
