#include <mgp.hpp>

#include "algorithms/pairs.hpp"

constexpr std::string_view kReturnPairs = "pairs";

constexpr std::string_view kProcedurePairs = "pairs";

constexpr std::string_view kInputList = "inputList";

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  mgp::memory = memory;
  try {
    AddProcedure(Pairs, kProcedurePairs, mgp::ProcedureType::Read,
                 {mgp::Parameter(kInputList, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(kReturnPairs, {mgp::Type::List, mgp::Type::Any})}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
