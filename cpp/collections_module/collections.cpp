#include <mgp.hpp>

#include "algorithms/algorithms.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  mgp::memory = memory;
  try {
    AddProcedure(Collections::Split, Collections::kProcedureSplit, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kArgumentInputList, {mgp::Type::List, mgp::Type::Any}),
                  mgp::Parameter(Collections::kArgumentDelimiter, mgp::Type::Any)},
                 {mgp::Return(Collections::kReturnSplit, {mgp::Type::List, mgp::Type::Any})}, module, memory);

    AddProcedure(Collections::Pairs, Collections::kProcedurePairs, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kInputList, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Collections::kReturnPairs, {mgp::Type::List, mgp::Type::Any})}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
